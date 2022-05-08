#!/usr/bin/env python3

from dataclasses import dataclass
from email.encoders import encode_7or8bit
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from unicodedata import bidirectional
from fairseq.data.dictionary import Dictionary
from fairseq.models.transducer.modules import label_smoothing_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask, pad_list, subsequent_mask
from fairseq.models import (
    FairseqEncoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transducer.modules.label_smoothing_loss import LabelSmoothingLoss
from fairseq.models.transducer.modules.net_utils import get_subsample
from fairseq.models.transducer.modules.subsampling import Conv2dSubsampling
from fairseq.models.transformer import Embedding 
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from torch import Tensor
from warp_rnnt import rnnt_loss

logger = logging.getLogger(__name__)


class JointNetwork(nn.Module):
    def __init__(
        self,
        enc_embed_dim,
        dec_embed_dim,
        joint_dim,
        num_outputs,
    ):
        super(JointNetwork, self).__init__()
        self.linear_enc = nn.Linear(enc_embed_dim, joint_dim)
        self.linear_dec = nn.Linear(dec_embed_dim, joint_dim)
        self.linear_out = nn.Linear(joint_dim, num_outputs)
        self.tanh = nn.Tanh()

    def forward(self, enc_out, pred_out, is_aux: bool=False):
    
        if enc_out.dim() == 3 and pred_out.dim() == 3:  # training
            seq_lens = enc_out.size(1)
            target_lens = pred_out.size(1)
            
            enc_out = enc_out.unsqueeze(2)
            pred_out = pred_out.unsqueeze(1)
            
            enc_out = enc_out.repeat(1, 1, target_lens, 1)
            pred_out = pred_out.repeat(1, seq_lens, 1, 1)
        else:
            assert enc_out.dim() == pred_out.dim()
        
        if is_aux:
            out = self.tanh(enc_out + self.linear_dec(pred_out))
        else:
            out = self.tanh(self.linear_enc(enc_out) + self.linear_dec(pred_out))
        
        out = self.linear_out(out)

        return out

class Conv1dSubsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)

@register_model("rnn_transducer")
class RNNTransducerModel(BaseFairseqModel):
    def __init__(self, args, encoder, decoder, joint, ctc_proj, lm_proj, LabelSmoothingLoss, aux_mlp, kl_div):
        super().__init__()

        self.blank_idx = 0
        self.padding_idx = 1

        # transducer
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
    
        # ctc
        self.ctc_proj = ctc_proj

        # lm
        self.lm_proj = lm_proj
        self.label_smoothing_loss = LabelSmoothingLoss

        # aux
        self.use_auxiliary = args.use_auxiliary
        self.aux_mlp = aux_mlp

        # symm_kl_div
        self.use_symm_kl_div = args.use_symm_kl_div
        self.kl_div = kl_div

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freezing-updates",
            type=int,
            metavar="N",
            help="freeze encoder for first N updates",
        )
        parser.add_argument(
            "--no-finetuning",
            action="store_true",
            help="if True, dont finetune models",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2TRNNEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        encoder = LTTRNNEncoder(args, task.target_dictionary, embed_tokens)
        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        # Transducer
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        joint = JointNetwork(args.encoder_embed_dim, args.decoder_embed_dim, args.joint_dim, len(task.target_dictionary))

        # CTC
        ctc_proj = nn.Linear(args.encoder_embed_dim, len(task.target_dictionary))

        # LM
        lm_proj = nn.Linear(args.decoder_embed_dim, len(task.target_dictionary))
        smoothing_loss = LabelSmoothingLoss(len(task.target_dictionary), 1, 0.1, normalize_length=False)

        # Auxiliary
        aux_mlp = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.aux_mlp_dim),
            nn.LayerNorm(args.aux_mlp_dim),
            nn.Dropout(p=0),
            nn.ReLU(),
            nn.Linear(args.aux_mlp_dim, args.joint_dim)
        )

        # KL div
        kl_div = nn.KLDivLoss(reduction="sum")

        return cls(args, encoder, decoder, joint, ctc_proj, lm_proj, smoothing_loss, aux_mlp, kl_div)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (T, B, D) tensor
        encoder_output, decoder_output = net_output
        encoder_output = encoder_output["encoder_out"]

        logits = self.joint(
            encoder_output,
            decoder_output
        )
        logits = F.log_softmax(logits, dim=-1)
        logits = logits.float()

        return logits

    def get_ctc_normalized_probs(
        self,
        net_output,
    ):
        encoder_output, _ = net_output
        encoder_output = encoder_output["encoder_out"]
        
        ctc_lin = self.ctc_proj(
            encoder_output
        )
        
        ctc_logp = F.log_softmax(ctc_lin.transpose(0, 1), dim=-1)
        
        return ctc_logp
    
    def get_lm_normalized_probs(
        self,
        net_output,
        targets
    ):
        _, decoder_output = net_output
        
        lm_lin = self.lm_proj(decoder_output)
        lm_loss = self.label_smoothing_loss(lm_lin, targets)
        return lm_loss

    def get_auxiliary_normalized_probs(
        self,
        aux_enc_out,
        dec_out,
        joint_out,
        target,
        aux_t_len,
        u_len
    ):
        aux_trans_loss = 0
        symm_kl_div_loss = 0

        B, T, U, D = joint_out.shape

        num_aux_layers = len(aux_enc_out)

        for p in self.joint.parameters():
            p.requires_grad = False
        
        for i, aux_enc_out_i in enumerate(aux_enc_out):
            aux_mlp = self.aux_mlp(aux_enc_out_i)

            aux_joint_out = F.log_softmax(self.joint(
                aux_mlp,
                dec_out,
                is_aux=True
            ), dim=-1)
            if self.use_auxiliary:
                aux_trans_loss += (
                    rnnt_loss(
                        aux_joint_out,
                        target.int(),
                        frames_lengths=aux_t_len[i].int(),
                        labels_lengths=u_len.int(),
                        reduction="sum",
                        blank=self.blank_idx
                    ) / B
                )
                
            if self.use_symm_kl_div:
                denom = B * T * U

                kl_main_aux = (
                    self.kl_div(
                        F.log_softmax(joint_out, dim=-1),
                        F.softmax(aux_joint_out, dim=-1)
                    ) / denom
                )
                kl_aux_main = (
                    self.kl_div(
                        F.log_softmax(aux_joint_out, dim=-1),
                        F.softmax(joint_out, dim=-1)
                    ) / denom
                )

                symm_kl_div_loss += kl_main_aux + kl_aux_main
            
        for p in self.joint.parameters():
            p.requires_grad = True
        
        aux_trans_loss /= num_aux_layers

        if self.use_symm_kl_div:
            symm_kl_div_loss /= num_aux_layers
        
        return aux_trans_loss, symm_kl_div_loss

    def get_decoder_input(self, labels):
        device = labels.device

        labels_unbpad = [label[label != self.padding_idx] for label in labels]
        blank = labels[0].new([self.blank_idx])

        decoder_input = pad_list(
            [torch.cat([blank, label], dim=0) for label in labels_unbpad], self.blank_idx
        ).to(device)

        return decoder_input

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_tokens_length):
        dec_in = self.get_decoder_input(prev_output_tokens)

        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(dec_in)
        return encoder_out, decoder_out 

class LTTRNNEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embeded_tokens):
        super().__init__(None)

        self.padding_idx = 1
        self.blank_idx = 0

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0
        self.embed = nn.Embedding(len(dictionary), args.decoder_embed_dim, padding_idx=self.blank_idx)
        self.dropout = nn.Dropout(p=args.dropout)

        rnn_layer = nn.LSTM if args.rnn_type == "lstm" else nn.GRU

        self.rnn = nn.ModuleList(
            [rnn_layer(args.decoder_embed_dim, args.decoder_embed_dim, 1, batch_first=True)]
        )

        for _ in range(1, args.decoder_layers):
            self.rnn += [rnn_layer(args.decoder_embed_dim, args.decoder_embed_dim, 1, batch_first=True)]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = args.rnn_type
        self.dlayers = args.decoder_layers
        self.dunits = args.decoder_embed_dim
        self.odim = args.joint_dim

    def init_state(self, batch_size):
        h_n = torch.zeros(
            self.dlayers,
            batch_size,
            self.dunits,
            device=self.device,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                batch_size,
                self.dunits,
                device=self.device,
            )

            return (h_n, c_n)

        return (h_n, None)

    def score(self, hyp, cache={}):
        """One-step forward hypothesis.
        Args:
            hyp: Hypothesis.
            cache: Pairs of (dec_out, state) for each label sequence. (key)
        Returns:
            dec_out: Decoder output sequence. (1, D_dec)
            new_state: Decoder hidden states. ((N, 1, D_dec), (N, 1, D_dec))
            label: Label ID for LM. (1,)
        """
        label = torch.full((1, 1), hyp.yseq[-1], dtype=torch.long, device=self.device)

        str_labels = "_".join(list(map(str, hyp.yseq)))

        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            dec_emb = self.embed(label)

            dec_out, dec_state = self._forward(dec_emb, hyp.dec_state)
            cache[str_labels] = (dec_out, dec_state)
        
        return dec_out[0][0], dec_state, label[0]



    def _forward(self, sequence, prev_state):
        """"
        Args:
            sequence : (B, L)
            prev_state : (B, D)
        
        Returns:
            sequence: RNN output sequences. (B, D)
            (h_next, c_next): Decoder hidden states. (N, B, D), (N, B, D)

        """
        
        h_prev, c_prev = prev_state
        h_next, c_next = self.init_state(sequence.size(0))

        for layer in range(self.dlayers):
            if self.dtype == "lstm":
                sequence, (
                    h_next[layer : layer + 1],
                    c_next[layer : layer + 1],
                ) = self.rnn[layer](
                    sequence, hx=(h_prev[layer : layer + 1], c_prev[layer : layer + 1])
                )
            else:
                sequence, h_next[layer : layer + 1] = self.rnn[layer](
                    sequence, hx=h_prev[layer : layer + 1]
                )
            
        return sequence, (h_next, c_next)


    def forward(self, labels):
        '''
        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, L, D)
        '''

        init_state = self.init_state(labels.size(0))
        dec_embed = self.embed(labels)
        
        dec_out, _ = self._forward(dec_embed, init_state)

        return dec_out

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class S2TRNNEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        for i in range(args.encoder_layers):

            # if i == 0:
            #     input_dim = args.input_feat
            # else:
            #     input_dim = args.encoder_embed_dim
            input_dim = args.encoder_embed_dim
            rnn_layer = nn.LSTM if "lstm" in args.rnn_type else nn.GRU
            rnn = rnn_layer(
                input_dim, args.encoder_embed_dim, num_layers=1, bidirectional=args.bidirectional, batch_first=True
            )
            
            setattr(self, "%s%d" % ("birnn" if args.bidirectional else "rnn", i), rnn)

        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0

        self.dropout = nn.Dropout(p=args.dropout)
        
        self.elayers = args.encoder_layers
        self.embed_dim = args.encoder_embed_dim
        self.joint_dim = args.joint_dim
        self.rnn_type = args.rnn_type
        self.bidir = args.bidirectional

        # self.subsample = get_subsample(args, mode="asr", arch="rnn-t")
        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")]
        )

        self.linear = nn.Linear(args.encoder_embed_dim, args.aux_mlp_dim)

        if args.use_auxiliary:
            self.aux_out_layers = self.valid_aux_enc_out_layers(
                                    args.aux_trans_loss_enc_out_layers,
                                    self.elayers-1,
                                    args.use_symm_kl_div,
                                    )
        else:
            self.aux_out_layers = []

        self.padding_idx = 1

    def _forward(
        self, 
        src_tokens: torch.Tensor, 
        src_lengths: torch.Tensor, 
        prev_states: Optional[List[torch.Tensor]] = None):
        '''
        Args:
            src_tokens : RNN input sequences. (B, L, D)
            src_lengths : RNN input sequences lengths. (B,)
            prev_states : RNN hidden states. [N x (B, L, D)]

        Returns:
            enc_out : RNN output sequences. (B, L, D)
            enc_out_lengths : RNN output sequences lengths. (B,)
            current_States : RNN hidden states. [N x (B, L, D)]

        '''
        
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x        
        x = x.permute(1, 0, 2)
        # x = src_tokens
        # input_lengths = src_lengths
        aux_rnn_outputs = []
        aux_rnn_lens = []
        current_states = []
        
        for layer in range(self.elayers):
            if not isinstance(input_lengths, torch.Tensor):
                input_lengths = torch.tensor(input_lengths)
            
            pack_enc_input = pack_padded_sequence(
                x, input_lengths.cpu(), batch_first=True
            )

            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))

            if isinstance(rnn, (nn.LSTM, nn.GRU)):
                rnn.flatten_parameters()

            if prev_states is not None and rnn.bidirectional:
                prev_states = reset_backward_rnn_state(prev_states)
            pack_enc_output, states = rnn(
                pack_enc_input, hx=None if prev_states is None else prev_states[layer]
            )
            current_states.append(states)

            enc_out, enc_len = pad_packed_sequence(pack_enc_output, batch_first=True)

            if self.bidir:
                enc_out = (
                    enc_out[:, :, : self.embed_dim] + enc_out[:, :, self.embed_dim :]
                )

            if layer in self.aux_out_layers:
                aux_rnn_outputs.append(torch.tanh(enc_out))
                aux_rnn_lens.append(enc_len)

            if layer < self.elayers - 1:
                x = self.dropout(enc_out)

        enc_out = torch.tanh(enc_out)
        # enc_out = enc_out.view(enc_out.size(0), enc_out.size(1), -1)

        if aux_rnn_outputs:
            return {
                "encoder_out": enc_out,  # (B, L, D)
                "src_lengths": input_lengths,
                "encoder_states": current_states,  # List[T x B x C]    
                "aux_rnn_out": aux_rnn_outputs,
                "aux_rnn_lens": aux_rnn_lens
            }
        else:
            return {
                "encoder_out": enc_out,  # (B, L, D)
                "src_lengths": enc_len,
                "encoder_states": current_states,  # List[T x B x C]
            }

    def forward(
        self,
        src_tokens: torch.Tensor, 
        src_lengths: torch.Tensor, 
        prev_states: Optional[List[torch.Tensor]] = None):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, prev_states
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, prev_states
            )
        return x
        
    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


    def valid_aux_enc_out_layers(
        self,
        aux_layer_id: List[int],
        enc_num_layers: int,
        use_symm_kl_div_loss: bool,
    ) -> List[int]:
        """Check whether provided auxiliary encoder layer IDs are valid.
        Return the valid list sorted with duplicates removed.
        Args:
            aux_layer_id: Auxiliary encoder layer IDs.
            enc_num_layers: Number of encoder layers.
            use_symm_kl_div_loss: Whether symmetric KL divergence loss is used.
            subsample: Subsampling rate per layer.
        Returns:
            valid: Valid list of auxiliary encoder layers.
        """
        if (
            not isinstance(aux_layer_id, list)
            or not aux_layer_id
            or not all(isinstance(layer, int) for layer in aux_layer_id)
        ):
            raise ValueError(
                "aux-transducer-loss-enc-output-layers option takes a list of layer IDs."
                " Correct argument format is: '[0, 1]'"
            )

        sorted_list = sorted(aux_layer_id, key=int, reverse=False)
        valid = list(filter(lambda x: 0 <= x < enc_num_layers, sorted_list))

        if sorted_list != valid:
            raise ValueError(
                "Provided argument for aux-transducer-loss-enc-output-layers is incorrect."
                " IDs should be between [0, %d]" % enc_num_layers
            )



        return valid

def reset_backward_rnn_state(
        states: Union[torch.Tensor, List[Optional[torch.Tensor]]]
    ):
        """Set backward BRNN states to zeroes.
    Args:
        states: Encoder hidden states.
    Returns:
        states: Encoder hidden states with backward set to zero.
    """
        if isinstance(states, list):
            for state in states:
                state[1::2] = 0.0
        else:
            states[1::2] = 0.0

        return states    

@register_model_architecture(model_name="rnn_transducer", arch_name="rnn_transducer")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # RNN
    args.rnn_type = getattr(args, "rnn_type", "lstm")
    args.input_feat = getattr(args, "input_feat", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.bidirectional = getattr(args, "bidirectional", False)
    args.subsample = getattr(args, "subsample", "3")

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    
    args.joint_dim = getattr(args, "joint_dim", 128)
    args.dropout = getattr(args, "dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)

    # use Auxiliary loss
    args.use_auxiliary = getattr(args, "use_auxiliary", True)
    args.aux_trans_loss_enc_out_layers = getattr(args, "aux_transducer_loss_enc_out_layers", [0])
    args.aux_mlp_dim = getattr(args, "aux_mlp_dim", 128)

    # use Symmetric KL divergence loss
    args.use_symm_kl_div = getattr(args, "use_symm_kl_div", True)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    

@register_model_architecture("rnn_transducer", "rnn_transducer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)