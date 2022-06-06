#!/usr/bin/env python3

from base64 import decode
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
        
        out = self.tanh(out)
        out = self.linear_out(out)

        return out

@register_model("rnn_transducer_old")
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
        # encoder_output = encoder_output["encoder_out"]

        logits = self.joint(
            encoder_output.unsqueeze(2),
            decoder_output.unsqueeze(1),
        )
        # logits = self.joint(
        #     encoder_output,
        #     decoder_output
        # )
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
        # dec_in = self.get_decoder_input(prev_output_tokens)

        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens)
        
        return encoder_out, decoder_out 

class LTTRNNEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embeded_tokens):
        super().__init__(None)

        self.padding_idx = 1
        self.blank_idx = 0

        self.embed = nn.Embedding(len(dictionary), args.decoder_embed_dim, padding_idx=self.blank_idx)
        self.dropout = nn.Dropout(p=args.dropout)

        self.rnn = nn.LSTMCell(args.decoder_embed_dim, args.decoder_embed_dim) if args.rnn_type == "lstm" else nn.GRUCell(args.decoder_embed_dim, args.decoder_embed_dim)
        self.linear = nn.Linear(args.decoder_embed_dim, args.joint_dim)

        self.initial_state = nn.Parameter(torch.randn(args.decoder_embed_dim))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = args.rnn_type
        self.dlayers = args.decoder_layers
        self.dunits = args.decoder_embed_dim
        self.odim = args.joint_dim
        self.vocab_size = len(dictionary)

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
        embedding = self.embed(sequence)   
        state = self.rnn.forward(embedding, prev_state)
        out = self.linear(state)
        return out, state


    def forward(self, labels):
        '''
        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, L, D)
        '''
        batch_size = labels.shape[0]
        U = labels.shape[1]
        outs = []
        state = torch.stack([self.initial_state] * batch_size).to(labels.device)
        for u in range(U+1):
            if u == 0:
                decoder_input = torch.tensor([self.blank_idx] * batch_size).to(labels.device)
            else:
                decoder_input = labels[:, u-1]
            out, state = self._forward(decoder_input, state)
            outs.append(out)
        out = torch.stack(outs, dim=1)
        return out

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
        
        self.rnn = nn.LSTM(args.input_feat, args.encoder_embed_dim, args.encoder_layers, batch_first=True, bidirectional=args.bidirectional, dropout=args.dropout)

        self.elayers = args.encoder_layers
        self.embed_dim = args.encoder_embed_dim
        self.joint_dim = args.joint_dim
        self.rnn_type = args.rnn_type
        self.bidir = args.bidirectional

        self.linear = nn.Linear(args.encoder_embed_dim * 2, args.joint_dim) if self.bidir is True else nn.Linear(args.encoder_embed_dim, args.joint_dim)
        self.padding_idx = 1

    def forward(
        self,
        src_tokens: torch.Tensor, 
        src_lengths: torch.Tensor, 
        prev_states: Optional[List[torch.Tensor]] = None):
        self.rnn.flatten_parameters()
        out = self.rnn(src_tokens)[0]
        out = self.linear(out)
        return out
        
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

@register_model_architecture(model_name="rnn_transducer_old", arch_name="rnn_transducer_old")
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
    

@register_model_architecture("rnn_transducer_old", "rnn_transducer_old_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)