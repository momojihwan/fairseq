#!/usr/bin/env python3

from dataclasses import dataclass
from email.encoders import encode_7or8bit
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from unicodedata import bidirectional
from fairseq.data.dictionary import Dictionary

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
from fairseq.models.transformer import Embedding 
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from torch import Tensor


logger = logging.getLogger(__name__)

@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None

class JointNetwork(nn.Module):
    def __init__(self, joint_dim, num_outputs):
        super(JointNetwork, self).__init__()
        self.linear = nn.Linear(joint_dim * 2, num_outputs)
        self.tanh = nn.Tanh()

    def forward(self, enc_out, pred_out):
    
        if enc_out.dim() == 3 and pred_out.dim() == 3:  # training
            seq_lens = enc_out.size(1)
            target_lens = pred_out.size(1)
            
            enc_out = enc_out.unsqueeze(2)
            pred_out = pred_out.unsqueeze(1)
            
            enc_out = enc_out.repeat(1, 1, target_lens, 1)
            pred_out = pred_out.repeat(1, seq_lens, 1, 1)
        else:
            print("a : ", enc_out.size())
            print("b : ", pred_out.size())
            assert enc_out.dim() == pred_out.dim()

        out = torch.cat((enc_out, pred_out), dim=-1)
        out = self.tanh(out)
        out = self.linear(out)
        out = F.log_softmax(out, dim=-1)
        
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
class TransformerTransducerModel(BaseFairseqModel):
    def __init__(self, args, encoder, encoder_proj, decoder, decoder_proj, joint):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_proj = encoder_proj
        self.decoder_proj = decoder_proj
        self.joint = joint

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
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        encoder_proj = nn.Linear(args.encoder_embed_dim, args.joint_dim)
        decoder_proj = nn.Linear(args.decoder_embed_dim, args.joint_dim)
        joint = JointNetwork(args.joint_dim, len(task.target_dictionary))

        return cls(args, encoder, encoder_proj, decoder, decoder_proj, joint)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (T, B, D) tensor
        encoder_output, decoder_output = net_output
        encoder_output = encoder_output["encoder_out"]
        encoder_output = self.encoder_proj(encoder_output)
        decoder_output = self.decoder_proj(decoder_output)

        logits = self.joint(
            encoder_output,
            decoder_output
        )
        logits = logits.float()

        return logits

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_tokens_length):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens)
        return encoder_out, decoder_out 

    def greedy_search(self, enc_outputs):
        """Greedy search implementation.
        Args:
            enc_outputs: Encoder output sequence. (T, D_enc)
        Returns:
            hyp: 1-best hypotheses.
        """

        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.decoder.blank_idx], dec_state=dec_state)
        cache = {}

        dec_out, state, _ = self.decoder.score(hyp, cache)
        dec_out = self.decoder_proj(dec_out)

        for t in range(enc_outputs.size(0)):    

            logp = self.joint(enc_outputs[t], dec_out)
            top_logp, pred = torch.max(logp, dim=-1)

            if pred != self.decoder.blank_idx:
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)
                hyp.dec_state = state

                dec_out, state, _ = self.decoder.score(hyp, cache)
                dec_out = self.decoder_proj(dec_out)
        return hyp

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
        self.joint_dim = args.joint_dim

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
            if i == 0:
                input_dim = args.input_feat
            else:
                input_dim = args.encoder_embed_dim
            
            rnn_layer = nn.LSTM if "lstm" in args.rnn_type else nn.GRU
            rnn = rnn_layer(
                input_dim, args.encoder_embed_dim, num_layers=1, bidirectional=args.bidirectional, batch_first=True
            )
            
            setattr(self, "%s%d" % ("birnn" if args.bidirectional else "rnn", i), rnn)

        self.dropout = nn.Dropout(p=args.dropout)
        
        self.elayers = args.encoder_layers
        self.embed_dim = args.encoder_embed_dim
        self.joint_dim = args.joint_dim
        self.rnn_type = args.rnn_type
        self.bidir = args.bidirectional
        self.linear = nn.Linear(self.embed_dim, self.joint_dim)

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
        current_states = []
        
        for layer in range(self.elayers):
            if not isinstance(src_lengths, torch.Tensor):
                src_lengths = torch.tensor(src_lengths)
            
            pack_enc_input = pack_padded_sequence(
                src_tokens, src_lengths.cpu(), batch_first=True
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

            if layer < self.elayers - 1:
                src_tokens = self.dropout(enc_out)

        enc_out = torch.tanh(enc_out)
        # enc_out = enc_out.view(enc_out.size(0), enc_out.size(1), -1)

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
    # RNN
    args.rnn_type = getattr(args, "rnn_type", "lstm")
    args.input_feat = getattr(args, "input_feat", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.bidirectional = getattr(args, "bidirectional", True)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    
    args.joint_dim = getattr(args, "joint_dim", 128)
    args.dropout = getattr(args, "dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    

@register_model_architecture("rnn_transducer", "rnn_transducer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)