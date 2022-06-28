import logging
import math
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple, Union, Any
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fairseq.models.transducer.modules.time_reduction import TimeReduction
from fairseq.models import (
    FairseqEncoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

logger = logging.getLogger(__name__)

class CustomRNNEncoder(nn.Module):
    def __init__(self, args):
        super(CustomRNNEncoder, self).__init__()
        
        self.time_reduction = TimeReduction(args.time_reduction_stride)

        if args.use_time_reduction:
            self.input_size = args.input_feat * args.time_reduction_stride
            
        else:
            self.input_size = args.input_feat

        self.rnn = nn.GRU(input_size=self.input_size,
                            hidden_size=args.encoder_embed_dim,
                            num_layers=args.encoder_layers,
                            batch_first=True,
                            bidirectional=args.bidirectional,
                            dropout=args.dropout)
        
        if args.bidirectional is True:
            self.linear = nn.Linear(args.encoder_embed_dim * 2, args.joint_dim)
        else:
            self.linear = nn.Linear(args.encoder_embed_dim, args.joint_dim)

        self.use_time_reduction = args.use_time_reduction

    def forward(self, src_tokens, src_lengths):
        self.rnn.flatten_parameters()
        
        if self.use_time_reduction:
            src_tokens, src_lengths = self.time_reduction(src_tokens, src_lengths)
         
        # x = x.permute(0, 2, 1)
        out = self.rnn(src_tokens)[0]
        out = self.linear(out)
        return out

class ESPnet_RNNEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(None)

        self.time_reduction = TimeReduction(args.time_reduction_stride)

        self.elayers = args.encoder_layers
        self.embed_dim = args.encoder_embed_dim
        self.joint_dim = args.joint_dim
        self.rnn_type = args.rnn_type
        self.bidir = args.bidirectional
        
        for i in range(args.encoder_layers):

            if args.use_time_reduction:
                if i == 0:
                    input_dim = args.input_feat * args.time_reduction_stride
                else:
                    input_dim = args.encoder_embed_dim
            else:
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
        

        # self.subsample = get_subsample(args, mode="asr", arch="rnn-t")
        # self.subsample = Conv1dSubsampler(
        #     args.input_feat_per_channel * args.input_channels,
        #     args.conv_channels,
        #     args.encoder_embed_dim,
        #     [int(k) for k in args.conv_kernel_sizes.split(",")]
        # )

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
        
        # x, input_lengths = self.subsample(src_tokens, src_lengths)
        # x = self.embed_scale * x        
        # x = x.permute(1, 0, 2)

        time_reduction_out, time_reduction_lengths = self.time_reduction(src_tokens, src_lengths)

        aux_rnn_outputs = []
        aux_rnn_lens = []
        current_states = []
        
        for layer in range(self.elayers):
            if not isinstance(time_reduction_lengths, torch.Tensor):
                time_reduction_lengths = torch.tensor(time_reduction_lengths)
            
            pack_enc_input = pack_padded_sequence(
                time_reduction_out, time_reduction_lengths.cpu(), batch_first=True
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
                time_reduction_out = self.dropout(enc_out)

        enc_out = torch.tanh(enc_out)
        # enc_out = enc_out.view(enc_out.size(0), enc_out.size(1), -1)

        if aux_rnn_outputs:
            return {
                "encoder_out": enc_out,  # (B, L, D)
                "src_lengths": time_reduction_lengths,
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
