#!/usr/bin/env python3

from dataclasses import dataclass
from email.encoders import encode_7or8bit
import logging
import math
import sre_compile
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from unicodedata import bidirectional
from fairseq.data.dictionary import Dictionary
from fairseq.models.transducer.modules import label_smoothing_loss
from fairseq.models.transducer.modules.time_reduction import TimeReduction

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
from fairseq.models.transducer.encoder.rnn_encoder import CustomRNNEncoder, ESPnet_RNNEncoder
from fairseq.models.transducer.decoder.rnn_decoder import CustomRNNDecoder
from fairseq.models.transducer.joint_network import CustomJointNetwork
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

@dataclass
class Sequence(object):
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = [] # predictions of phoneme language model
            self.k = [blank] # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = None
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp


@register_model("custom_rnn_transducer")
class CustomRNNTransducerModel(BaseFairseqModel):
    def __init__(self, args, encoder, decoder, joint):
        super().__init__()

        # transducer
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint

        self.blank_idx = self.decoder.blank_idx
        self.padding_idx = self.decoder.padding_idx

    @classmethod
    def build_encoder(cls, args):
        encoder = CustomRNNEncoder(args)
        # encoder = ESPnet_RNNEncoder(args)
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
    def build_decoder(cls, args, task):
        encoder = CustomRNNDecoder(args, task.target_dictionary)
        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        # Transducer
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task)
        joint = CustomJointNetwork(args.joint_dim, len(task.target_dictionary))

        return cls(args, encoder, decoder, joint)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (T, B, D) tensor
        encoder_output, decoder_output = net_output
        
        logits = self.joint(
            encoder_output.unsqueeze(2),
            decoder_output.unsqueeze(1)
        )
        logits = F.log_softmax(logits, dim=-1)

        return logits

    def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_tokens_length):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, prev_output_tokens_length)
        return encoder_out, decoder_out 
    
    def greedy_search(self, src_tokens, src_lengths):
        y_batch = []
        B = len(src_tokens)
        enc_out = self.encoder.forward(src_tokens, src_lengths)
        U_max = 300
        
        for b in range(B):
            t = 0; u = 0;
            y = [self.decoder.start_symbol]

            pred_state = self.decoder.initial_state.unsqueeze(0)
            
            while t < src_lengths[b] and u < U_max:

                pred_input = torch.tensor([y[-1]]).to(src_tokens.device)
                g_u, pred_state = self.decoder.forward_one_step(pred_input, pred_state)
                f_t = enc_out[b, t]
                h_t_u = self.joint.forward(f_t, g_u)
                
                argmax = h_t_u.max(-1)[1].item()
                
                if argmax == 0:
                    t += 1
                else:    # argmax is a label
                    u += 1
                    y.append(argmax)
                 
            y_batch.append(y[1:])    # except start symbol
        
        return y_batch

    def beam_search(self, src_tokens, src_lengths, beam_size=2):
        B = [Sequence(blank=0)]
        batch_size = len(src_tokens)
        enc_out = self.encoder.forward(src_tokens, src_lengths)
        # enc_out = enc_out.squeeze() # batch_size 1 일 때 코드 

        for b in range(batch_size):
            print("b : ", b)
            for i, f_t in enumerate(enc_out[b]):
        
                sorted(B, key=lambda a: len(a.k), reverse=True)
                A = B       # Hypotheses that has emitted null from frame 't' and are now in 't+1'
                B = []      # Hypotheses that has not yet emitted null from frame 't' and so can continue emitting more symbols from 't'

                pred_state = self.decoder.initial_state.unsqueeze(0)
                
                while True:
                    y_hat = max(A, key=lambda a: a.logp) # y^ most probable in A
                    
                    A.remove(y_hat) # remove y^ from A
                    
                    pred_input = torch.tensor([y_hat.k[-1]]).to(src_tokens.device)

                    g_u, pred_state = self.decoder.forward_one_step(pred_input, y_hat.h)

                    h_t_u = self.joint(f_t, g_u[0]) # g_u -> [120(out_dim)] , h_t_u -> [29(vocab)]
        
                    logp = F.log_softmax(h_t_u, dim=0)  # pr(y^) = ~~~~

                    for k in range(len(logp)):
                        yk = Sequence(y_hat)
                        
                        yk.logp += float(logp[k]) # pr(y^+k) = ~~~~
                        
                        if k == 0:
                            B.append(yk) # add y^ to B     
                            continue
                        
                        yk.h = pred_state; yk.k.append(k);

                        A.append(yk)    # add y^ + k to A
                    
                    y_hat = max(A, key=lambda a: a.logp)   # elements most probable in A
                    
                    yb = max(B, key=lambda a: a.logp)   # elements most probable in B
                    
                    if len(B) >= beam_size and yb.logp >= y_hat.logp: break

                sorted(B, key=lambda a: a.logp, reverse=True)
        
            print("B : ", [(B[0].k)[1:]])
            exit()
            
        return [(B[0].k)[1:]]

@register_model_architecture(model_name="custom_rnn_transducer", arch_name="custom_rnn_transducer")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Time Reduction
    # args.time_reduction_input_dim = getattr(args, "time_reduction_input_dim", )
    args.use_time_reduction = getattr(args, "use_time_reduction", False)
    args.time_reduction_stride = getattr(args, "time_reduction_stride", 4)
    # RNN
    args.rnn_type = getattr(args, "rnn_type", "lstm")
    args.input_feat = getattr(args, "input_feat", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.bidirectional = getattr(args, "bidirectional", True)
    
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    
    args.joint_dim = getattr(args, "joint_dim", 128)
    args.dropout = getattr(args, "dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)

    # loss condition
    args.use_auxiliary = getattr(args, "use_auxiliary", False)
    args.use_ctc = getattr(args, "use_ctc", False)
    args.use_lm = getattr(args, "use_lm", False)
    args.use_kl_div = getattr(args, "use_kl_div", False)

    # use Auxiliary loss
    args.aux_trans_loss_enc_out_layers = getattr(args, "aux_transducer_loss_enc_out_layers", [0])
    args.aux_mlp_dim = getattr(args, "aux_mlp_dim", 128)

    # use Symmetric KL divergence loss
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    

@register_model_architecture("custom_rnn_transducer", "custom_rnn_transducer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)