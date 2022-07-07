# #!/usr/bin/env python3

# from email.encoders import encode_7or8bit
# import logging
# import math
# from typing import Dict, List, Optional, Tuple
# from pathlib import Path
# from fairseq.data.dictionary import Dictionary
# from fairseq.models.transducer.modules.label_smoothing_loss import LabelSmoothingLoss
# from fairseq.models.transducer.modules.net_utils import get_subsample
# from fairseq.models.transducer.modules.subsampling import Conv2dSubsampling
# from fairseq.models.transducer.encoder.custom_encoder import CustomEncoder
# from fairseq.models.transducer.decoder.custom_decoder import CustomDecoder
# from fairseq.models.transducer.decoder.transformer_decoder import TransformerTransducerDecoderLayer
# from fairseq.models.transducer.joint_network import JointNetwork
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fairseq import checkpoint_utils, utils
# from fairseq.data.data_utils import lengths_to_padding_mask, pad_list, subsequent_mask
# from fairseq.models import (
#     FairseqEncoder,
#     BaseFairseqModel,
#     register_model,
#     register_model_architecture,
# )
# from fairseq.models.transformer import Embedding 
# from fairseq.modules import (
#     FairseqDropout,
#     LayerNorm,
#     PositionalEmbedding,
#     TransformerEncoderLayer,
# )
# from torch import Tensor


# logger = logging.getLogger(__name__)

# @register_model("transformer_transducer")
# class TransformerTransducerModel(BaseFairseqModel):
#     def __init__(self, args, encoder, decoder, joint):
#         super().__init__()

#         self.subsample = get_subsample(
#             args, mode="asr", arch="transformer" if args.encoder_type == "custom" else "rnn-t"
#         )

#         self.encoder = encoder
#         self.decoder = decoder
#         self.joint = joint

#         self.blank.idx = self.decoder.blank_idx
#         self.padding_idx = self.decoder.padding_idx

#     @classmethod
#     def build_encoder(cls, args):
#         encoder = TransformerTransducerEncoderLayer(args)
#         pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
#         if pretraining_path is not None:
#             if not Path(pretraining_path).exists():
#                 logger.warning(
#                     f"skipped pretraining because {pretraining_path} does not exist"
#                 )
#             else:
#                 encoder = checkpoint_utils.load_pretrained_component_from_model(
#                     component=encoder, checkpoint=pretraining_path
#                 )
#                 logger.info(f"loaded pretrained encoder from: {pretraining_path}")
#         return encoder

#     @classmethod
#     def build_decoder(cls, args, task):
#         encoder = TransformerTransducerDecoderLayer(args, task.target_dictionary)
#         return encoder

#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""

#         # make sure all arguments are present in older models
#         base_architecture(args)

#         encoder = cls.build_encoder(args)
#         decoder = cls.build_decoder(args, task)
#         joint = JointNetwork(args.encoder_embed_dim, args.decoder_embed_dim, args.joint_dim, len(task.target_dictionary))

#         return cls(args, encoder, decoder, joint)

#     def get_normalized_probs(
#         self,
#         net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
#         log_probs: bool,
#         sample: Optional[Dict[str, Tensor]] = None,
#     ):
#         # net_output['encoder_out'] is a (T, B, D) tensor
#         encoder_output, decoder_output = net_output
#         encoder_output = encoder_output["encoder_out"][0]
#         decoder_output = decoder_output["encoder_out"][0]
#         encoder_output = self.encoder_proj(encoder_output)
#         decoder_output = self.decoder_proj(decoder_output)

#         logits = self.joint(
#             encoder_output.transpose(1, 0),
#             decoder_output
#         )
#         logits = logits.float()

#         return logits

#     def forward(self, src_tokens, src_lengths, prev_output_tokens, prev_output_tokens_length):
#         encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
#         decoder_out = self.decoder(src_tokens=prev_output_tokens, src_lengths=prev_output_tokens_length)
#         return encoder_out, decoder_out 

# class LTTTransformerEncoder(FairseqEncoder):
#     def __init__(self, args, dictionary, embeded_tokens):
#         super().__init__(None)


#         self.padding_idx = dictionary.pad()
#         self.blank_idx = dictionary.unk()

#         self.embed_tokens = embeded_tokens

#         self.embed_positions = (
#             PositionalEmbedding(
#                 args.max_target_positions,
#                 args.decoder_embed_dim,
#                 self.padding_idx,
#                 learned=args.decoder_learned_pos,
#             )
#             if not args.no_token_positional_embeddings
#             else None
#         )

#         # self.transformer_layers = nn.ModuleList(
#         #     [TransformerEncoderLayer(args) for _ in range(args.decoder_layers)]
#         # )

#         self.transformer_layers = nn.ModuleList(
#             [TransformerTransducerDecoderLayer(args.decoder_embed_dim, args.decoder_ffn_embed_dim, args.encoder_attention_heads, args.dropout)]
#         )
#         if args.encoder_normalize_before:
#             self.layer_norm = LayerNorm(args.decoder_embed_dim)
#         else:
#             self.layer_norm = None

#     def score(self, dec_input, cache={}):
#         """One-step forward hypothesis.
#         Args:
#             hyp: Hypothesis.
#             cache: Pairs of (dec_out, state) for each label sequence. (key)
#         Returns:
#             dec_out: Decoder output sequence. (1, D_dec)
#             new_state: Decoder hidden states. ((N, 1, D_dec), (N, 1, D_dec))
#             label: Label ID for LM. (1,)
#         """

#         x = self.embed_scale * self.embed_tokens(dec_input)

#         dec_out_mask = ~subsequent_mask(len(x), device=dec_input.device).unsqueeze_(0)


#         positions = None
#         if self.embed_positions is not None:
#             positions = self.embed_positions(dec_input)
#             x += positions

#         for layer in self.transformer_layers:
#             dec_out, _ = layer(x, dec_out_mask)
        
#         return dec_out

#         # labels = torch.tensor([hyp.yseq])
#         # x = self.embed_scale * self.embed_tokens(labels)

#         # str_labels = "_".join(list(map(str, hyp.yseq)))
        
#         # if str_labels in cache:
#         #     dec_out, dec_state = cache[str_labels]
#         # else:
#         #     dec_out_mask = ~subsequent_mask(len(hyp.yseq)).unsqueeze_(0)

#         #     positions = None
#         #     if self.embed_positions is not None:
#         #         positions = self.embed_positions(labels)
#         #         x += positions

#         #     dec_state = []
                
#         #     for layer in self.transformer_layers:
#         #         dec_out, _ = layer(x, dec_out_mask)
#         #         dec_state.append(dec_out)

#         #     if self.layer_norm is not None:
#         #         dec_out = self.layer_norm(dec_out[:, -1])

#         #     cache[str_labels] = (dec_out, dec_state)
        
#         # return dec_out[0]


#     def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
#         blank = src_tokens.new([self.blank_idx])
#         src = [y[y != self.padding_idx] for y in src_tokens]
#         src = [torch.cat([blank, y], dim=0) for y in src]
#         src_tokens = pad_list(src, self.padding_idx)
#         src_lengths = src_lengths + 1

#         x = self.embed_scale * self.embed_tokens(src_tokens)

#         encoder_padding_mask = ~lengths_to_padding_mask(src_lengths)[:, None, :].to(src_tokens.device) # (B, 1, L)
#         encoder_subsequent_mask = subsequent_mask(src_tokens.size(-1), device=src_tokens.device).unsqueeze(0)  # (1, L, L)
#         tgt_mask = encoder_padding_mask & encoder_subsequent_mask   # (B, L, L)

#         positions = None
#         if self.embed_positions is not None:
#             positions = self.embed_positions(src_tokens)
#             x += positions
#         x = self.dropout_module(x)
#         # x = x.transpose(0, 1)
        
#         encoder_states = []
        
#         for layer in self.transformer_layers:
#             x, _ = layer(x, tgt_mask)
#             if return_all_hiddens:
#                 encoder_states.append(x)

#         if self.layer_norm is not None:
#             x = self.layer_norm(x)

#         return {
#             "encoder_out": [x],  # T x B x C
#             "encoder_padding_mask": [encoder_padding_mask]
#             if encoder_padding_mask.any()
#             else [],  # B x T
#             "encoder_embedding": [],  # B x T x C
#             "encoder_states": encoder_states,  # List[T x B x C]
#             "src_tokens": [],
#             "src_lengths": [],
#         }

#     def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
#         if self.num_updates < self.encoder_freezing_updates:
#             with torch.no_grad():
#                 x = self._forward(
#                     src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
#                 )
#         else:
#             x = self._forward(
#                 src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
#             )
#         return x

#     def reorder_encoder_out(self, encoder_out, new_order):
#         new_encoder_out = (
#             []
#             if len(encoder_out["encoder_out"]) == 0
#             else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
#         )

#         new_encoder_padding_mask = (
#             []
#             if len(encoder_out["encoder_padding_mask"]) == 0
#             else [
#                 x.index_select(0, new_order)
#                 for x in encoder_out["encoder_padding_mask"]
#             ]
#         )

#         new_encoder_embedding = (
#             []
#             if len(encoder_out["encoder_embedding"]) == 0
#             else [
#                 x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
#             ]
#         )

#         encoder_states = encoder_out["encoder_states"]
#         if len(encoder_states) > 0:
#             for idx, state in enumerate(encoder_states):
#                 encoder_states[idx] = state.index_select(1, new_order)

#         return {
#             "encoder_out": new_encoder_out,  # T x B x C
#             "encoder_padding_mask": new_encoder_padding_mask,  # B x T
#             "encoder_embedding": new_encoder_embedding,  # B x T x C
#             "encoder_states": encoder_states,  # List[T x B x C]
#             "src_tokens": [],  # B x T
#             "src_lengths": [],  # B x 1
#         }

#     def set_num_updates(self, num_updates):
#         super().set_num_updates(num_updates)
#         self.num_updates = num_updates


# class S2TTransformerEncoder(FairseqEncoder):
#     def __init__(self, args):
#         super().__init__(None)

#         self.encoder_freezing_updates = args.encoder_freezing_updates
#         self.num_updates = 0

#         self.dropout_module = FairseqDropout(
#             p=args.dropout, module_name=self.__class__.__name__
#         )
#         self.embed_scale = math.sqrt(args.encoder_embed_dim)
#         if args.no_scale_embedding:
#             self.embed_scale = 1.0
#         self.padding_idx = 1

#         self.subsample = Conv1dSubsampler(
#             args.input_feat_per_channel * args.input_channels,
#             args.conv_channels,
#             args.encoder_embed_dim,
#             [int(k) for k in args.conv_kernel_sizes.split(",")],
#         )

#         self.embed_positions = PositionalEmbedding(
#             args.max_source_positions, args.encoder_embed_dim, self.padding_idx
#         )

#         self.transformer_layers = nn.ModuleList(
#             [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
#         )
#         # self.transformer_layers = nn.ModuleList(
#         #     [TransformerTransducerDecoderLayer(args.encoder_embed_dim, args.encoder_ffn_embed_dim, args.encoder_attention_heads, args.dropout)]
#         # )
#         if args.encoder_normalize_before:
#             self.layer_norm = LayerNorm(args.encoder_embed_dim)
#         else:
#             self.layer_norm = None

#     def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):

#         x, input_lengths = self.subsample(src_tokens, src_lengths)
#         x = self.embed_scale * x

#         encoder_padding_mask = lengths_to_padding_mask(input_lengths)
#         positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
#         x += positions
#         x = self.dropout_module(x)

#         encoder_states = []

#         for layer in self.transformer_layers:
#             x = layer(x, encoder_padding_mask)
#             if return_all_hiddens:
#                 encoder_states.append(x)

#         if self.layer_norm is not None:
#             x = self.layer_norm(x)

#         return {
#             "encoder_out": [x],  # T x B x C
#             "encoder_padding_mask": [encoder_padding_mask]
#             if encoder_padding_mask.any()
#             else [],  # B x T
#             "encoder_embedding": [],  # B x T x C
#             "encoder_states": encoder_states,  # List[T x B x C]
#             "input_lengths": input_lengths,
#             "src_tokens": [],
#             "src_lengths": [],
#         }

#     def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
#         if self.num_updates < self.encoder_freezing_updates:
#             with torch.no_grad():
#                 x = self._forward(
#                     src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
#                 )
#         else:
#             x = self._forward(
#                 src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
#             )
#         return x

#     def reorder_encoder_out(self, encoder_out, new_order):
#         new_encoder_out = (
#             []
#             if len(encoder_out["encoder_out"]) == 0
#             else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
#         )

#         new_encoder_padding_mask = (
#             []
#             if len(encoder_out["encoder_padding_mask"]) == 0
#             else [
#                 x.index_select(0, new_order)
#                 for x in encoder_out["encoder_padding_mask"]
#             ]
#         )

#         new_encoder_embedding = (
#             []
#             if len(encoder_out["encoder_embedding"]) == 0
#             else [
#                 x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
#             ]
#         )

#         encoder_states = encoder_out["encoder_states"]
#         if len(encoder_states) > 0:
#             for idx, state in enumerate(encoder_states):
#                 encoder_states[idx] = state.index_select(1, new_order)

#         return {
#             "encoder_out": new_encoder_out,  # T x B x C
#             "encoder_padding_mask": new_encoder_padding_mask,  # B x T
#             "encoder_embedding": new_encoder_embedding,  # B x T x C
#             "encoder_states": encoder_states,  # List[T x B x C]
#             "src_tokens": [],  # B x T
#             "src_lengths": [],  # B x 1
#         }

#     def set_num_updates(self, num_updates):
#         super().set_num_updates(num_updates)
#         self.num_updates = num_updates



# @register_model_architecture(model_name="transformer_transducer", arch_name="transformer_transducer")
# def base_architecture(args):
#     args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
#     # Convolutional subsampler
#     args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
#     args.conv_channels = getattr(args, "conv_channels", 1024)
#     # Transformer
#     args.encoder_type = getattr(args, "encoder_type", "custom")
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
#     args.encoder_layers = getattr(args, "encoder_layers", 6)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)

#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
#     args.decoder_ffn_embed_dim = getattr(
#         args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
#     )
#     args.decoder_layers = getattr(args, "decoder_layers", 2)
#     # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
#     # args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
#     args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

#     args.dropout = getattr(args, "dropout", 0.1)
#     # args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
#     # args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
#     args.activation_fn = getattr(args, "activation_fn", "relu")
#     args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
#     args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
#     # args.share_decoder_input_output_embed = getattr(
#     #     args, "share_decoder_input_output_embed", False
#     # )
#     args.no_token_positional_embeddings = getattr(
#         args, "no_token_positional_embeddings", False
#     )

#     # loss condition
#     args.use_auxiliary = getattr(args, "use_auxiliary", True)
#     args.use_ctc = getattr(args, "use_ctc", False)
#     args.use_lm = getattr(args, "use_lm", False)
#     args.use_kl_div = getattr(args, "use_kl_div", False)

#     # use Auxiliary loss
#     args.aux_trans_loss_enc_out_layers = getattr(args, "aux_transducer_loss_enc_out_layers", [0])
#     args.aux_mlp_dim = getattr(args, "aux_mlp_dim", 128)

#     args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    


# @register_model_architecture("transformer_transducer", "transformer_transducer_s")
# def s2t_transformer_s(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
#     args.dropout = getattr(args, "dropout", 0.1)
#     base_architecture(args)