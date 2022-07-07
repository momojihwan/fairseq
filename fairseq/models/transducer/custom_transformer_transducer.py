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
# from fairseq.models.transducer.joint_network import JointNetwork
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fairseq import checkpoint_utils, utils
# from fairseq.models import (
#     FairseqEncoder,
#     BaseFairseqModel,
#     register_model,
#     register_model_architecture,
# )

# from torch import Tensor


# logger = logging.getLogger(__name__)

# @register_model("custom_transformer_transducer")
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
#         encoder = CustomEncoder(args)
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
#         encoder = CustomDecoder(args, task.target_dictionary)
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

# @register_model_architecture(model_name="custom_transformer_transducer", arch_name="custom_transformer_transducer")
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
    


# @register_model_architecture("custom_transformer_transducer", "custom_transformer_transducer_s")
# def s2t_transformer_s(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
#     args.dropout = getattr(args, "dropout", 0.1)
#     base_architecture(args)