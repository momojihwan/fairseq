# from email.encoders import encode_7or8bit
# from typing import List, Tuple, Union

# import torch

# from fairseq.models.transducer.modules.blocks import build_blocks, get_pos_enc_and_att_class
# from fairseq.models.transducer.modules.vgg2l import VGG2L
# from fairseq.models.transducer.modules.layer_norm import LayerNorm
# from fairseq.models.transducer.modules.subsampling import Conv2dSubsampling

# class CustomEncoder(torch.nn.Module):
#     """Custom encoder module for transducer models.

#     Args:
#         idim: Input dimension.
#         enc_arch: Encoder block architecture (type and parameters).
#         input_layer: Input layer type.
#         repeat_block: Number of times blocks_arch is repeated.
#         self_attn_type: Self-attention type.
#         positional_encoding_type: Positional encoding type.
#         positionwise_layer_type: Positionwise layer type.
#         positionwise_activation_type: Positionwise activation type.
#         conv_mod_activation_type: Convolutional module activation type.
#         aux_enc_output_layers: Layer IDs for auxiliary encoder output sequences.
#         input_layer_dropout_rate: Dropout rate for input layer.
#         input_layer_pos_enc_dropout_rate: Dropout rate for input layer pos. enc.
#         padding_idx: Padding symbol ID for embedding layer.

#     """

#     def __init__(
#         self,
#         args
#     ):

#        super().__init__()

#        (
#            self.embed,
#            self.encoders,
#            self.enc_out,
#            self.conv_subsampling_factor
#        ) = build_blocks(
#            "encoder",
#             args
#        )

#        self.after_norm = LayerNorm(self.enc_out)
#        self.n_blocks = len(args.enc_arch) * args.repeat_block

#     def forward(
#         self,
#         feats: torch.Tensor,
#         mask: torch.Tensor,
#     ):

#         if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
#             enc_out, mask = self.embed(feats, mask)
#         else:
#             enc_out = self.embed(feats)

#         enc_out, mask = self.encoders(enc_out, mask)

#         if isinstance(enc_out, tuple):
#             enc_out = enc_out[0]

#         enc_out = self.after_norm(enc_out)  

#         return enc_out, mask