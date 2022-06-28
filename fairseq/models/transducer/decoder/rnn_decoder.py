import logging
import math
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple, Union, Any
from fairseq.models.transducer.modules.time_reduction import TimeReduction
from fairseq.models import (
    FairseqEncoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

logger = logging.getLogger(__name__)

class CustomRNNDecoder(nn.Module):
    def __init__(self, args, dictionary):
        super(CustomRNNDecoder, self).__init__()
        self.embed = nn.Embedding(len(dictionary), args.decoder_embed_dim)
        self.rnn = nn.GRUCell(input_size=args.decoder_embed_dim,
                              hidden_size=args.decoder_embed_dim)
        self.linear = nn.Linear(args.decoder_embed_dim, args.joint_dim)

        self.initial_state = nn.Parameter(torch.randn(args.decoder_embed_dim))
        self.start_symbol = dictionary.bos()
        self.blank_idx = dictionary.bos()
        self.padding_idx = dictionary.pad()

    def forward_one_step(self, input, prev_state):
        embedding = self.embed(input)
        state = self.rnn.forward(embedding, prev_state)
        out = self.linear(state)
        return out, state

    def forward(self, src_tokens, src_lengths):
        batch_size = src_tokens.shape[0]
        U = src_tokens.shape[1]
        outs = []
        state = torch.stack([self.initial_state] * batch_size).to(src_tokens.device)
        for u in range(U+1):
            if u == 0:
                decoder_input = torch.tensor([self.start_symbol] * batch_size).to(src_tokens.device)
            else:
                decoder_input = src_tokens[:,u-1]
                
            out, state = self.forward_one_step(decoder_input, state)
            outs.append(out)
        out = torch.stack(outs, dim=1)
        return out

class ESPnet_RNNDecoder(FairseqEncoder):
    def __init__(self, args, dictionary, embeded_tokens):
        super().__init__(None)

        self.padding_idx = dictionary.pad()
        self.blank_idx = dictionary.bos()

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
        self.vocab_size = len(dictionary)

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
