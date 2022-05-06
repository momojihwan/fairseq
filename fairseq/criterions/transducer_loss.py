# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
import math
from ntpath import join
from re import U
import numpy as np
from dataclasses import dataclass
from logging import logProcesses

import torch
import torch.nn.functional as F

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from fairseq import metrics, utils
from fairseq.data.data_utils import pad_list
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from warp_rnnt import rnnt_loss


@dataclass
class TransducerCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

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


@register_criterion("transducer", dataclass=TransducerCriterionConfig)
class Transducer(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.padding_idx = task.tgt_dict.pad() 
        self.bos_idx = task.tgt_dict.bos()
        self.blank_idx = 0

        self.ctc_loss = torch.nn.CTCLoss(
            blank=self.blank_idx,
            reduction="none",
            zero_infinity=True
        )

        self.trans_loss_weight = 1.0
        self.ctc_loss_weight = 0.5
        self.aux_trans_loss_weight = 0.2
        self.symm_kl_div_loss_weight = 0.2
        self.lm_loss_weight = 0.5

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        net_output = model(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"], sample["net_input"]["prev_output_tokens"], sample["target_lengths"])
        target, lm_loss_target, t_len, aux_t_len, u_len = self.get_transducer_tasks_io(sample["target"], net_output)
        
        trans_loss, joint_out = self.compute_transducer_loss(model, net_output, target, t_len, u_len, reduce=reduce)
        ctc_loss = self.compute_ctc_loss(model, net_output, target, t_len, u_len)
        lm_loss = self.compute_lm_loss(model, net_output, lm_loss_target)
        aux_loss, symm_kl_div_loss = self.compute_aux_transducer_and_symm_kl_div_losses(model, net_output, joint_out, target, aux_t_len, u_len)

        loss = (self.trans_loss_weight * trans_loss) + (self.ctc_loss_weight * ctc_loss) + (self.aux_trans_loss_weight * aux_loss) + (self.symm_kl_div_loss_weight * symm_kl_div_loss) + (self.lm_loss_weight * lm_loss)

        # if not self.training:
        # pred1 = model.greedy_search(net_output)
        # print("pred1 : ", pred1.yseq)
        # exit()
            # pred2 = self.beam_search(net_output[0], 5)
            # print("pred2 : ", pred2)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        # n_correct, total = self.compute_accuracy(model, net_output, target, sample)
        # logging_output["n_correct"] = utils.item(n_correct.data)
        # logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output


    def beam_search(self, x, T, W):

        B = [Sequence(blank=0)]
        batch_size = len(x)
        enc_out = self.encoder.forward(x)
        enc_out = enc_out.squeeze() # batch_size 1 일 때 코드
        for i, f_t in enumerate(enc_out):

            sorted(B, key=lambda a: len(a.k), reverse=True)
            A = B       # Hypotheses that has emitted null from frame 't' and are now in 't+1'
            B = []      # Hypotheses that has not yet emitted null from frame 't' and so can continue emitting more symbols from 't'

            pred_state = self.predictor.initial_state.unsqueeze(0)

            while True:
                y_hat = max(A, key=lambda a: a.logp) # y^ most probable in A

                A.remove(y_hat) # remove y^ from A

                pred_input = torch.tensor([y_hat.k[-1]]).to(x.device)

                g_u, pred_state = self.predictor.forward_one_step(pred_input, y_hat.h)

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

                if len(B) >= W and yb.logp >= y_hat.logp: break

            sorted(B, key=lambda a: a.logp, reverse=True)

        return [(B[0].k)[1:]]

    def compute_transducer_loss(self, model, net_output, target, t_len, u_len, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        loss = rnnt_loss(
            lprobs.float(),
            target,
            frames_lengths=t_len, 
            labels_lengths=u_len,
            reduction="sum",
            blank=self.blank_idx
        )

        loss /= lprobs.size(0)

        return loss, lprobs
    
    def compute_ctc_loss(self, model, net_output, target, t_len, u_len):
        lprobs = model.get_ctc_normalized_probs(net_output)

        with torch.backends.cudnn.flags(deterministic=True):
            loss_ctc = self.ctc_loss(lprobs, target, t_len, u_len)
        
        return loss_ctc.mean()

    def compute_aux_transducer_and_symm_kl_div_losses(self, model, net_output, joint_out, target, aux_t_len, u_len):
        encoder_out, dec_out = net_output
        aux_enc_out = encoder_out["aux_rnn_out"]

        aux_trans_loss, symm_kl_div_loss = model.get_auxiliary_normalized_probs(aux_enc_out, dec_out, joint_out, target, aux_t_len, u_len)
        
        return aux_trans_loss, symm_kl_div_loss

    def compute_lm_loss(self, model, net_output, lm_loss_target):
        lm_loss = model.get_lm_normalized_probs(net_output, lm_loss_target)

        return lm_loss

    def get_transducer_tasks_io(
        self,
        labels: torch.Tensor,
        net_output                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    ):
        """Get Transducer tasks inputs and outputs

        Args:
            labels: Label ID sequences. (B, U)
            net_output: enc_output, dec_output
        
        Returns:
            target: Targer label ID sequences. (B, L)
            T_lengths: Time lengths. (B)
            U_lengths: Label legths. (B)
        """

        encoder_output, decoder_output = net_output
        enc_out_len = encoder_output["src_lengths"]
        aux_enc_out_len = encoder_output["aux_rnn_lens"]

        device = labels.device
        labels_unpad = [label[label != self.padding_idx] for label in labels]
        blank = labels[0].new([self.blank_idx])

        target = pad_list(labels_unpad, self.blank_idx).type(torch.int32).to(device)
        lm_loss_target = (
            pad_list(
                [torch.cat([y, blank], dim=0) for y in labels_unpad], self.padding_idx
            )
            .type(torch.int64)
            .to(device)
        )

        if enc_out_len.dim() > 1:
            enc_mask_unpad = [m[m != 0] for m in enc_out_len]
            enc_out_len = list(map(int, [m.size(0) for m in enc_mask_unpad]))
        else:
            enc_out_len = list(map(int, enc_out_len))
        
        t_len = torch.IntTensor(enc_out_len).to(device)
        u_len = torch.IntTensor([label.size(0) for label in labels_unpad]).to(device)

        if aux_enc_out_len:
            aux_t_len = []

            for i in range(len(aux_enc_out_len)):
                if aux_enc_out_len[i].dim() > 1:
                    aux_mask_unpad = [aux[aux != 0] for aux in aux_enc_out_len[i]]
                    aux_t_len.append(
                        torch.IntTensor(
                            list(map(int, [aux.size(0) for aux in aux_mask_unpad]))
                        ).to(device)
                    )
                else:
                    aux_t_len.append(
                        torch.IntTensor(list(map(int, aux_enc_out_len[i]))).to(device)
                    )
        else:
            aux_t_len = aux_enc_out_len

        return target, lm_loss_target, t_len, aux_t_len, u_len

    @torch.no_grad()
    def compute_accuracy(self, model, net_output, target, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # target = model.get_targets(sample, net_output)
        mask = target.ne(self.padding_idx) # B, U

        bsz, tsz, usz, dsz = lprobs.size()
        max_score = lprobs.max(-1).values
        max_score_idx = max_score.argmax(-2) # B, U
        max_score_idx = max_score_idx.view(bsz * usz)

        label_matrix = lprobs.argmax(-1).transpose(2, 1) # B, U, T
        label_matrix = label_matrix.reshape(bsz * usz, tsz)
        target_pred = label_matrix[torch.arange(bsz * usz), max_score_idx].view(bsz, usz)

        n_correct = torch.sum(
            target_pred.masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True