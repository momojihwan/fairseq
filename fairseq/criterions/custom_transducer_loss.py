# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
import math
from ntpath import join
from re import U
from fairseq.models.transducer.rnn_transducer import JointNetwork
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
from fairseq.models.transducer.modules.error_calculator import ErrorCalculator
from omegaconf import II

from warp_rnnt import rnnt_loss


@dataclass
class TransducerCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("custom_transducer", dataclass=TransducerCriterionConfig)
class Transducer(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.padding_idx = task.tgt_dict.pad() 
        self.blank_idx = task.tgt_dict.bos()
        
        self.token_list = task.tgt_dict


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        net_output = model(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"], sample["net_input"]["prev_output_tokens"], sample["target_lengths"])
        
        loss, _ = self.compute_transducer_loss(model, net_output, sample["target"], sample["net_input"]["src_lengths"], sample["target_lengths"])
        
        # if not self.training:
        #     hyp = model.beam_search(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"])
        #     print("hyp : ", hyp)
            # self.error_calculator = ErrorCalculator(
            #     model.decoder, model.joint, self.token_list, "▁", "<unk>", False, True, "greedy"
            # )
            # cer, wer = self.error_calculator(enc_out, target)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def compute_transducer_loss(self, model, net_output, target, t_len, u_len, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        loss = rnnt_loss(
            lprobs.float(),
            target.int(),
            frames_lengths=t_len.int(), 
            labels_lengths=u_len.int(),
            reduction="mean",
            blank=self.blank_idx
        )

        return loss, lprobs


    @torch.no_grad()
    def compute_accuracy(self, model, net_output, target, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        
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