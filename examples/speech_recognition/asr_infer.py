#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys

import editdistance
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data.data_utils import pad_list

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_transducer_tasks_io(
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

        padding_idx = 1
        blank_idx = 0

        encoder_output, decoder_output = net_output
        enc_out_len = encoder_output["src_lengths"]
        aux_enc_out_len = encoder_output["aux_rnn_lens"]

        device = labels.device
        labels_unpad = [label[label != padding_idx] for label in labels]
        blank = labels[0].new([blank_idx])

        target = pad_list(labels_unpad, blank_idx).type(torch.int32).to(device)
        lm_loss_target = (
            pad_list(
                [torch.cat([y, blank], dim=0) for y in labels_unpad], padding_idx
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

def add_asr_eval_argument(parser):
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary\
output units",
    )
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    parser.add_argument(
        "--asr-decoder",
        choices=["greedy", "default", "tsd", "alsd", "nsc"],
        help="use a asr decoder",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )
    return parser


def check_args(args):
    # assert args.path is not None, "--path required for generation!"
    # assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def get_dataset_itr(args, task, models):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def process_predictions(
    args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id
):
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())

        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces, args.post_process)

        if res_files is not None:
            print(
                "{} ({}-{})".format(hyp_pieces, speaker, id),
                file=res_files["hypo.units"],
            )
            print(
                "{} ({}-{})".format(hyp_words, speaker, id),
                file=res_files["hypo.words"],
            )

        tgt_pieces = tgt_dict.string(target_tokens)
        tgt_words = post_process(tgt_pieces, args.post_process)

        if res_files is not None:
            print(
                "{} ({}-{})".format(tgt_pieces, speaker, id),
                file=res_files["ref.units"],
            )
            print(
                "{} ({}-{})".format(tgt_words, speaker, id), file=res_files["ref.words"]
            )

        if not args.quiet:
            logger.info("HYPO:" + hyp_words)
            logger.info("TARGET:" + tgt_words)
            logger.info("___________________")

        hyp_words = hyp_words.split()
        tgt_words = tgt_words.split()
        return editdistance.eval(hyp_words, tgt_words), len(tgt_words)


def prepare_result_files(args):
    def get_res_file(file_prefix):
        if args.num_shards > 1:
            file_prefix = f"{args.shard_id}_{file_prefix}"
        path = os.path.join(
            args.results_path,
            "{}-{}-{}.txt".format(
                file_prefix, os.path.basename(args.path), args.gen_subset
            ),
        )
        return open(path, "w", buffering=1)

    if not args.results_path:
        return None

    return {
        "hypo.words": get_res_file("hypo.word"),
        "hypo.units": get_res_file("hypo.units"),
        "ref.words": get_res_file("ref.word"),
        "ref.units": get_res_file("ref.units"),
    }


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


class ExistingEmissionsDecoder(object):
    def __init__(self, decoder, emissions):
        self.decoder = decoder
        self.emissions = emissions

    def generate(self, models, sample, **unused):
        ids = sample["id"].cpu().numpy()
        try:
            emissions = np.stack(self.emissions[ids])
        except:
            print([x.shape for x in self.emissions[ids]])
            raise Exception("invalid sizes")
        emissions = torch.from_numpy(emissions)
        return self.decoder.decode(emissions)


def main(args, task=None, model_state=None):
    check_args(args)

    use_fp16 = args.fp16
    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 4000000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    logger.info("| decoding with criterion {}".format(args.criterion))

    task = tasks.setup_task(args)

    # Load ensemble
    if args.load_emissions:
        models, criterions = [], []
        task.load_dataset(args.gen_subset)
    else:
        logger.info("| loading model(s) from {}".format(args.path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(args.path, separator="\\"),
            arg_overrides=ast.literal_eval(args.model_overrides),
            task=task,
            suffix=args.checkpoint_suffix,
            strict=(args.checkpoint_shard_count == 1),
            num_shards=args.checkpoint_shard_count,
            state=model_state,
        )
        optimize_models(args, use_cuda, models)
        task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)


    # Set dictionary
    tgt_dict = task.target_dictionary

    logger.info(
        "| {} {} {} examples".format(
            args.data, args.gen_subset, len(task.dataset(args.gen_subset))
        )
    )

    # hack to pass transitions to W2lDecoder
    if args.criterion == "asg_loss":
        raise NotImplementedError("asg_loss is currently not supported")
        # trans = criterions[0].asg.trans.data
        # args.asg_transitions = torch.flatten(trans).tolist()

    # Load dataset (possibly sharded)
    itr = get_dataset_itr(args, task, models)

    # Initialize generator
    gen_timer = StopwatchMeter()

    models = models[0]

    def build_generator(args):
        asr_decoder = getattr(args, "asr_decoder", None)
        from fairseq.models.transducer.modules.error_calculator import ErrorCalculator
        if asr_decoder == "greedy":
            return ErrorCalculator(
                models.decoder, models.joint, task.tgt_dict, "▁", "<s>", False, True, "greedy"
            )
        elif asr_decoder == "default":
            return ErrorCalculator(
                models.decoder, models.joint, task.tgt_dict, "▁", "<s>", False, True, "default"
            )
        elif asr_decoder == "tsd":
            return ErrorCalculator(
                models.decoder, models.joint, task.tgt_dict, "▁", "<s>", False, True, "tsd"
            )
        else:
            print(
                "only asr decoders with (greedy, default, tsd) options are supported at the moment"
            )

    # please do not touch this unless you test both generate.py and infer.py with audio_pretraining task
    generator = build_generator(args)

    if args.results_path is not None and not os.path.exists(args.results_path):
        os.makedirs(args.results_path)


    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if use_fp16:
                sample = utils.apply_to_sample(apply_half, sample)
            if "net_input" not in sample:
                continue

            with torch.no_grad():
                net_output = models(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"], sample["net_input"]["prev_output_tokens"], sample["target_lengths"])
                target, lm_loss_target, t_len, aux_t_len, u_len = get_transducer_tasks_io(sample["target"], net_output)
                emm = models.get_normalized_probs(net_output, log_probs=True)
                enc_out, _ = net_output
                enc_out = enc_out["encoder_out"]
            
            cer, wer = generator(enc_out, target)
    logger.info(f"WER: {wer}")
    logger.info("| Generate {} with beam={}".format(args.gen_subset, args.beam))
    return task, wer


def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
