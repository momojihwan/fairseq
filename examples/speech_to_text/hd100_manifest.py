#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--test-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="percentage of data to use as test set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0
    assert args.test_percent >= 0 and args.test_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    train_path = os.path.join("/DB/HD_100h", "HD100h/train")
    dev_path = os.path.join("/DB/HD_100h", "HD100h/valid")
    test_path = os.path.join("/DB/HD_100h", "HD100h/test")
    rand = random.Random(args.seed)

    valid_f = (
        open(os.path.join(args.dest, "valid.tsv"), "w")
        if args.valid_percent > 0
        else None
    )
    test_f = (
        open(os.path.join(args.dest, "test.tsv"), "w")
        if args.test_percent > 0
        else None
    )

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f:
        print(train_path, file=train_f)

        if valid_f is not None:
            print(dev_path, file=valid_f)
        if test_f is not None:
            print(test_path, file=test_f)
            
        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            if args.path_must_contain and args.path_must_contain not in file_path:
                continue

            frames = soundfile.info(fname).frames
            dest = train_f if rand.random() > args.valid_percent else valid_f
            f = os.path.relpath(file_path, dir_path)

            if dest == train_f:
                dest = train_f if rand.random() > args.test_percent else test_f
            
            file_spk, file_wav = f.split("/")
            file_txt = file_wav.split(".")[0] + ".txt"
            f_t = os.path.join(file_spk, file_txt)
            txt_fname = fname.split(".")[0] + ".txt"
            if dest == train_f:
                os.makedirs(os.path.join(train_path, file_spk), exist_ok=True)
                sym_wav_path = os.path.join(train_path, f)
                sym_txt_path = os.path.join(train_path, f_t)
                os.symlink(fname, sym_wav_path)
                os.symlink(txt_fname, sym_txt_path)
            elif dest == valid_f:
                os.makedirs(os.path.join(dev_path, file_spk), exist_ok=True)
                sym_wav_path = os.path.join(dev_path, f)
                sym_txt_path = os.path.join(dev_path, f_t)
                os.symlink(fname, sym_wav_path)
                os.symlink(txt_fname, sym_txt_path)
            else:
                os.makedirs(os.path.join(test_path, file_spk), exist_ok=True)
                sym_wav_path = os.path.join(test_path, f)
                sym_txt_path = os.path.join(test_path, f_t)
                os.symlink(fname, sym_wav_path)
                os.symlink(txt_fname, sym_txt_path)

            print(
                "{}\t{}".format(f, frames), file=dest
            )
    if valid_f is not None:
        valid_f.close()
    if test_f is not None:
        test_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
