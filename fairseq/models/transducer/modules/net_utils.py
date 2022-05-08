# -*- coding: utf-8 -*-

"""Network related utility tools."""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch

@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None

@dataclass
class ExtendedHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search and mAES."""

    dec_out: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None

def get_subsample(train_args, mode, arch):
    """Parse the subsampling factors from the args for the specified `mode` and `arch`.
    Args:
        train_args: argument Namespace containing options.
        mode: one of ('asr', 'mt', 'st')
        arch: one of ('rnn', 'rnn-t', 'rnn_mix', 'rnn_mulenc', 'transformer')
    Returns:
        np.ndarray / List[np.ndarray]: subsampling factors.
    """
    if arch == "transformer":
        return np.array([1])

    elif mode == "mt" and arch == "rnn":
        # +1 means input (+1) and layers outputs (train_args.elayer)
        subsample = np.ones(train_args.encoder_layers + 1, dtype=np.int)
        logging.warning("Subsampling is not performed for machine translation.")
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif (
        (mode == "asr" and arch in ("rnn", "rnn-t"))
        or (mode == "mt" and arch == "rnn")
        or (mode == "st" and arch == "rnn")
    ):
        subsample = np.ones(train_args.encoder_layers + 1, dtype=np.int)
        print("sub 1 : ", subsample)
        ss = train_args.subsample.split("_")
        for j in range(min(train_args.encoder_layers + 1, len(ss))):
            subsample[j] = int(ss[j])
        print("sub 2 : ", subsample)
        exit()
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    else:
        raise ValueError("Invalid options: mode={}, arch={}".format(mode, arch))

def subtract(
    x: List[ExtendedHypothesis], subset: List[ExtendedHypothesis]
) -> List[ExtendedHypothesis]:
    """Remove elements of subset if corresponding label ID sequence already exist in x.
    Args:
        x: Set of hypotheses.
        subset: Subset of x.
    Returns:
       final: New set of hypotheses.
    """
    final = []

    for x_ in x:
        if any(x_.yseq == sub.yseq for sub in subset):
            continue
        final.append(x_)

    return final

def select_lm_state(
    lm_states: Union[List[Any], Dict[str, Any]],
    idx: int,
    lm_layers: int,
    is_wordlm: bool,
) -> Union[List[Any], Dict[str, Any]]:
    """Get ID state from LM hidden states.
    Args:
        lm_states: LM hidden states.
        idx: LM state ID to extract.
        lm_layers: Number of LM layers.
        is_wordlm: Whether provided LM is a word-level LM.
    Returns:
       idx_state: LM hidden state for given ID.
    """
    if is_wordlm:
        idx_state = lm_states[idx]
    else:
        idx_state = {}

        idx_state["c"] = [lm_states["c"][layer][idx] for layer in range(lm_layers)]
        idx_state["h"] = [lm_states["h"][layer][idx] for layer in range(lm_layers)]

    return idx_state

def select_k_expansions(
    hyps: List[ExtendedHypothesis],
    logps: torch.Tensor,
    beam_size: int,
    gamma: float,
    beta: float,
) -> List[ExtendedHypothesis]:
    """Return K hypotheses candidates for expansion from a list of hypothesis.
    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.
    Args:
        hyps: Hypotheses.
        beam_logp: Log-probabilities for hypotheses expansions.
        beam_size: Beam size.
        gamma: Allowed logp difference for prune-by-value method.
        beta: Number of additional candidates to store.
    Return:
        k_expansions: Best K expansion hypotheses candidates.
    """
    k_expansions = []

    for i, hyp in enumerate(hyps):
        hyp_i = [(int(k), hyp.score + float(logp)) for k, logp in enumerate(logps[i])]
        k_best_exp = max(hyp_i, key=lambda x: x[1])[1]

        k_expansions.append(
            sorted(
                filter(lambda x: (k_best_exp - gamma) <= x[1], hyp_i),
                key=lambda x: x[1],
                reverse=True,
            )[: beam_size + beta]
        )

    return k_expansions

def recombine_hyps(hyps: List[Hypothesis]) -> List[Hypothesis]:
    """Recombine hypotheses with same label ID sequence.
    Args:
        hyps: Hypotheses.
    Returns:
       final: Recombined hypotheses.
    """
    final = []

    for hyp in hyps:
        seq_final = [f.yseq for f in final if f.yseq]

        if hyp.yseq in seq_final:
            seq_pos = seq_final.index(hyp.yseq)

            final[seq_pos].score = np.logaddexp(final[seq_pos].score, hyp.score)
        else:
            final.append(hyp)

    return final

def is_prefix(x: List[int], pref: List[int]) -> bool:
    """Check if pref is a prefix of x.
    Args:
        x: Label ID sequence.
        pref: Prefix label ID sequence.
    Returns:
        : Whether pref is a prefix of x.
    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False

    return True

def init_lm_state(lm_model: torch.nn.Module):
    """Initialize LM hidden states.
    Args:
        lm_model: LM module.
    Returns:
        lm_state: Initial LM hidden states.
    """
    lm_layers = len(lm_model.rnn)
    lm_units_typ = lm_model.typ
    lm_units = lm_model.n_units

    p = next(lm_model.parameters())

    h = [
        torch.zeros(lm_units).to(device=p.device, dtype=p.dtype)
        for _ in range(lm_layers)
    ]

    lm_state = {"h": h}

    if lm_units_typ == "lstm":
        lm_state["c"] = [
            torch.zeros(lm_units).to(device=p.device, dtype=p.dtype)
            for _ in range(lm_layers)
        ]

    return lm_state

def create_lm_batch_states(
    lm_states: Union[List[Any], Dict[str, Any]], lm_layers, is_wordlm: bool
) -> Union[List[Any], Dict[str, Any]]:
    """Create LM hidden states.
    Args:
        lm_states: LM hidden states.
        lm_layers: Number of LM layers.
        is_wordlm: Whether provided LM is a word-level LM.
    Returns:
        new_states: LM hidden states.
    """
    if is_wordlm:
        return lm_states

    new_states = {}

    new_states["c"] = [
        torch.stack([state["c"][layer] for state in lm_states])
        for layer in range(lm_layers)
    ]
    new_states["h"] = [
        torch.stack([state["h"][layer] for state in lm_states])
        for layer in range(lm_layers)
    ]

    return new_states