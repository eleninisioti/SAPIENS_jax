"""Evaluation metrics: pass@k per learner, group pass@k, and answer conformity between learners."""
from math import comb

import numpy as np


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def miss_probability(n, c, k):
    return 1.0 - pass_at_k(n, c, k)


def group_pass_at_k(correct, k):
    """correct: (learners, problems, samples) booleans. Probability that at least one learner solves a
    problem when each draws k of its samples, averaged over problems."""
    learners, problems, n = correct.shape
    counts = correct.sum(-1)
    return float(np.mean([1.0 - np.prod([miss_probability(n, int(counts[i, p]), k) for i in range(learners)])
                          for p in range(problems)]))


def learner_pass_at_k(correct, k):
    learners, problems, n = correct.shape
    counts = correct.sum(-1)
    return [float(np.mean([pass_at_k(n, int(counts[i, p]), k) for p in range(problems)])) for i in range(learners)]


def last_boxed(text):
    """Content of the last \\boxed{...} in text (braces balanced), or None."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j].strip()
    return None


def conformity(answers):
    """answers: (learners, problems, samples) of strings or None. Fraction of problems on which all
    learners' most frequent answers coincide."""
    learners, problems, _ = len(answers), len(answers[0]), None
    agree = []
    for p in range(problems):
        majority = []
        for i in range(learners):
            values = [a for a in answers[i][p] if a is not None]
            majority.append(max(set(values), key=values.count) if values else None)
        agree.append(len(set(majority)) == 1 and majority[0] is not None)
    return float(np.mean(agree))
