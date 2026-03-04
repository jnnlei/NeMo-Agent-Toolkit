# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import string
import unicodedata
from collections import Counter
from typing import override

from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem


def _normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation/whitespace, and normalize unicode.

    Follows the normalization convention from SQuAD evaluation scripts.
    """
    text = unicodedata.normalize("NFKD", str(text))
    text = text.lower()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = " ".join(text.split())
    return text


class ExactMatchEvaluator(BaseEvaluator):
    """Exact Match: 1.0 if normalized prediction equals normalized reference, else 0.0."""

    def __init__(self, max_concurrency: int = 4):
        super().__init__(max_concurrency, tqdm_desc="Evaluating Exact Match")

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        predicted = _normalize_answer(str(item.output_obj))
        expected = _normalize_answer(str(item.expected_output_obj))
        score = 1.0 if predicted == expected else 0.0
        return EvalOutputItem(
            id=item.id,
            score=score,
            reasoning={"predicted_normalized": predicted, "expected_normalized": expected},
        )


class TokenF1Evaluator(BaseEvaluator):
    """Token-level F1: harmonic mean of token precision and recall against reference."""

    def __init__(self, max_concurrency: int = 4):
        super().__init__(max_concurrency, tqdm_desc="Evaluating Token F1")

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        pred_tokens = _normalize_answer(str(item.output_obj)).split()
        gold_tokens = _normalize_answer(str(item.expected_output_obj)).split()

        if not pred_tokens and not gold_tokens:
            return EvalOutputItem(id=item.id, score=1.0, reasoning={"precision": 1.0, "recall": 1.0, "f1": 1.0})
        if not pred_tokens or not gold_tokens:
            return EvalOutputItem(id=item.id, score=0.0, reasoning={"precision": 0.0, "recall": 0.0, "f1": 0.0})

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return EvalOutputItem(
            id=item.id,
            score=round(f1, 4),
            reasoning={"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
        )


class RougeLEvaluator(BaseEvaluator):
    """ROUGE-L: F-measure based on Longest Common Subsequence between prediction and reference."""

    def __init__(self, max_concurrency: int = 4):
        super().__init__(max_concurrency, tqdm_desc="Evaluating ROUGE-L")

    @staticmethod
    def _lcs_length(x: list[str], y: list[str]) -> int:
        m, n = len(x), len(y)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev = curr
        return prev[n]

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        pred_tokens = _normalize_answer(str(item.output_obj)).split()
        gold_tokens = _normalize_answer(str(item.expected_output_obj)).split()

        if not pred_tokens and not gold_tokens:
            return EvalOutputItem(id=item.id, score=1.0, reasoning={"precision": 1.0, "recall": 1.0, "rouge_l": 1.0})
        if not pred_tokens or not gold_tokens:
            return EvalOutputItem(id=item.id, score=0.0, reasoning={"precision": 0.0, "recall": 0.0, "rouge_l": 0.0})

        lcs_len = self._lcs_length(pred_tokens, gold_tokens)
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(gold_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return EvalOutputItem(
            id=item.id,
            score=round(f1, 4),
            reasoning={"precision": round(precision, 4), "recall": round(recall, 4), "rouge_l": round(f1, 4)},
        )


class BERTScoreEvaluator(BaseEvaluator):
    """BERTScore: semantic similarity using contextual embeddings (F1 variant)."""

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", max_concurrency: int = 2):
        super().__init__(max_concurrency, tqdm_desc="Evaluating BERTScore")
        self.model_type = model_type
        self._scorer = None

    def _get_scorer(self):
        if self._scorer is None:
            from bert_score import BERTScorer
            self._scorer = BERTScorer(model_type=self.model_type, lang="en", rescale_with_baseline=True)
        return self._scorer

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        import asyncio

        predicted = str(item.output_obj)
        expected = str(item.expected_output_obj)

        scorer = self._get_scorer()
        p, r, f1 = await asyncio.to_thread(scorer.score, [predicted], [expected])

        return EvalOutputItem(
            id=item.id,
            score=round(f1.item(), 4),
            reasoning={
                "precision": round(p.item(), 4),
                "recall": round(r.item(), 4),
                "f1": round(f1.item(), 4),
                "model_type": self.model_type,
            },
        )
