# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig


class ExactMatchEvaluatorConfig(EvaluatorBaseConfig, name="exact_match"):
    """Exact Match: 1.0 if normalized prediction equals normalized reference."""

    max_concurrency: int = Field(default=4, description="Max concurrency for evaluation.")


class TokenF1EvaluatorConfig(EvaluatorBaseConfig, name="token_f1"):
    """Token-level F1 between prediction and reference after normalization."""

    max_concurrency: int = Field(default=4, description="Max concurrency for evaluation.")


class RougeLEvaluatorConfig(EvaluatorBaseConfig, name="rouge_l"):
    """ROUGE-L F-measure based on Longest Common Subsequence."""

    max_concurrency: int = Field(default=4, description="Max concurrency for evaluation.")


class BERTScoreEvaluatorConfig(EvaluatorBaseConfig, name="bert_score"):
    """BERTScore: semantic similarity using contextual embeddings."""

    model_type: str = Field(default="microsoft/deberta-xlarge-mnli", description="HuggingFace model for BERTScore.")
    max_concurrency: int = Field(default=2, description="Max concurrency (lower due to GPU memory).")


@register_evaluator(config_type=ExactMatchEvaluatorConfig)
async def register_exact_match_evaluator(config: ExactMatchEvaluatorConfig, builder: EvalBuilder):
    from .answer_quality_evaluators import ExactMatchEvaluator

    evaluator = ExactMatchEvaluator(max_concurrency=config.max_concurrency)
    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Exact Match Evaluator")


@register_evaluator(config_type=TokenF1EvaluatorConfig)
async def register_token_f1_evaluator(config: TokenF1EvaluatorConfig, builder: EvalBuilder):
    from .answer_quality_evaluators import TokenF1Evaluator

    evaluator = TokenF1Evaluator(max_concurrency=config.max_concurrency)
    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Token F1 Evaluator")


@register_evaluator(config_type=RougeLEvaluatorConfig)
async def register_rouge_l_evaluator(config: RougeLEvaluatorConfig, builder: EvalBuilder):
    from .answer_quality_evaluators import RougeLEvaluator

    evaluator = RougeLEvaluator(max_concurrency=config.max_concurrency)
    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="ROUGE-L Evaluator")


@register_evaluator(config_type=BERTScoreEvaluatorConfig)
async def register_bert_score_evaluator(config: BERTScoreEvaluatorConfig, builder: EvalBuilder):
    from .answer_quality_evaluators import BERTScoreEvaluator

    evaluator = BERTScoreEvaluator(model_type=config.model_type, max_concurrency=config.max_concurrency)
    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="BERTScore Evaluator")
