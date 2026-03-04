# flake8: noqa

# Import the generated workflow function to trigger registration
from .wix_rag import wix_rag_function

# Import custom evaluators to trigger registration
from .evaluator_register import register_exact_match_evaluator
from .evaluator_register import register_token_f1_evaluator
from .evaluator_register import register_rouge_l_evaluator
from .evaluator_register import register_bert_score_evaluator