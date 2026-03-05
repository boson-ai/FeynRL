# Adapted from https://github.com/verl-project/verl/blob/d9d94b4da93fbacc06bb546629171c67c0a674aa/verl/utils/reward_score/math_reward.py

import logging
from typing import Callable, Optional, Sequence, Dict, Any
import torch

from math_verify.errors import TimeoutException
from math_verify.grader import verify
from math_verify.parser import ExprExtractionConfig, ExtractionTarget, LatexExtractionConfig, parse
from math_verify.utils import timeout

logger = logging.getLogger(__name__)


def math_metric(
    gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    aggregation_function: Callable[[list[float]], float] = max,
    precision: int = 6,
) -> Callable[[list[str], list[str]], tuple[float, Optional[tuple[list[str], list[str]]]]]:
    """Creates a language-aware extractive match metric that extracts answers from the model's output.

    Known issues:
    - If the task is to simplify an expression, the metric might overestimate the accuracy. This is because if the model doesn't output any anchor for the extraction (e.g final answer is..),
        it's possible that the the extracted prediction will be the expression to simplify. Because we do simplifications ourselves, it can thus happen that sympy will correctly simplify the expression,
        thus it will match gold, despite model not doing anything. PRs to fix this are welcome.

    Args:
        gold_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for gold answers. Defaults to extracting simple math expressions.
        pred_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for predictions. Defaults to extracting simple math expressions.
        aggregation_function: Callable[[list[float]], float]
            Function to aggregate scores when multiple golds/predictions are present. Defaults to max.
        precision: int
            Number of decimal places to use when comparing numerical values. Defaults to 6.

    Returns:
        A sample level metric that extracts and compares mathematical expressions.

    """

    @timeout(30)
    def get_str_preds_with_timeout(extracted_predictions: list[list[str]], extracted_golds: list[list[str]]) -> tuple[list[str], list[str]]:
        golds = [str(gold) for golds in extracted_golds for gold in golds]
        predictions = [str(pred) for preds in extracted_predictions for pred in preds]
        return (golds, predictions)

    def sample_level_fn(golds: list[str], predictions: list[str]) -> tuple[float, Optional[tuple[list[str], list[str]]]]:
        extracted_predictions = [parse(pred, pred_extraction_target) for pred in predictions]
        extracted_golds = [parse(gold, gold_extraction_target, parsing_timeout=30) for gold in golds]

        # Assert on empty gold and warn on empty pred
        if any(len(g) == 0 for g in extracted_golds):
            raise ValueError(f"No gold targets found for at least one gold. Gold: {golds}, Pred: {predictions}")

        # We have to use timeout because the sypmy to str conversion can be very slow
        str_preds = None
        try:
            str_preds = get_str_preds_with_timeout(extracted_predictions, extracted_golds)
        except TimeoutException:
            logger.warning("Timeout when adding extracted predictions and golds to specific")

        return (
            aggregation_function([(1.0 if any(verify(gold, pred, precision, timeout_seconds=30) for gold in extracted_golds) else 0.0) for pred in extracted_predictions]),
            str_preds,
        )

    return sample_level_fn


def compute_score(prompt_data: Dict[str, Any], response_data: Dict[str, Any], timeout_score: float = 0.0):
    """
    input args:
        prompt_data: Dict[str, Any] - dictionary containing prompt data
        response_data: Dict[str, Any] - dictionary containing response data
        timeout_score: float - score to return on timeout
    output args:
        r: torch.Tensor - reward tensor
        is_per_token: bool - whether the reward is per token
    """
    
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    
    solution_str = response_data.text
    ground_truth = prompt_data["solution"]
    
    # GSM8K style reward returns a tensor of same length as response token IDs
    r = torch.zeros((len(response_data.token_ids),), dtype=torch.float32)
    is_per_token = False
    
    if len(response_data.token_ids) == 0:
        return r, is_per_token

    # Wrap the ground truth in \boxed{} format for verification if it's not already
    ground_truth_boxed = ground_truth
    if "\\boxed" not in ground_truth:
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    ret_score = 0.0
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
    except TimeoutException:
        ret_score = timeout_score
    except Exception as e:
        logger.error(f"Error in math_verify compute_score: {e}")
        ret_score = 0.0

    r[-1] = float(ret_score)
    return r, is_per_token
