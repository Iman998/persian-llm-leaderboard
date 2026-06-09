"""Evaluator for the Natural Questions open-domain benchmark."""

from .triviaqa_evaluator import TriviaQAEvaluator


class NaturalQuestionsEvaluator(TriviaQAEvaluator):
    """Produce concise answer strings for NQ-Open questions."""
