"""Evaluator for the SocialIQA social commonsense benchmark."""

from .mcq_evaluator import MCQEvaluator


class SocialIQAEvaluator(MCQEvaluator):
    """Answer questions about likely motives, reactions, and social outcomes."""

    SYSTEM_PROMPT = (
        "You are evaluating social commonsense reasoning. Given a short social "
        "situation, a question, and three possible answers, choose the answer "
        "that best reflects likely intentions, emotions, needs, reactions, or "
        "consequences in everyday human interactions."
    )

    def _extract(self, text: str | None) -> int | None:
        """Return a numeric option ID compatible with the accuracy metric."""
        option = super()._extract(text)
        return int(option) if option is not None else None
