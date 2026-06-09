"""Evaluator for the PIQA physical commonsense benchmark."""

from .mcq_evaluator import MCQEvaluator


class PIQAEvaluator(MCQEvaluator):
    """Choose the more physically plausible solution for a goal."""

    SYSTEM_PROMPT = (
        "You are evaluating physical commonsense reasoning. Given a practical "
        "goal and two possible solutions, choose the solution that is more "
        "likely to work in the real world. Consider physical constraints, "
        "object properties, safety, and whether the proposed steps achieve "
        "the stated goal."
    )

    def _extract(self, text: str | None) -> int | None:
        """Return a numeric option ID compatible with the accuracy metric."""
        option = super()._extract(text)
        return int(option) if option is not None else None
