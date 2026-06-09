"""Evaluator for the BoolQ yes/no reading-comprehension benchmark."""

from .mcq_evaluator import MCQEvaluator


class BoolQEvaluator(MCQEvaluator):
    """Answer a yes/no question using evidence from its passage."""

    SYSTEM_PROMPT = (
        "You are evaluating yes/no reading comprehension. Read the passage, "
        "then decide whether the correct answer to the question is no or yes. "
        "Use only the information supported by the passage and choose exactly "
        "one of the two numbered options."
    )

    def _extract(self, text: str | None) -> int | None:
        """Return a numeric option ID compatible with the accuracy metric."""
        option = super()._extract(text)
        return int(option) if option is not None else None
