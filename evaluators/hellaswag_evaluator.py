"""Evaluator for the HellaSwag commonsense sentence-completion benchmark."""

from .mcq_evaluator import MCQEvaluator


class HellaSwagEvaluator(MCQEvaluator):
    """Select the most plausible continuation for a HellaSwag context."""

    SYSTEM_PROMPT = (
        "You are evaluating commonsense sentence completion. Given a context "
        "and four possible continuations, choose the single continuation that "
        "most plausibly and coherently describes what happens next. Pay close "
        "attention to physical possibility, temporal order, and consistency "
        "with the people and objects in the context."
    )

    def _extract(self, text: str | None) -> int | None:
        """Return a numeric option ID compatible with the accuracy metric."""
        option = super()._extract(text)
        return int(option) if option is not None else None
