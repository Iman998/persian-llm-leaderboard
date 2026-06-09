"""Evaluator for the WinoGrande commonsense coreference benchmark."""

from .mcq_evaluator import MCQEvaluator


class WinoGrandeEvaluator(MCQEvaluator):
    """Choose the option that correctly fills the sentence blank."""

    SYSTEM_PROMPT = (
        "You are evaluating commonsense coreference resolution. Given a "
        "sentence containing one blank and two candidate noun phrases, choose "
        "the candidate that makes the completed sentence coherent and "
        "consistent with everyday commonsense."
    )

    def _extract(self, text: str | None) -> int | None:
        """Return a numeric option ID compatible with the accuracy metric."""
        option = super()._extract(text)
        return int(option) if option is not None else None
