"""Reserved prompt variant names that have no text yet."""

from aity_web.llm.prompts.base import PromptVariant

EXTRACTION_V1 = PromptVariant(
    name="extraction_v1",
    description="Planned: structured field extraction (e.g. SMS data-extraction forms).",
    status="placeholder",
)

QA_V1 = PromptVariant(
    name="qa_v1",
    description="Planned: answer a research question from one document.",
    status="placeholder",
)
