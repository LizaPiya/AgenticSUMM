"""
Shared prompt text for the sensitivity analysis.

MIMIC uses the DraftAgent prompt from Agents/draft_agent.py (copied verbatim).
SOAP uses the SOAP-format prompt from Baselines/draft_agent_soap.py (copied verbatim), because
the SOAP dialogues are summarized into SOAP notes, not hospital-stay summaries.
"""

DRAFT_MAX_NEW_TOKENS = 256


def build_draft_prompt(dataset: str, context: str) -> str:
    if dataset == "mimic":
        return (
            "Based on the patient record below, write a concise clinical summary "
            "describing the patient's hospital stay. Include reason for admission, "
            "key findings, and treatments given. Use only information from the record. "
            "Do NOT add notes, commentary, or explanations about the summary itself.\n\n"
            f"{context}\n\n"
            "Summary:"
        )
    if dataset == "soap":
        return (
            "You are a helpful medical assistant. Analyze the following patient-doctor dialogue and "
            "create a concise medical summary in SOAP format (Subjective, Objective, Assessment, Plan). "
            "Focus on key symptoms, findings, diagnosis, and treatment plan:\n\n"
            f"{context}\n\nSOAP Summary:"
        )
    raise ValueError(dataset)


def build_scored_judge_prompt(source: str, summary: str) -> str:
    """All four scales point the same way (5 is best), and the reply format is fixed."""
    return f"""You are checking a clinical summary against its source note.

SOURCE NOTE:
{source}

SUMMARY:
{summary}

Rate the summary on four scales from 1 (worst) to 5 (best). Higher is always better:
- Faithfulness: 5 = every statement is supported by the source; 1 = many statements are invented or contradict it.
- Accuracy: 5 = all clinical facts, doses and negations are correct; 1 = many are wrong.
- Completeness: 5 = covers the key clinical information; 1 = misses most of it.
- Coherence: 5 = fluent and well organised; 1 = hard to follow.

Reply with exactly these four lines and nothing else:
Faithfulness: <1-5>
Accuracy: <1-5>
Completeness: <1-5>
Coherence: <1-5>"""
