import time
from typing import Any, Dict, List, Optional, Tuple

import torch


class ClinicalSupervisorAgent:
    """
    Orchestrates compression, drafting, detection and targeted repair; terminates when no span is flagged,
    when the (spans, flags) state repeats, or after `max_iterations` repair rounds. Calls no model itself.
    Per-stage wall-clock time is recorded (computational overhead), as is the summary
    after every repair round (C2 = draft, C3 = first repair, C4 = final).
    """

    def __init__(self, focus_agent, draft_agent, hallucination_detector_agent, fix_agent, max_iterations: int = 3):
        self.focus_agent, self.draft_agent = focus_agent, draft_agent
        self.detector, self.fix_agent = hallucination_detector_agent, fix_agent
        self.max_iterations = max_iterations

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run(self, document: str) -> Dict[str, Any]:
        self.detector.reset()
        t = {"focus": 0.0, "draft": 0.0, "detect": 0.0, "fix": 0.0}

        def timed(key, fn, *a, **k):
            self._sync(); t0 = time.perf_counter()
            out = fn(*a, **k)
            self._sync(); t[key] += time.perf_counter() - t0
            return out

        focus = timed("focus", self.focus_agent.compress, document)
        draft = timed("draft", self.draft_agent.generate, focus["sentences"])
        summary = draft
        history: List[str] = [draft]
        previous: Optional[Tuple] = None
        iteration, reason, first_flagged, spans, mask = 0, None, None, [], {}
        rounds: List[Dict[str, Any]] = []

        while True:
            det = timed("detect", self.detector.analyze, source_document=document, draft_summary=summary)
            spans, mask = det["spans"], det["hallucination_mask"]
            n_flag = sum(mask.values())
            rounds.append({"aura": [round(det["span_aura_scores"][j], 4) for j in range(len(spans))],
                           "p_not": [round(det["semantic_p_not"][j], 4) for j in range(len(spans))],
                           "mask": [mask[j] for j in range(len(spans))]})
            if first_flagged is None:
                first_flagged = n_flag
            state = (tuple(spans), tuple(sorted(mask.items())))
            if n_flag == 0:
                reason = "no_hallucinations_remaining"
            elif iteration >= self.max_iterations:
                reason = "max_iterations_reached"
            elif previous is not None and state == previous:
                reason = "state_unchanged"
            if reason:
                break
            previous = state
            iteration += 1
            summary = timed("fix", self.fix_agent.fix, source_document=document, spans=spans, hallucination_mask=mask)
            history.append(summary)

        return {"draft_summary": draft, "fixed_summary": summary, "summaries_by_round": history,
                "num_iterations": iteration, "termination_reason": reason,
                "n_spans": len(spans), "n_flagged_first": first_flagged, "n_flagged_final": sum(mask.values()),
                "aura_skipped": bool(det.get("aura_skipped", False)), "detect_log": rounds,
                "timings_s": t, "focus": {k: focus[k] for k in
                                          ("fallback_used", "fallback_reason", "total_tokens", "retained_tokens")}}
