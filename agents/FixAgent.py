import torch
from typing import Dict, List


class FixAgent:
    """
    FixAgent performs targeted correction of hallucinated spans
    under the supervision of ClinicalSupervisorAgent.
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 250):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def fix(
        self,
        source_document: str,
        spans: List[str],
        hallucination_mask: Dict[int, int],
    ) -> str:
        """Rewrites hallucinated spans and returns a revised summary."""
        revised_spans = []
        
        for idx, span in enumerate(spans):
            if hallucination_mask.get(idx, 0) == 0:
                revised_spans.append(span)
            else:
                fixed_span = self._rewrite_span(source_document, span)
                if fixed_span:
                    revised_spans.append(fixed_span)
        
        return " ".join(revised_spans)

    def _rewrite_span(self, source_document: str, hallucinated_span: str) -> str:
        """Rewrites a hallucinated span. Returns empty string to delete."""
        
        prompt = (
            f"Source: {source_document}\n\n"
            f"Fix this sentence using ONLY the source, or respond 'DELETE':\n"
            f"{hallucinated_span}\n\n"
            f"Fixed sentence:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        rewritten = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Delete if model says DELETE or output is suspicious
        if "delete" in rewritten.lower() or len(rewritten) < 10:
            return ""
        
        return rewritten