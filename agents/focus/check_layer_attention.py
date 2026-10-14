"""
GPU check: single-layer attention computed from q/k projections must match the
attention weights returned by HF eager attention (output_attentions=True).
Both are run on short documents only (eager materializes every layer).
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from focus_scoring import capture_layer_inputs, layer_attention_chunks

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert TOKEN, "HUGGINGFACE_HUB_TOKEN not set"
assert torch.cuda.is_available(), "CUDA required"

DOCS = [
    "The patient was admitted with chest pain. Troponin was negative. "
    "She was discharged on aspirin 81 mg daily. No evidence of pulmonary embolism.",
    "Chief complaint: shortness of breath. History of COPD. "
    "Albuterol 2 puffs every 4 hours as needed. Follow up with primary care in one week. "
    "Allergies: penicillin (rash).",
    "Alert and oriented. Lungs clear. Abdomen soft and nontender. "
    "Plan: continue current medications and recheck labs tomorrow.",
]

tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    token=TOKEN, attn_implementation="eager",
)
model.eval()
n_layers = model.config.num_hidden_layers

worst = 0.0
for layer in sorted({n_layers - 1, n_layers // 2}):
    for doc in DOCS:
        ids = tok(doc, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            ref = model(input_ids=ids, output_attentions=True, use_cache=False).attentions[layer][0].float()
        cap = capture_layer_inputs(model, ids, layer)
        mine = torch.cat([A for _, A in layer_attention_chunks(model, cap, layer, 8)], dim=0)
        diff = (mine - ref).abs().max().item()
        worst = max(worst, diff)
        print(f"layer {layer:2d} n={ids.shape[1]:3d} max|diff|={diff:.5f} "
              f"row-sum err={(mine.sum(-1) - 1).abs().max().item():.2e}")

# bf16 eager rounds attention weights to bf16 (about 3 significant digits).
TOL = 2e-2
print(f"worst max|diff| = {worst:.5f} (tolerance {TOL})")
sys.exit(0 if worst < TOL else 1)
