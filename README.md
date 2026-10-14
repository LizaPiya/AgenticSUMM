# AgenticSum

Inference-time agentic framework for faithful clinical summarization with a single frozen decoder-only LLM.
Five roles share one backbone and no parameter is updated:

| Role | File |
|---|---|
| FocusAgent (attention-based input compression, FOCUS) | `agents/focus/focus_agent.py`, `agents/focus/focus_scoring.py` |
| DraftAgent | `agents/draft_agent.py` |
| HallucinationDetectorAgent (AURA + entailment judge) | `agents/hallucination_detector.py` |
| FixAgent (targeted repair of flagged sentences) | `agents/FixAgent.py` |
| ClinicalSupervisorAgent (refinement loop and termination) | `agents/clinical_supervisor.py` |

`agents/run_agenticsum.py` runs the full pipeline on one dataset split and writes one CSV row per note.

## Setup

```
pip install -r requirements.txt
export HUGGINGFACE_HUB_TOKEN=<token>        # for gated models such as Llama
export AGENTICSUM_DATA=/path/to/data        # see "Data"
```

## Data

Two public datasets are used: MIMIC-IV-Ext-BHC (`$AGENTICSUM_DATA/mimic-iv-bhc.csv`) and a patient-doctor
dialogue collection with SOAP notes (`$AGENTICSUM_DATA/soap_{train,val,test}.jsonl`, fields `ID`, `dialogue`, `note`).
Notes are assigned to calibration / development / test buckets by a hash of the patient id (MIMIC-IV) or of the
dialogue text (SOAP), so the splits are disjoint.

## Run

```
cd agents/focus

# 1. calibrate the positional baseline of one attention layer on the calibration split
python run_layer_sweep.py --dataset mimic --split cal --calibrate --n-notes 200 --tables layer_tables_mimic.pt --out ../../outputs

# 2. run the pipeline on test notes
cd ..
python run_agenticsum.py --dataset mimic --split test --n-notes 500 --layer 24 --rho 0.9 \
    --tau <aura-threshold> --entail-cut <entailment-cutoff> --tables focus/layer_tables_mimic.pt \
    --out ../outputs/pipeline_mimic_test.csv
```

Detector thresholds are chosen on development notes with `focus/detector_validation.py` and
`focus/analyze_detval.py`. Baselines: `focus/baseline_models.py`, `focus/agentic_baselines.py`.
Scoring: `focus/eval_systems.py` (overlap metrics), `focus/judge_jury.py` and `focus/aggregate_jury.py` (LLM judges),
`focus/paired_analysis.py`, `focus/efficiency.py`.

## Tests

```
cd agents/focus
python test_focus_scoring.py
python test_focus_fast.py
python test_agentic_baselines.py
```
