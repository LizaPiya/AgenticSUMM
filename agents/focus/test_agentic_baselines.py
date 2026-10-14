"""
CPU tests for the control flow of Self-Refine, Chain-of-Verification and Reflexion (agentic_baselines.py),
using a scripted fake model so no GPU and no weights are needed.
Run: python test_agentic_baselines.py
"""
from agentic_baselines import cove, reflexion, self_refine

NOTE = "The patient was admitted with chest pain. Troponin was negative. She was discharged on aspirin."


class FakeGen:
    """Routes by the instruction text in the prompt; `script` maps a kind to a list of replies (the last repeats)."""

    KINDS = [
        ("reviewing a draft summary", "feedback"),
        ("Revise the draft", "refine"),
        ("fact-checking questions", "plan"),
        ("Answer the question using only the SOURCE", "answer"),
        ("final, verified version", "final"),
        ("List every statement in the summary", "evaluate"),
        ("say what went wrong", "reflect"),
    ]

    def __init__(self, script):
        self.script = script
        self.counts = {}
        self.prompts = []
        self.calls = 0

    def __call__(self, prompt, max_new_tokens):
        self.calls += 1
        self.prompts.append(prompt)
        kind = next((k for key, k in self.KINDS if key in prompt), "draft")
        i = self.counts.get(kind, 0)
        self.counts[kind] = i + 1
        replies = self.script.get(kind, [f"<{kind}>"])
        return replies[min(i, len(replies) - 1)]


def test_self_refine_stops_when_no_issues():
    g = FakeGen({"draft": ["S0"], "feedback": ["The aspirin dose is not stated.", "NO ISSUES"], "refine": ["S1"]})
    summary, iters = self_refine(g, "mimic", NOTE, max_iter=3)
    assert (summary, iters) == ("S1", 1), (summary, iters)
    assert g.calls == 4            # draft, feedback, refine, feedback
    assert "The aspirin dose is not stated." in g.prompts[2]    # the refine step sees the feedback


def test_self_refine_respects_budget():
    g = FakeGen({"draft": ["S0"], "feedback": ["still wrong"], "refine": ["S1", "S2", "S3"]})
    summary, iters = self_refine(g, "mimic", NOTE, max_iter=3)
    assert (summary, iters) == ("S3", 3)
    assert g.calls == 1 + 3 * 2


def test_cove_parses_questions_and_uses_answers():
    g = FakeGen({"draft": ["Draft with aspirin."], "plan": ["1. Was troponin negative?\n2) Was aspirin given?\n3. Was she admitted?"],
                 "answer": ["yes", "not stated", "yes"], "final": ["FINAL"]})
    final, n_qa = cove(g, "mimic", NOTE)
    assert (final, n_qa) == ("FINAL", 3)
    assert g.calls == 1 + 1 + 3 + 1
    last = g.prompts[-1]
    assert "Q: Was aspirin given?" in last and "A: not stated" in last and "Draft with aspirin." in last
    assert g.prompts[2].count("SOURCE") >= 1 and NOTE in g.prompts[2]       # each answer is grounded in the source


def test_cove_handles_no_questions():
    g = FakeGen({"draft": ["D"], "plan": ["I cannot think of questions."], "final": ["FINAL"]})
    final, n_qa = cove(g, "mimic", NOTE)
    assert (final, n_qa) == ("FINAL", 0)
    assert "(no verification questions were produced)" in g.prompts[-1]


def test_reflexion_stops_on_success_and_feeds_memory():
    g = FakeGen({"draft": ["T1", "T2"], "evaluate": ["- The dose is invented.", "NONE"],
                 "reflect": ["Do not invent doses."]})
    best, trials = reflexion(g, "mimic", NOTE, max_trials=3)
    assert (best, trials) == ("T2", 2)
    actor_prompts = [p for p in g.prompts if "Based on the patient record below" in p]
    assert len(actor_prompts) == 2
    assert "Lessons from earlier attempts" not in actor_prompts[0]
    assert "Do not invent doses." in actor_prompts[1]


def test_reflexion_returns_fewest_unsupported_when_never_successful():
    g = FakeGen({"draft": ["T1", "T2", "T3"],
                 "evaluate": ["- a\n- b", "- a\n- b\n- c", "- a"], "reflect": ["r"]})
    best, trials = reflexion(g, "mimic", NOTE, max_trials=3)
    assert (best, trials) == ("T3", 3)       # fewest unsupported statements wins (here the last trial, with 1)


def test_reflexion_free_text_verdict_counts_as_a_failure():
    g = FakeGen({"draft": ["T1", "T2"], "evaluate": ["The summary invents a dose.", "NONE"], "reflect": ["r"]})
    best, trials = reflexion(g, "mimic", NOTE, max_trials=3)
    assert (best, trials) == ("T2", 2)


def test_reflexion_ties_prefer_the_later_trial():
    g = FakeGen({"draft": ["T1", "T2"], "evaluate": ["- x", "- y"], "reflect": ["r"]})
    best, trials = reflexion(g, "mimic", NOTE, max_trials=2)
    assert (best, trials) == ("T2", 2)


def test_soap_prompts_use_soap_task():
    g = FakeGen({"draft": ["D"], "feedback": ["NO ISSUES"]})
    self_refine(g, "soap", "Doctor: hello. Patient: my back hurts.", max_iter=1)
    assert "SOAP Summary:" in g.prompts[0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
