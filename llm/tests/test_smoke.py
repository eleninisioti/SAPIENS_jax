"""End to end on CPU with a small model and Hugging Face generation (vLLM needs Linux and a GPU)."""
import json

import pytest

from sapiens_llm.train import DEFAULTS, run


@pytest.mark.parametrize("topology", ["fully", "centralized"])
def test_one_iteration(tmp_path, topology):
    cfg = dict(DEFAULTS, MODEL="Qwen/Qwen2.5-0.5B-Instruct", BACKEND="hf", TOPOLOGY=topology, NUM_LEARNERS=2,
               ITERATIONS=1, PROMPTS_PER_ITER=2, GROUP_SIZE=2, SHARE_PER_PROMPT=1, MAX_TOKENS=12, MICRO_BATCH=2,
               EVAL_EVERY=1, EVAL_PROBLEMS=2, EVAL_SAMPLES=2, LORA_RANK=4, LORA_ALPHA=8)
    run(cfg, tmp_path)
    records = [json.loads(line) for line in open(tmp_path / "metrics.jsonl")]
    train = [r for r in records if r["kind"] == "train"]
    evals = [r for r in records if r["kind"] == "eval"]
    assert len(train) == 1 and len(evals) == 2
    if topology == "fully":
        assert train[0]["received"] == [2, 2]       # 1 completion x 2 problems from the other learner
    assert "group_pass_at_2" in evals[-1]
    assert (tmp_path / "adapters").exists()
