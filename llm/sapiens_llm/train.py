"""A population of GRPO learners that share completions over a social network (MATH).

Each learner is a LoRA adapter on one frozen base model. Every iteration all learners receive the
same problems; vLLM samples GROUP_SIZE completions per problem for each learner (one engine serves
all adapters); each learner sends SHARE_PER_PROMPT of its completions per problem to its current
neighbours (a different random subset for each); then each learner takes one GRPO step on its own
and received completions (see grpo.py). TOPOLOGY=centralized is standard GRPO: one learner with the
whole population's sample budget.

    uv run python -m sapiens_llm.train --config configs/math.yaml --set TOPOLOGY=periodic --output runs/x
"""
import argparse
import json
import os
import pathlib
import tempfile
import time

import numpy as np
import torch
import yaml

from sapiens_llm import grpo, math_task, metrics
from sapiens_llm.topology import Topology

DEFAULTS = dict(
    MODEL="Qwen/Qwen2.5-1.5B-Instruct",
    TOPOLOGY="fully",              # independent | fully | ring | dynamic | periodic | centralized
    NUM_LEARNERS=4,
    ITERATIONS=100,
    PROMPTS_PER_ITER=16,
    GROUP_SIZE=8,                  # completions per problem per learner
    SHARE_PER_PROMPT=2,            # completions per problem sent to each neighbour
    P_SHARE=1.0,                   # probability that a learner shares in an iteration
    CLIP_IS=1.0,                   # truncation of the importance ratio of received completions
    LR=2e-5,
    LORA_RANK=32,
    LORA_ALPHA=64,
    MAX_TOKENS=1024,
    TEMPERATURE=1.0,
    MICRO_BATCH=4,
    # topology parameters, in iterations
    RING_K=2,
    GROUP_SIZE_DYNAMIC=2,
    P_VISIT=0.1,
    VISIT_DURATION=5,
    T_LOW=8,
    T_HIGH=2,
    # evaluation on a fixed subset of MATH-500
    EVAL_EVERY=20,
    EVAL_PROBLEMS=200,
    EVAL_SAMPLES=8,                # per learner (centralized: times NUM_LEARNERS)
    EVAL_TEMPERATURE=0.6,
    SEED=0,
    BACKEND="vllm",                # vllm | hf (slow, for local tests without a GPU)
    GPU_MEMORY=0.35,               # fraction of GPU memory for vLLM; the rest trains
)


class Population:
    """The base model with one LoRA adapter (and optimizer) per learner."""

    def __init__(self, cfg, n, device):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(cfg["MODEL"])
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(cfg["MODEL"], torch_dtype=dtype).to(device)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        lora = LoraConfig(r=cfg["LORA_RANK"], lora_alpha=cfg["LORA_ALPHA"], lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.names = [f"l{i}" for i in range(n)]
        self.model = get_peft_model(base, lora, adapter_name=self.names[0])
        for name in self.names[1:]:
            self.model.add_adapter(name, lora)
        self.params = {name: [p for k, p in self.model.named_parameters() if f".{name}." in k] for name in self.names}
        self.opt = {name: torch.optim.AdamW(self.params[name], lr=cfg["LR"], weight_decay=0.0) for name in self.names}
        self.cfg, self.device = cfg, device

    def _use(self, i):
        self.model.set_adapter(self.names[i])
        for p in self.params[self.names[i]]:
            p.requires_grad_(True)

    @torch.no_grad()
    def logprobs(self, i, samples):
        """Per-sample token log-probabilities of the completions under learner i (lists of tensors)."""
        self._use(i)
        self.model.eval()
        out = []
        for b in range(0, len(samples), self.cfg["MICRO_BATCH"]):
            chunk = samples[b:b + self.cfg["MICRO_BATCH"]]
            ids, att, mask = (t.to(self.device) for t in grpo.collate(chunk, self.pad))
            lp = grpo.token_logprobs(self.model, ids, att)
            out += [row[m > 0].float().cpu() for row, m in zip(lp, mask)]
        return out

    def update(self, i, samples, advantages):
        """One GRPO step for learner i on its batch (own and received completions)."""
        self._use(i)
        self.model.train()
        opt = self.opt[self.names[i]]
        opt.zero_grad(set_to_none=True)
        tokens = sum(len(s["completion_ids"]) for s in samples)
        total = 0.0
        for b in range(0, len(samples), self.cfg["MICRO_BATCH"]):
            chunk = samples[b:b + self.cfg["MICRO_BATCH"]]
            ids, att, mask = (t.to(self.device) for t in grpo.collate(chunk, self.pad))
            lp = grpo.token_logprobs(self.model, ids, att)
            behaviour = torch.zeros_like(lp)
            for r, s in enumerate(chunk):
                if s["received"]:
                    behaviour[r, mask[r] > 0] = s["behaviour"].to(self.device)
            received = torch.tensor([s["received"] for s in chunk], device=self.device)
            adv = torch.tensor(advantages[b:b + len(chunk)], device=self.device, dtype=torch.float32)
            loss = grpo.policy_loss(lp, behaviour, adv, mask, received, self.cfg["CLIP_IS"]) / tokens
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(self.params[self.names[i]], 1.0)
        opt.step()
        return total

    def save(self, i, root):
        self.model.save_pretrained(root, selected_adapters=[self.names[i]])
        path = pathlib.Path(root) / self.names[i] if self.names[i] != "default" else pathlib.Path(root)
        assert (path / "adapter_config.json").exists(), path
        return str(path)


class VLLMGenerator:
    def __init__(self, cfg, n):
        from vllm import LLM
        self.llm = LLM(cfg["MODEL"], enable_lora=True, max_loras=n, max_lora_rank=cfg["LORA_RANK"], dtype="bfloat16",
                       gpu_memory_utilization=cfg["GPU_MEMORY"], seed=cfg["SEED"], enable_prefix_caching=True)
        self.next_id = 1

    def generate(self, prompts, adapter_paths, n, temperature, max_tokens, seed):
        """prompts: token-id lists; adapter_paths: one per learner. Returns [learner][prompt][sample] =
        (completion token ids, text)."""
        from vllm import SamplingParams, TokensPrompt
        from vllm.lora.request import LoRARequest
        requests, loras = [], []
        for i, path in enumerate(adapter_paths):
            lora = LoRARequest(f"learner{i}-{self.next_id}", self.next_id, path)
            self.next_id += 1
            requests += [TokensPrompt(prompt_token_ids=p) for p in prompts]
            loras += [lora] * len(prompts)
        params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, seed=seed)
        outs = self.llm.generate(requests, params, lora_request=loras, use_tqdm=False)
        k = len(prompts)
        return [[[(list(c.token_ids), c.text) for c in outs[i * k + p].outputs] for p in range(k)]
                for i in range(len(adapter_paths))]


class HFGenerator:
    """Generation with the training model itself; slow, for tests without vLLM or a GPU."""

    def __init__(self, population):
        self.pop = population

    @torch.no_grad()
    def generate(self, prompts, adapter_paths, n, temperature, max_tokens, seed):
        torch.manual_seed(seed)
        result = []
        for i in range(len(adapter_paths)):
            self.pop._use(i)
            self.pop.model.eval()
            per_prompt = []
            for p in prompts:
                ids = torch.tensor([p], device=self.pop.device)
                out = self.pop.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                              temperature=temperature, max_new_tokens=max_tokens,
                                              num_return_sequences=n, pad_token_id=self.pop.pad)
                comps = []
                for row in out[:, len(p):].tolist():
                    row = row[:row.index(self.pop.tok.eos_token_id) + 1] if self.pop.tok.eos_token_id in row else row
                    row = [t for t in row if t != self.pop.pad] or [self.pop.tok.eos_token_id]
                    comps.append((row, self.pop.tok.decode(row, skip_special_tokens=True)))
                per_prompt.append(comps)
            result.append(per_prompt)
        return result


def evaluate(cfg, pop, gen, problems, adapter_paths, n_samples, it):
    prompts = [math_task.prompt_ids(pop.tok, q["problem"]) for q in problems]
    outs = gen.generate(prompts, adapter_paths, n_samples, cfg["EVAL_TEMPERATURE"], cfg["MAX_TOKENS"], cfg["SEED"] + 10_000 + it)
    correct = np.array([[[math_task.reward(text, q["answer"]) > 0 for _, text in outs[i][p]]
                         for p, q in enumerate(problems)] for i in range(len(adapter_paths))])
    answers = [[[metrics.last_boxed(text) for _, text in outs[i][p]] for p in range(len(problems))]
               for i in range(len(adapter_paths))]
    ks = [k for k in (1, 2, 4, 8, 16) if k <= cfg["EVAL_SAMPLES"]]
    centralized = cfg["TOPOLOGY"] == "centralized"
    result = dict(iteration=it, pass_at_1=[float(x) for x in correct.mean((1, 2))],
                  conformity=metrics.conformity(answers) if len(adapter_paths) > 1 else 1.0)
    for k in ks:
        result[f"pass_at_{k}"] = metrics.learner_pass_at_k(correct, k)
        # each learner draws k samples; the centralized learner draws the whole population's k * N
        result[f"group_pass_at_{k}"] = metrics.group_pass_at_k(correct, k * cfg["NUM_LEARNERS"] if centralized else k)
    return result


def run(cfg, output):
    out = pathlib.Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    rng = np.random.default_rng(cfg["SEED"])
    torch.manual_seed(cfg["SEED"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    centralized = cfg["TOPOLOGY"] == "centralized"
    n = 1 if centralized else cfg["NUM_LEARNERS"]
    group = cfg["GROUP_SIZE"] * (cfg["NUM_LEARNERS"] if centralized else 1)
    n_eval = cfg["EVAL_SAMPLES"] * (cfg["NUM_LEARNERS"] if centralized else 1)

    pop = Population(cfg, n, device)
    gen = VLLMGenerator(cfg, n) if cfg["BACKEND"] == "vllm" else HFGenerator(pop)
    topo = Topology(dict(cfg, NUM_LEARNERS=n), rng)
    train, test = math_task.load("train"), math_task.load("test")
    eval_problems = [test[i] for i in np.sort(np.random.default_rng(0).choice(len(test), min(cfg["EVAL_PROBLEMS"], len(test)), replace=False))]
    log = open(out / "metrics.jsonl", "a")
    tmp = tempfile.mkdtemp(prefix="sapiens_llm_")

    for it in range(cfg["ITERATIONS"] + 1):
        start = time.time()
        paths = [pop.save(i, os.path.join(tmp, f"it{it}")) for i in range(n)]
        if it % cfg["EVAL_EVERY"] == 0 or it == cfg["ITERATIONS"]:
            log.write(json.dumps(dict(kind="eval", **evaluate(cfg, pop, gen, eval_problems, paths, n_eval, it))) + "\n")
            log.flush()
        if it == cfg["ITERATIONS"]:
            break

        problems = [train[j] for j in rng.choice(len(train), cfg["PROMPTS_PER_ITER"], replace=False)]
        prompts = [math_task.prompt_ids(pop.tok, q["problem"]) for q in problems]
        outs = gen.generate(prompts, paths, group, cfg["TEMPERATURE"], cfg["MAX_TOKENS"], cfg["SEED"] * 100_003 + it)
        own = [[dict(prompt_ids=prompts[p], completion_ids=ids, prompt=p, received=False, sender=i,
                     reward=math_task.reward(text, problems[p]["answer"]))
                for p in range(len(problems)) for ids, text in outs[i][p]] for i in range(n)]

        # sharing: each active sender gives each neighbour SHARE_PER_PROMPT random completions per problem
        adj = topo.adjacency(it) & (rng.random(n) < cfg["P_SHARE"])[:, None]
        inbox = [[] for _ in range(n)]
        for j in range(n):
            for i in np.flatnonzero(adj[j]):
                for p in range(len(problems)):
                    pool = [s for s in own[j] if s["prompt"] == p]
                    for s in rng.choice(len(pool), min(cfg["SHARE_PER_PROMPT"], len(pool)), replace=False):
                        inbox[i].append(dict(pool[s], received=True))
        for j in range(n):   # behaviour log-probabilities under the sender, before anyone updates
            sent = [s for box in inbox for s in box if s["sender"] == j]
            for s, lp in zip(sent, pop.logprobs(j, sent) if sent else []):
                s["behaviour"] = lp

        losses = []
        for i in range(n):
            batch = own[i] + inbox[i]
            adv = grpo.group_advantages([s["reward"] for s in batch], [s["prompt"] for s in batch])
            losses.append(pop.update(i, batch, adv))
        topo.step(it)

        record = dict(kind="train", iteration=it, seconds=time.time() - start, loss=losses,
                      reward=[float(np.mean([s["reward"] for s in own[i]])) for i in range(n)],
                      received=[len(box) for box in inbox], degree=adj.sum(1).tolist(),
                      completion_tokens=float(np.mean([len(s["completion_ids"]) for o in own for s in o])))
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(f"it {it:4d}  reward {np.mean(record['reward']):.3f}  received {sum(record['received']):5d}  "
              f"tokens {record['completion_tokens']:6.0f}  {record['seconds']:.0f}s", flush=True)

    for i in range(n):
        pop.save(i, out / "adapters")


def parse_value(v):
    if v in ("None", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return yaml.safe_load(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update({k: parse_value(v) if isinstance(v, str) else v for k, v in yaml.safe_load(open(args.config)).items()})
    for item in args.set:
        k, v = item.split("=", 1)
        cfg[k] = parse_value(v)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config keys {sorted(unknown)}")
    run(cfg, args.output)


if __name__ == "__main__":
    main()
