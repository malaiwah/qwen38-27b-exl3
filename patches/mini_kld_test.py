#!/usr/bin/env python3
"""Mini-KLD test: compare logit distributions between levers ON vs OFF."""

import json
import time
import requests
import math
import sys
import os

URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen3.8-27B"

PROMPTS = [
    "The quick brown fox", "In a galaxy far far away",
    "The mitochondria is the powerhouse", "Quantum entanglement occurs when",
    "The Treaty of Westphalia established", "In machine learning, gradient descent",
    "The weather today is", "To be or not to be",
    "The capital of France is", "Water boils at 100 degrees",
    "The Python programming language", "Neural networks are inspired by",
    "The speed of light is", "Climate change refers to",
    "The French Revolution began in", "DNA contains the genetic instructions",
    "The Great Wall of China", "Artificial intelligence is",
    "The stock market fluctuates", "Photosynthesis converts sunlight",
]

def collect_logits(label):
    results = []
    for i, prompt in enumerate(PROMPTS):
        resp = requests.post(URL, json={
            "model": MODEL, "prompt": prompt,
            "max_tokens": 1, "temperature": 0, "logprobs": 20,
        }, timeout=60)
        data = resp.json()
        choice = data["choices"][0]
        lp = choice.get("logprobs", {})
        top_lp = lp.get("top_logprobs", [{}])[0] if lp.get("top_logprobs") else {}
        chosen = lp.get("tokens", [""])[0] if lp.get("tokens") else ""
        chosen_lp = lp.get("token_logprobs", [0])[0] if lp.get("token_logprobs") else 0
        # Ensure chosen token is in the dict
        if chosen and chosen not in top_lp:
            top_lp[chosen] = chosen_lp
        results.append({"prompt": prompt, "logprobs": top_lp, "chosen": chosen})
        if (i + 1) % 5 == 0:
            print(f"  [{label}] {i+1}/{len(PROMPTS)} collected")
        time.sleep(0.05)
    return results

def compute_kld(p, q):
    all_tokens = set(p.keys()) | set(q.keys())
    min_lp = min(list(p.values()) + list(q.values())) if p and q else -100
    smooth = min_lp - 10
    kld = 0.0
    for token in all_tokens:
        p_lp = p.get(token, smooth)
        q_lp = q.get(token, smooth)
        p_prob = math.exp(p_lp)
        q_prob = math.exp(q_lp)
        if p_prob > 0 and q_prob > 0:
            kld += p_prob * (p_lp - q_lp)
    return kld

def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    print(f"Collecting logits for {label}...")
    results = collect_logits(label)
    with open(f"/tmp/kld_{label}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Collected {len(results)} prompts for {label}")
    
    if os.path.exists("/tmp/kld_on.json") and os.path.exists("/tmp/kld_off.json"):
        with open("/tmp/kld_on.json") as f: on_r = json.load(f)
        with open("/tmp/kld_off.json") as f: off_r = json.load(f)
        print(f"\n=== KLD Comparison (levers ON vs OFF) ===")
        klds = []
        for on, off in zip(on_r, off_r):
            kld = compute_kld(on["logprobs"], off["logprobs"])
            klds.append(kld)
            print(f"  {on['prompt'][:35]:>35}: KLD={kld:.8f}")
        mean_kld = sum(klds) / len(klds) if klds else 0
        max_kld = max(klds) if klds else 0
        print(f"\n  Mean KLD: {mean_kld:.8f}")
        print(f"  Max KLD:  {max_kld:.8f}")
        print(f"  Threshold: 0.0000290 (2.9e-5)")
        print(f"  Result: {'PASS' if mean_kld < 2.9e-5 else 'FAIL'}")

if __name__ == "__main__":
    main()
