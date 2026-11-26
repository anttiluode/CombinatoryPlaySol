"""Optional lexical comparison with native Qwen continuation; run locally."""
import argparse
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--play", default="play-results.json")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--out", default="continuation-results.json")
    a = p.parse_args()
    if a.samples < 1 or a.tokens < 1:
        p.error("samples and tokens must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = json.loads(Path(a.play).read_text(encoding="utf-8"))
    words = d["words"]
    cues = [words[i] for i in d["cues"]]
    tag = [words[i] for i in d["options"].get("tag", [])]
    # The external episode values are not provided to the model. This is
    # a descriptive comparison, not a matched-information performance test.
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    dtype = torch.float16 if a.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype, local_files_only=a.local_files_only
    ).to(a.device).eval()
    def lexical(text):
        return set(re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE))
    bank = {w.casefold() for w in words}
    play = {
        name: {words[z["i"]].casefold() for row in arm["trace"] for z in row["workspace"]}
        for name, arm in d["arms"].items()
    }
    prompts = {
        "continuation": ", ".join(cues) + ",",
        "association": "Starting concepts: " + ", ".join(cues)
            + "\nRelated concepts and associations:",
    }
    if tag:
        prompts["tagged_association"] = "Context: " + ", ".join(tag) + "\n" + prompts["association"]
    result = []
    for name, prompt in prompts.items():
        inp = tokenizer(prompt, return_tensors="pt").to(a.device)
        for sample in range(a.samples):
            torch.manual_seed(a.seed + sample)
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=a.tokens, do_sample=True,
                    temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            observed = lexical(text) & bank
            overlaps = {}
            for arm, seen in play.items():
                both = observed & seen
                overlaps[arm] = {"intersection": sorted(both),
                    "jaccard": len(both) / max(1, len(observed | seen)),
                    "generated_bank_words": len(observed), "play_words": len(seen)}
            result.append(dict(prompt_type=name, prompt=prompt, sample=sample, text=text,
                               bank_words=sorted(observed), overlaps=overlaps))
            print(name, sample, text, flush=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "cues": cues, "tag": tag,
        "samples": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", a.out, "(lexical overlap, not a reasoning or semantic quality score)")


if __name__ == "__main__":
    main()
