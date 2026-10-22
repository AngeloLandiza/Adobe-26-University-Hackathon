import json, sys, time
from llama_cpp import Llama
model, prompts, out, n, threads = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
SYSTEM = "You are an ad layout engine. Given a source ad as JSON and a target canvas, return the resized layout as JSON."
t0 = time.time()
llm = Llama(model_path=model, n_ctx=4096, n_gpu_layers=0, n_threads=threads, verbose=False)
print(f"load {time.time()-t0:.1f}s, threads={threads}", flush=True)
rows = [json.loads(l) for l in open(prompts)][:n]
with open(out, "w") as f:
    for r in rows:
        t = time.time()
        raw = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{r['prompt']}<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n")   # exactly the training format
        res = llm(raw, max_tokens=900, temperature=0.0, stop=["<|im_end|>"])
        u = res["usage"]; dt = time.time() - t
        txt = res["choices"][0]["text"]
        f.write(json.dumps({"cell": r["cell"], "output": txt}) + "\n"); f.flush()
        print(f"{r['cell'][:45]:45s} {dt:5.1f}s  prompt {u['prompt_tokens']} tok, output {u['completion_tokens']} tok ({u['completion_tokens']/dt:.1f} tok/s overall)", flush=True)
