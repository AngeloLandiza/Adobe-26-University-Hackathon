# colab/

`chameleon_layout_distill.ipynb` fine-tunes a small open model (LoRA on Qwen3-4B-Instruct by default; change `BASE` to retrain on another approved model) to propose ad layouts the way Claude Sonnet does, from the source ad's JSON and a target canvas.

Inputs (upload to Google Drive `MyDrive/chameleon-distill/`): `train.jsonl` and `eval_prompts.jsonl` from `training-data/llm-distill/` (built by `python -m dev.distill.llm_layout datagen` and `eval-prompts`).
Outputs (same Drive folder): `eval_outputs.jsonl` (bring it back and run `python -m dev.distill.llm_layout ingest --outputs <file>`), `lora_adapter/`, `gguf/`, `train_log.json`.
Checkpoints go to Drive every 25 steps; re-running the notebook resumes.
