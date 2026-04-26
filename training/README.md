# Training Assets

This folder contains the minimal training path requested by the OpenEnv Hackathon checklist.

Recommended flow:

1. Generate heuristic demonstrations:
   - `python training/generate_sft_dataset.py --episodes-per-task 8`
2. Fine-tune a small instruction model with HF TRL:
   - `python training/train_trl_sft.py --dataset artifacts/training/cybersoc_sft_train.jsonl --output-dir artifacts/training/trl-smollm2`
3. Commit or link the generated artifacts:
   - `artifacts/training/*/training_loss.png`
   - `artifacts/training/*/score_comparison.png`
   - `artifacts/training/*/training_summary.json`
4. Link the final notebook/video/blog/slide deck from the main `README.md`.

The notebook in this folder is designed for Colab and mirrors the same script flow.
