"""Minimal Hugging Face TRL SFT pipeline for CyberSOC OpenEnv++."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from cybersoc_openenv.client import CyberSOCEnvClient, InProcessCyberSOCEnvClient
from cybersoc_openenv.graders import grade_state
from cybersoc_openenv.models import CyberSOCAction
from cybersoc_openenv.training import DEFAULT_TASKS, SYSTEM_PROMPT, build_prompt_payload

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
FALLBACK_ACTION = {"action_type": "noop", "justification": "training_fallback"}


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _extract_json(text: str) -> dict[str, Any]:
    match = JSON_BLOCK_RE.search(text)
    if not match:
        return dict(FALLBACK_ACTION)
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(FALLBACK_ACTION)


def _build_env_client(env_base_url: str | None):
    if env_base_url:
        return CyberSOCEnvClient(base_url=env_base_url)

    from server.app import app

    return InProcessCyberSOCEnvClient(app)


def _format_messages(messages: list[dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def _prepare_dataset(dataset_path: str, tokenizer):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return dataset.map(
        lambda example: {"text": _format_messages(example["messages"], tokenizer)},
        remove_columns=dataset.column_names,
    )


def _llm_action(model, tokenizer, observation, max_new_tokens: int) -> CyberSOCAction:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(build_prompt_payload(observation), indent=2)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_tokens = inputs["input_ids"].shape[1]
    text = tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True)
    payload = _extract_json(text)
    try:
        return CyberSOCAction.model_validate(payload)
    except Exception:
        return CyberSOCAction.model_validate(FALLBACK_ACTION)


def _evaluate_model(model, tokenizer, env_base_url: str | None, max_new_tokens: int) -> dict[str, Any]:
    client = _build_env_client(env_base_url)
    task_results: list[dict[str, Any]] = []
    try:
        for task_id in DEFAULT_TASKS:
            print(f"[eval] starting task={task_id}", flush=True)
            observation = client.reset(task_id=task_id, seed=7).observation
            done = False
            while not done:
                action = _llm_action(model, tokenizer, observation, max_new_tokens=max_new_tokens)
                step = client.step(action)
                observation = step.observation
                done = step.done
            state = client.state()
            task_results.append(
                {
                    "task_id": task_id,
                    "score": grade_state(state),
                    "steps": state.step_count,
                    "terminal_reason": state.terminal_reason,
                }
            )
    finally:
        client.close()

    average_score = round(sum(result["score"] for result in task_results) / len(task_results), 4)
    return {"task_results": task_results, "average_score": average_score}


def _svg_line_chart(
    points: list[tuple[float, float]],
    output_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    color: str = "#04d9ff",
) -> None:
    width = 900
    height = 520
    left = 90
    right = 40
    top = 60
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    xs = [point[0] for point in points] or [0.0, 1.0]
    ys = [point[1] for point in points] or [0.0, 1.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.1
    y_min -= y_pad
    y_max += y_pad

    def x_px(x: float) -> float:
        return left + ((x - x_min) / (x_max - x_min)) * plot_width

    def y_px(y: float) -> float:
        return top + plot_height - ((y - y_min) / (y_max - y_min)) * plot_height

    path = " ".join(
        ("M" if index == 0 else "L") + f" {x_px(x):.2f} {y_px(y):.2f}"
        for index, (x, y) in enumerate(points)
    )

    x_ticks = 5
    y_ticks = 5
    grid_lines: list[str] = []
    labels: list[str] = []

    for tick in range(x_ticks + 1):
        ratio = tick / x_ticks
        value = x_min + ratio * (x_max - x_min)
        px = left + ratio * plot_width
        grid_lines.append(
            f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_height}" stroke="#223347" stroke-width="1" />'
        )
        labels.append(
            f'<text x="{px:.2f}" y="{height - 32}" fill="#9fb3c8" font-size="12" text-anchor="middle">{value:.0f}</text>'
        )

    for tick in range(y_ticks + 1):
        ratio = tick / y_ticks
        value = y_min + ratio * (y_max - y_min)
        py = top + plot_height - ratio * plot_height
        grid_lines.append(
            f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_width}" y2="{py:.2f}" stroke="#223347" stroke-width="1" />'
        )
        labels.append(
            f'<text x="{left - 12}" y="{py + 4:.2f}" fill="#9fb3c8" font-size="12" text-anchor="end">{value:.3f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0b1220"/>
<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#101a2b" stroke="#35516e" stroke-width="1.5"/>
{''.join(grid_lines)}
<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<text x="{width / 2:.2f}" y="32" fill="#f5f7fa" font-size="24" text-anchor="middle" font-family="Cambria, Georgia, serif">{html.escape(title)}</text>
<text x="{width / 2:.2f}" y="{height - 10}" fill="#c7d5e0" font-size="15" text-anchor="middle">{html.escape(x_label)}</text>
<text x="24" y="{height / 2:.2f}" fill="#c7d5e0" font-size="15" text-anchor="middle" transform="rotate(-90 24 {height / 2:.2f})">{html.escape(y_label)}</text>
{''.join(labels)}
</svg>"""
    output_path.write_text(svg, encoding="utf-8")


def _svg_bar_chart(
    labels_values: list[tuple[str, float]],
    output_path: Path,
    *,
    title: str,
    y_label: str,
) -> None:
    width = 760
    height = 520
    left = 90
    right = 40
    top = 60
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min = 0.0
    y_max = max(max((value for _, value in labels_values), default=1.0), 1.0)

    def y_px(y: float) -> float:
        return top + plot_height - ((y - y_min) / (y_max - y_min)) * plot_height

    bar_width = plot_width / max(len(labels_values) * 1.8, 2.0)
    gap = bar_width * 0.8
    start_x = left + gap

    bars: list[str] = []
    labels: list[str] = []
    colors = ["#8b949e", "#2ea043", "#04d9ff", "#f2cc60"]

    for index, (label, value) in enumerate(labels_values):
        x = start_x + index * (bar_width + gap)
        y = y_px(value)
        height_px = top + plot_height - y
        color = colors[index % len(colors)]
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height_px:.2f}" fill="{color}" rx="8" />'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - 44}" fill="#c7d5e0" font-size="13" text-anchor="middle">{html.escape(label)}</text>'
        )
        labels.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 10:.2f}" fill="#f5f7fa" font-size="13" text-anchor="middle">{value:.4f}</text>'
        )

    y_ticks = 5
    grid_lines: list[str] = []
    tick_labels: list[str] = []
    for tick in range(y_ticks + 1):
        ratio = tick / y_ticks
        value = y_min + ratio * (y_max - y_min)
        py = top + plot_height - ratio * plot_height
        grid_lines.append(
            f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_width}" y2="{py:.2f}" stroke="#223347" stroke-width="1" />'
        )
        tick_labels.append(
            f'<text x="{left - 12}" y="{py + 4:.2f}" fill="#9fb3c8" font-size="12" text-anchor="end">{value:.2f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0b1220"/>
<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#101a2b" stroke="#35516e" stroke-width="1.5"/>
{''.join(grid_lines)}
{''.join(bars)}
<text x="{width / 2:.2f}" y="32" fill="#f5f7fa" font-size="24" text-anchor="middle" font-family="Cambria, Georgia, serif">{html.escape(title)}</text>
<text x="24" y="{height / 2:.2f}" fill="#c7d5e0" font-size="15" text-anchor="middle" transform="rotate(-90 24 {height / 2:.2f})">{html.escape(y_label)}</text>
{''.join(tick_labels)}
{''.join(labels)}
</svg>"""
    output_path.write_text(svg, encoding="utf-8")


def _plot_training_loss(log_history: list[dict[str, Any]], output_path: Path) -> Path | None:
    points = [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry and "step" in entry]
    if not points:
        return None
    plt = _try_import_matplotlib()
    if plt is not None:
        steps, losses = zip(*points)
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, losses, color="#04d9ff", linewidth=2)
        plt.title("CyberSOC HF TRL Training Loss")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path
    fallback_path = output_path.with_suffix(".svg")
    _svg_line_chart(
        points,
        fallback_path,
        title="CyberSOC HF TRL Training Loss",
        x_label="Training step",
        y_label="Loss",
    )
    return fallback_path


def _plot_score_comparison(before_after: dict[str, Any], output_path: Path) -> Path:
    labels = ["baseline", "trained"]
    scores = [before_after["baseline"]["average_score"], before_after["trained"]["average_score"]]
    plt = _try_import_matplotlib()
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, scores, color=["#8b949e", "#2ea043"])
        plt.ylim(0.0, 1.0)
        plt.ylabel("Average task score")
        plt.title("CyberSOC Before vs After Training")
        for index, score in enumerate(scores):
            plt.text(index, score + 0.02, f"{score:.4f}", ha="center")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return output_path
    fallback_path = output_path.with_suffix(".svg")
    _svg_bar_chart(
        list(zip(labels, scores)),
        fallback_path,
        title="CyberSOC Before vs After Training",
        y_label="Average task score",
    )
    return fallback_path


def _device_summary() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal CyberSOC SFT model with HF TRL.")
    parser.add_argument("--dataset", required=True, help="Path to the JSONL SFT dataset.")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints and plots.")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--env-base-url", default=None, help="Optional external CyberSOC API base URL.")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--force-cpu-eval",
        action="store_true",
        help="Run before/after environment evaluation even without CUDA. This can be very slow on CPU.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    has_cuda = torch.cuda.is_available()
    should_run_eval = has_cuda or args.force_cpu_eval

    print(f"[train] device={_device_summary()}", flush=True)
    if not has_cuda and not args.force_cpu_eval:
        print(
            "[train] CPU-only runtime detected; skipping before/after environment evaluation. "
            "Use --force-cpu-eval if you really want local eval on CPU.",
            flush=True,
        )

    print(f"[train] loading tokenizer={args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[train] loading model={args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
    )

    print(f"[train] preparing dataset={args.dataset}", flush=True)
    train_dataset = _prepare_dataset(args.dataset, tokenizer)
    print(f"[train] dataset_rows={len(train_dataset)}", flush=True)

    baseline_metrics = {"task_results": [], "average_score": None, "skipped": not should_run_eval}
    if should_run_eval:
        print("[train] running baseline environment evaluation", flush=True)
        baseline_metrics = _evaluate_model(model, tokenizer, args.env_base_url, args.max_new_tokens)

    print("[train] starting HF TRL fine-tuning", flush=True)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_seq_length,
            logging_steps=1,
            save_steps=args.max_steps,
            report_to=[],
            dataset_text_field="text",
            bf16=False,
            bf16_full_eval=False,
            fp16=False,
            fp16_full_eval=False,
            use_cpu=not has_cuda,
            optim="adamw_torch",
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    trained_metrics = {"task_results": [], "average_score": None, "skipped": not should_run_eval}
    if should_run_eval:
        print("[train] running trained environment evaluation", flush=True)
        trained_metrics = _evaluate_model(model, tokenizer, args.env_base_url, args.max_new_tokens)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model,
        "device": _device_summary(),
        "baseline": baseline_metrics,
        "trained": trained_metrics,
    }

    print("[train] writing plots and summary", flush=True)
    training_loss_artifact = _plot_training_loss(trainer.state.log_history, output_dir / "training_loss.png")
    score_artifact = None
    if should_run_eval:
        score_artifact = _plot_score_comparison(summary, output_dir / "score_comparison.png")
    summary["artifacts"] = {
        "training_loss": str(training_loss_artifact) if training_loss_artifact else None,
        "score_comparison": str(score_artifact) if score_artifact else None,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
