"""Write eval results (from evals/run_eval.py --out) into the README results section.

    python scripts/update_readme_results.py docs/eval_groq.json

Replaces everything between <!-- EVAL_RESULTS_START --> and <!-- EVAL_RESULTS_END -->.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- EVAL_RESULTS_START -->", "<!-- EVAL_RESULTS_END -->"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render(data: dict) -> tuple[str, str]:
    meta, summary = data["meta"], data["summary"]
    by = {r["setup"]: r for r in summary}
    auto = by.get("auto")
    base = summary[-1]
    upstream = {name: m["upstream"] for name, m in meta["models"].items()}

    lines = [START, "",
             f"Measured on **{meta['date']}** with real Groq models, {meta['dataset_size']} prompts, "
             "temperature 0, cache bypassed. Raw results: [`docs/eval_groq.json`](docs/eval_groq.json).", "",
             "| Setup | Accuracy | Easy | Hard | Cost (USD) | p50 latency | Errors |",
             "|---|---|---|---|---|---|---|"]
    for r in summary:
        label = "`auto` (smart routing)" if r["setup"] == "auto" else \
            f"always `{upstream.get(r['setup'], r['setup'])}`"
        lines.append(f"| {label} | **{pct(r['accuracy'])}** | {pct(r['accuracy_easy'])} | {pct(r['accuracy_hard'])} "
                     f"| ${r['total_cost_usd']:.5f} | {r['p50_latency_ms']:.0f} ms | {r['errors']} |")
    lines.append("")

    bullet = ""
    if auto and auto is not base and base["total_cost_usd"]:
        saved = (1 - auto["total_cost_usd"] / base["total_cost_usd"]) * 100
        rel = (auto["accuracy"] / base["accuracy"] * 100) if base["accuracy"] else 0
        served = ", ".join(f"{n} → `{upstream.get(k, k)}`" for k, n in sorted(auto["served_by"].items(),
                                                                         key=lambda kv: -kv[1]) if k)
        routed = auto.get("routed_as_intended")
        lines.append(f"**Smart routing cost {saved:.0f}% less than always using the large model, "
                     f"keeping {rel:.0f}% of its accuracy** ({pct(auto['accuracy'])} vs {pct(base['accuracy'])}).")
        lines.append("")
        lines.append(f"`auto` traffic split: {served}."
                     + (f" {routed}/{meta['dataset_size']} prompts went to the intended tier." if routed else ""))
        bullet = (f"Cut LLM spend by {saved:.0f}% versus always using {upstream.get(base['setup'], base['setup'])} "
                  f"while keeping {rel:.0f}% of its accuracy ({pct(auto['accuracy'])} vs {pct(base['accuracy'])}) "
                  f"on a {meta['dataset_size']}-prompt eval with real Groq models")
    lines += ["", "Caveat: 30 prompts is a small eval, so treat these as directional numbers.", "", END]
    return "\n".join(lines), bullet


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "eval_groq.json")
    data = json.loads(src.read_text())
    block, bullet = render(data)
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        sys.exit("README is missing the EVAL_RESULTS markers")
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    readme.write_text(text)
    print("README updated.")
    if bullet:
        print(f"\nSuggested resume bullet:\n  - {bullet}")
        (ROOT / "docs" / "resume_bullet.txt").write_text(bullet + "\n")


if __name__ == "__main__":
    main()
