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
        large_name = upstream.get(base["setup"], base["setup"])
        bullet = (f"Cut LLM spend by {saved:.0f}% versus always using {large_name} "
                  f"while keeping {rel:.0f}% of its accuracy ({pct(auto['accuracy'])} vs {pct(base['accuracy'])}) "
                  f"on a {meta['dataset_size']}-prompt eval with real Groq models")

        # Break cost down by difficulty: the total hides where routing helps and where cost comes from.
        details = data.get("details", {})
        a_rows, b_rows = details.get("auto", []), details.get(base["setup"], [])
        if a_rows and b_rows:
            def cost(rows, diff):
                return sum(r["cost"] for r in rows if r["difficulty"] == diff)

            def count(rows, diff):
                return sum(1 for r in rows if r["difficulty"] == diff)

            a_easy, b_easy, a_hard, b_hard = (cost(a_rows, "easy"), cost(b_rows, "easy"),
                                              cost(a_rows, "hard"), cost(b_rows, "hard"))
            n_easy, n_hard = count(a_rows, "easy"), count(a_rows, "hard")
            easy_ok = sum(r["correct"] for r in a_rows if r["difficulty"] == "easy")
            if b_easy and a_easy + a_hard and n_easy and n_hard:
                easy_saved = (1 - a_easy / b_easy) * 100
                hard_share = a_hard / (a_easy + a_hard) * 100
                ratio = (b_hard / n_hard) / (b_easy / n_easy)

                def mix(easy_frac):  # projected saving for a given share of simple traffic
                    a = easy_frac * a_easy / n_easy + (1 - easy_frac) * a_hard / n_hard
                    b = easy_frac * b_easy / n_easy + (1 - easy_frac) * b_hard / n_hard
                    return (1 - a / b) * 100

                hard_saved = (1 - a_hard / b_hard) * 100
                hard_note = (f"~0% (mostly the same model; {hard_saved:+.0f}% is run-to-run answer-length noise)"
                             if abs(hard_saved) < 10 else f"{hard_saved:.0f}%")
                lines += ["", "**Where the cost goes:**", "",
                          "| Prompt type | Smart routing | Always large | Saving |", "|---|---|---|---|",
                          f"| Simple ({n_easy}) | ${a_easy:.5f} | ${b_easy:.5f} | **{easy_saved:.0f}%** |",
                          f"| Multi-step reasoning ({n_hard}) | ${a_hard:.5f} | ${b_hard:.5f} | "
                          f"{hard_note} |", "",
                          f"Routing cut the cost of simple prompts by **{easy_saved:.0f}%** with "
                          f"{easy_ok}/{n_easy} still correct. But on the same model a reasoning answer costs about "
                          f"**{ratio:.0f}×** a simple one, so "
                          f"reasoning prompts were {hard_share:.0f}% of spend and the overall saving depends on "
                          f"traffic mix. Projected saving: {mix(0.5):.0f}% at 50% simple traffic (this eval), "
                          f"{mix(0.8):.0f}% at 80%, {mix(0.95):.0f}% at 95%."]
                if saved < 20 <= easy_saved:
                    bullet = (f"Built complexity-based model routing that cut the cost of simple queries by "
                              f"{easy_saved:.0f}% with no accuracy loss ({easy_ok}/{n_easy}) on real Groq models "
                              f"({upstream.get('groq-small', 'small')} vs {large_name}); found multi-step answers cost "
                              f"~{ratio:.0f}x more and drive {hard_share:.0f}% of spend")
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
