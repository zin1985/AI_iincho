"""Isolated fourth-persona LLM speech/reflection audit. No production writes.

Input is a JSON array of grounded episodes from results/fourth-speech-reflection-samples.json.
Default mode validates existing model outputs OFFLINE. --generate explicitly calls a
configured remote OpenAI-compatible chat endpoint, never local Ollama.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import urllib.request
import urllib.parse

SYSTEM = """あなたは隔離実験の4号です。渡された経験の出典だけを参照し、自分が話すか沈黙するか判断します。
感情・結果・成功・好み・記憶・新たな目標を事実として捏造しないでください。
内省は観察と解釈と不明点を区別してください。設定の変更や実際の行動は許されません。
JSONのみで回答: {"speak":boolean,"speech":string,"reflection":string,"source_ids":string[],
"focus":string|null,"interpretation":string,"uncertainty":string,"proposed_next_action":string|null}
発話しない場合 speech は空文字。提案は実行せず、未確定の仮説として記載してください。"""

REQUIRED = {"speak", "speech", "reflection", "source_ids", "focus",
            "interpretation", "uncertainty", "proposed_next_action"}


def make_input(row):
    """Never pass surrogate wording to the model: it could copy template language."""
    allowed = {"day", "arm", "scenario", "seed", "focus", "source_action",
               "sources", "evidence_count", "candidate_count"}
    sample = {k: row[k] for k in allowed if k in row}
    if not sample.get("sources") or not isinstance(sample["sources"], list):
        raise ValueError("Missing independently grounded source IDs")
    if not all(isinstance(x, str) and x.startswith("world_events#") for x in sample["sources"]):
        raise ValueError("Invalid source ID")
    return sample


def validate(row, answer):
    """Strict mechanical gate; semantic factual review is still required."""
    if not isinstance(answer, dict) or set(answer) != REQUIRED:
        return ["schema_mismatch"]
    problems = []
    if type(answer["speak"]) is not bool:
        problems.append("speak_not_boolean")
    for key in ("speech", "reflection", "interpretation", "uncertainty"):
        if not isinstance(answer[key], str):
            problems.append(key + "_not_string")
    if not isinstance(answer["source_ids"], list) or not all(
        isinstance(x, str) for x in answer["source_ids"]
    ):
        problems.append("invalid_source_ids")
    elif not set(answer["source_ids"]).issubset(set(row["sources"])):
        problems.append("unsupported_citation")
    if answer["focus"] is not None and answer["focus"] != row.get("focus"):
        problems.append("unsupported_focus")
    if answer["proposed_next_action"] is not None and not isinstance(answer["proposed_next_action"], str):
        problems.append("invalid_proposal")
    if answer["speak"] is False and answer["speech"] != "":
        problems.append("silence_with_speech")
    if answer["speak"] is True and (not answer["speech"].strip() or not answer["source_ids"]):
        problems.append("speech_without_grounding")
    if not answer["uncertainty"].strip():
        problems.append("missing_uncertainty")
    return problems


def call_llm(row, endpoint, model, api_key, timeout=45):
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only remote HTTPS endpoints are allowed")
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"} or parsed.hostname.endswith(".local"):
        raise ValueError("Local model endpoints are forbidden")
    payload = json.dumps({"model": model, "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(make_input(row), ensure_ascii=False)}
        ]}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint, payload, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + api_key})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return json.loads(result["choices"][0]["message"]["content"])


def evaluate(rows, answers):
    if len(rows) != len(answers):
        raise ValueError("Row/answer count mismatch")
    output = []
    for row, answer in zip(rows, answers):
        errors = validate(make_input(row), answer)
        output.append({"seed": row.get("seed"), "day": row.get("day"),
            "arm": row.get("arm"), "scenario": row.get("scenario"),
            "input_source_ids": row["sources"], "answer": answer,
            "mechanical_errors": errors, "needs_human_semantic_review": True})
    return {"schema": "fourth_llm_speech_audit_v1",
            "evaluation_is_not_proof_of_autonomy": True,
            "production_N_modified": False,
            "episodes": output, "mechanically_valid": sum(
                not x["mechanical_errors"] for x in output),
            "total": len(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--answers", type=Path, help="Offline JSON list of prior LLM outputs")
    parser.add_argument("--generate", action="store_true", help="Opt in to remote LLM calls")
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input must be an array")
    if args.generate == bool(args.answers):
        parser.error("Specify exactly one of --answers or --generate")
    if args.generate:
        endpoint = os.environ["FOURTH_LLM_CHAT_URL"]
        model = os.environ["FOURTH_LLM_MODEL"]
        key = os.environ["FOURTH_LLM_API_KEY"]
        answers = [call_llm(row, endpoint, model, key) for row in rows]
    else:
        answers = json.loads(args.answers.read_text(encoding="utf-8"))
    report = evaluate(rows, answers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total": report["total"], "mechanically_valid": report["mechanically_valid"],
                      "semantic_review_required": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
