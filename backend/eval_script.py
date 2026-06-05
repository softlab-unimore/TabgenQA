#!/usr/bin/env python3
"""
Worker script for evaluating LLM models on generated TabQA instances.
Supports OpenAI, Claude, Gemini (OpenAI-compat endpoint), and HuggingFace via vLLM.
Outputs JSON lines to stdout for the parent process to consume.
"""

import sys
import json
import os
import re
import string
import traceback
import time
from collections import Counter


def emit(msg: dict):
    print(json.dumps(msg), flush=True)


INFERENCE_PROMPT = (
    "Answer the following question given the provided HTML table(s).\n"
    'First reason step-by-step, then write "Final answer:" followed exclusively by the correct answer. '
    'Do not write anything else after "Final answer:"\n'
    "Every calculation must be done with a precision of exactly 6 decimal places.\n"
    "Only the numerical value must be written in the final answer.\n\n"
    "Question: {question}\nTable:\n{table}\n\nLet's think step-by-step. "
)


def _extract_final_answer(text: str) -> str:
    pos = text.lower().rfind("final answer:")
    if pos == -1:
        return text.strip()
    return text[pos + len("final answer:"):].strip()


def _remove_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", lambda m: re.sub(r"^```.*\n|```$", "", m.group()), text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return text.replace("python", "").strip()


def _extract_number(text: str):
    m = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", str(text))
    return float(m.group(0).replace(",", "")) if m else None


def _normalize(s: str) -> str:
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation or ch == ".")
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p_tok = _normalize(pred).split()
    g_tok = _normalize(gold).split()
    if not p_tok and not g_tok:
        return 1.0
    if not p_tok or not g_tok:
        return 0.0
    common = Counter(p_tok) & Counter(g_tok)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_tok)
    rec = num_same / len(g_tok)
    return 2 * prec * rec / (prec + rec)


def _score(prediction: str, label: str):
    pn = _extract_number(prediction)
    ln = _extract_number(label)
    if pn is not None and ln is not None:
        return abs(pn - ln) < 0.005, _f1(prediction, label)
    return _normalize(prediction) == _normalize(label), _f1(prediction, label)


def _call_openai_compat(client, model_name: str, prompt: str) -> str:
    kwargs = dict(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=2048,
    )
    for _ in range(4):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as exc:
            msg = str(exc).lower()
            if "max_tokens" in msg and ("unsupported" in msg or "not supported" in msg):
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = 2048
            elif "temperature" in msg and ("unsupported" in msg or "not supported" in msg):
                kwargs["temperature"] = 1
            else:
                raise
    raise RuntimeError("Failed to call API after parameter adjustments")


def _call_claude(client, model_name: str, prompt: str) -> str:
    msg = client.messages.create(
        model=model_name,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def main():
    if len(sys.argv) < 2:
        emit({"type": "error", "message": "No parameters provided"})
        sys.exit(1)

    try:
        params = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        emit({"type": "error", "message": f"Invalid JSON: {exc}"})
        sys.exit(1)

    instances_file = params["instances_file"]
    model_type = params.get("model_type", "openai")
    model_name = params.get("model_name", "gpt-4o-mini")
    api_key = params.get("api_key", "")
    base_url = params.get("base_url", "")
    output_dir = params["output_dir"]

    try:
        with open(instances_file, encoding="utf-8") as f:
            instances = json.load(f)
    except Exception as exc:
        emit({"type": "error", "message": f"Failed to load instances: {exc}"})
        sys.exit(1)

    if not instances:
        emit({"type": "error", "message": "No instances to evaluate"})
        sys.exit(1)

    total = len(instances)
    emit({"type": "status", "message": f"Loaded {total} instances. Initialising {model_type} model…"})

    client = None
    is_claude = (model_type == "claude")

    try:
        if model_type in ("openai", "huggingface"):
            from openai import OpenAI as _OAI
            kw: dict = {"api_key": api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")}
            if model_type == "huggingface":
                kw["base_url"] = base_url or os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1")
            client = _OAI(**kw)
        elif model_type == "gemini":
            # Use Google's OpenAI-compatible endpoint — no extra SDK needed
            from openai import OpenAI as _OAI
            client = _OAI(
                api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        elif model_type == "claude":
            import anthropic as _anth
            client = _anth.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
        else:
            emit({"type": "error", "message": f"Unknown model_type: {model_type}"})
            sys.exit(1)
    except Exception as exc:
        emit({"type": "error", "message": f"Failed to init model: {exc}\n{traceback.format_exc()}"})
        sys.exit(1)

    emit({"type": "status", "message": "Starting inference…"})

    predictions = []
    for i, inst in enumerate(instances):
        question = inst.get("question", "")
        tables = inst.get("tables", [])
        label = str(inst.get("answer", ""))
        table_str = "\n".join(tables) if isinstance(tables, list) else str(tables)
        prompt = INFERENCE_PROMPT.format(question=question, table=table_str)

        raw = ""
        for attempt in range(3):
            try:
                if is_claude:
                    raw = _call_claude(client, model_name, prompt)
                else:
                    raw = _call_openai_compat(client, model_name, prompt)
                break
            except Exception as exc:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    raw = f"ERROR: {exc}"

        prediction = _extract_final_answer(_remove_markdown(raw)).replace("%", "").strip()
        match, f1 = _score(prediction, label)

        pn = _extract_number(prediction)
        ln = _extract_number(label)
        if pn is not None and ln is not None:
            label_display = f"{round(ln, 6):g}"
            prediction_display = f"{round(pn, 6):g}"
        else:
            label_display = _normalize(label)
            prediction_display = _normalize(prediction)

        predictions.append({
            "id": inst.get("id", i),
            "question": question,
            "label": label,
            "label_display": label_display,
            "prediction": prediction,
            "prediction_display": prediction_display,
            "match": bool(match),
            "f1": float(f1),
            "reasoning": raw,
            "tables": tables if isinstance(tables, list) else ([str(tables)] if tables else []),
        })

        emit({
            "type": "progress",
            "current": i + 1,
            "total": total,
            "progress": (i + 1) / total,
            "desc": f"Evaluated {i + 1}/{total}",
        })

    accuracy = sum(p["match"] for p in predictions) / len(predictions) if predictions else 0.0
    avg_f1 = sum(p["f1"] for p in predictions) / len(predictions) if predictions else 0.0

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    emit({
        "type": "result",
        "accuracy": float(accuracy),
        "avg_f1": float(avg_f1),
        "total": total,
        "correct": int(sum(p["match"] for p in predictions)),
    })


if __name__ == "__main__":
    main()
