#!/usr/bin/env python3
"""Run VLM semantic filtering on an image manifest."""

from __future__ import annotations

import argparse
import base64
import os
import re
from pathlib import Path

import pandas as pd


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_api_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(f"Missing API key. Set the {env_var} environment variable.")
    return key


def parse_decision(text: str) -> tuple[str, str]:
    """Read the last standalone decision line, ignoring preceding reasoning.

    Returns ``(decision, rule)`` where ``decision`` is ``"keep"`` or
    ``"remove"`` and ``rule`` is ``"none"`` or ``"rule1"`` .. ``"rule5"``.
    A bare ``REMOVE`` from the rule-free prompt has an empty rule. Responses
    without a valid decision line raise ``ValueError``.
    """
    for line in reversed(str(text).splitlines()):
        # The verbatim prompts use arrows and code formatting for decisions.
        line = re.sub(r"[*`]", "", line).strip().lstrip(">#-+→⇒ ").strip()
        match = re.fullmatch(
            r"(?:(?:final\s+)?decision\s*:\s*)?"
            r"(?:(ACCEPTED)|REMOVE(?:\s*,\s*Rule\s*([1-5]))?)\.?",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            if match.group(1):
                return "keep", "none"
            return "remove", (f"rule{match.group(2)}" if match.group(2) else "")
    raise ValueError("VLM response has no valid final decision line.")


def image_prompt_parts(prompt: str, image_path: str, class_name: str, provider: str) -> list[str]:
    """Build the per-image text blocks in the order used in the article."""
    label = class_name.strip()
    if not label:
        raise ValueError("Every image needs a nonempty class name.")
    if "leaf" not in label.lower():
        label += " leaf"
    label_prefix = "Label" if provider == "gemini" else "The label is"
    parts = [prompt, "\n# Now evaluate the following image:", f"{label_prefix}: '{label}'"]
    if provider != "gemini":
        parts.append(f"Photo: {Path(image_path).name}")
    return parts


def call_openai(model: str, prompt: str, image_path: str, class_name: str, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
    encoded = encode_image(image_path)
    content = [{"type": "text", "text": part}
               for part in image_prompt_parts(prompt, image_path, class_name, "openai")]
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def call_gemini(model: str, prompt: str, image_path: str, class_name: str, temperature: float) -> str:
    from google import genai

    client = genai.Client(api_key=get_api_key("GOOGLE_API_KEY"))
    image_bytes = Path(image_path).read_bytes()
    image = genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    response = client.models.generate_content(
        model=model,
        contents=[*image_prompt_parts(prompt, image_path, class_name, "gemini"), image],
        config=genai.types.GenerateContentConfig(temperature=temperature),
    )
    return response.text or ""


def call_anthropic(model: str, prompt: str, image_path: str, class_name: str, temperature: float) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))
    encoded = encode_image(image_path)
    content = [{"type": "text", "text": part}
               for part in image_prompt_parts(prompt, image_path, class_name, "anthropic")]
    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded}})
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )
    return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--prompt_txt", required=True)
    parser.add_argument("--provider", choices=["openai", "gemini", "anthropic"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image_col", default="img_path")
    parser.add_argument("--class_col", default="class_name")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    missing = {args.image_col, args.class_col} - set(df.columns)
    if missing:
        parser.error(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")
    if df[args.class_col].isna().any() or df[args.class_col].astype(str).str.strip().eq("").any():
        parser.error("Every image needs a nonempty class name.")
    prompt = Path(args.prompt_txt).read_text(encoding="utf-8")

    rows = []
    for _, row in df.iterrows():
        image_path = row[args.image_col]
        class_name = str(row[args.class_col])
        if args.provider == "openai":
            raw = call_openai(args.model, prompt, image_path, class_name, args.temperature)
        elif args.provider == "gemini":
            raw = call_gemini(args.model, prompt, image_path, class_name, args.temperature)
        else:
            raw = call_anthropic(args.model, prompt, image_path, class_name, args.temperature)

        decision, rule = parse_decision(raw)
        out = row.to_dict()
        out["vlm_decision"] = decision
        out["vlm_rule"] = rule
        rows.append(out)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
