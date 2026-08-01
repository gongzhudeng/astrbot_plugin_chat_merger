from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from astrbot.api import logger

DEFAULT_IMAGE_CAPTION_PROMPT = (
    "请结合图片在消息中的出现顺序和相邻文字，分别客观转述每张图片。"
    "识别人物、动作、场景、可读文字和与用户问题有关的信息，不要编造。"
    "输出时严格遵守后面的 JSON 格式要求，不要输出额外解释。"
)

OUTPUT_PROTOCOL = (
    "只输出一个合法 JSON 对象，不要使用 Markdown 代码块或输出额外文字。"
    "键必须完全使用给出的图片编号，且每个编号都必须出现一次。"
    '例如：{"图1": "...", "图2": "..."}。不要遗漏、改写或合并编号。'
    "描述中的英文双引号必须使用反斜杠转义。"
)


def is_refusal_text(text: str, keywords: list[str]) -> bool:
    normalized = " ".join(text.lower().split())
    return bool(normalized) and any(
        keyword.strip().lower() in normalized for keyword in keywords if keyword.strip()
    )


def parse_caption_map(text: str, image_ids: list[str]) -> dict[str, str]:
    clean = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean, flags=re.DOTALL)
    if fenced:
        clean = fenced.group(1).strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return {
            image_id: str(payload.get(image_id, "") or "").strip()
            for image_id in image_ids
        }

    result: dict[str, str] = {}
    for image_id in image_ids:
        pattern = re.compile(
            rf"(?:<image_context\s+id=[\"']{re.escape(image_id)}[\"']>|"
            rf"[\"']?{re.escape(image_id)}[\"']?\s*[:：])\s*(.*?)"
            rf"(?:</image_context>|(?=\s*,?\s*[\"']?(?:"
            rf"{'|'.join(re.escape(value) for value in image_ids)}"
            rf")[\"']?\s*[:：])|\Z)",
            flags=re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(clean)
        if match:
            value = match.group(1)
            if "<image_context" not in match.group(0).lower():
                value = _clean_lenient_json_value(value)
            else:
                value = " ".join(value.split()).strip()
            if value:
                result[image_id] = value
    return result


def _clean_lenient_json_value(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    if value.endswith("}"):
        value = value[:-1].rstrip()
    if value[:1] in {'"', "'"}:
        value = value[1:]
    if value[-1:] in {'"', "'"}:
        value = value[:-1]
    return value.strip()


async def caption_ordered_images(
    ordered_parts: list[dict[str, Any]],
    *,
    providers: list[Any],
    prompt: str,
    refusal_keywords: list[str],
    timeout_seconds: float,
    max_images: int,
) -> dict[str, str]:
    image_parts = [part for part in ordered_parts if part.get("kind") == "image"]
    selected = image_parts[: max(1, max_images)]
    image_ids = [str(part["id"]) for part in selected]
    if not image_ids:
        return {}

    selected_ids = set(image_ids)
    prepared_image_ids: list[str] = []
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{prompt.strip() or DEFAULT_IMAGE_CAPTION_PROMPT}\n\n{OUTPUT_PROTOCOL}",
        }
    ]
    for part in ordered_parts:
        kind = part.get("kind")
        if kind == "text" and str(part.get("text", "")).strip():
            content.append(
                {
                    "type": "text",
                    "text": f"用户文字：{str(part['text']).strip()}",
                }
            )
        elif kind == "image" and str(part.get("id")) in selected_ids:
            image_id = str(part["id"])
            try:
                image_path = await part["component"].convert_to_file_path()
                image_url = _path_to_data_url(Path(image_path))
            except Exception as exc:
                logger.warning(
                    "[消息合并] 图片 %s 准备失败，将使用失败占位: %s",
                    image_id,
                    exc,
                )
                continue
            prepared_image_ids.append(image_id)
            content.extend(
                [
                    {"type": "text", "text": f"图片编号：{image_id}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "id": image_id,
                        },
                    },
                ]
            )

    if not prepared_image_ids:
        return {}

    for provider in providers:
        provider_name = _provider_name(provider)
        try:
            response = await asyncio.wait_for(
                provider.text_chat(contexts=[{"role": "user", "content": content}]),
                timeout=max(1.0, timeout_seconds),
            )
            text = str(getattr(response, "completion_text", "") or "").strip()
            if not text:
                logger.warning("[消息合并] 图片转述模型 %s 返回空结果", provider_name)
                continue
            if is_refusal_text(text, refusal_keywords):
                logger.warning("[消息合并] 图片转述模型 %s 返回拒绝内容", provider_name)
                continue
            parsed = parse_caption_map(text, prepared_image_ids)
            if all(parsed.get(image_id) for image_id in prepared_image_ids):
                return parsed
            missing_ids = [
                image_id for image_id in prepared_image_ids if not parsed.get(image_id)
            ]
            logger.warning(
                "[消息合并] 图片转述模型 %s 未返回完整编号映射，缺少: %s，尝试下一个",
                provider_name,
                ", ".join(missing_ids) or "未知",
            )
        except Exception as exc:
            logger.warning(
                "[消息合并] 图片转述模型 %s 调用失败，尝试下一个: %s",
                provider_name,
                exc,
            )
    return {}


def _path_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _provider_name(provider: Any) -> str:
    try:
        return str(provider.meta().id)
    except Exception:
        return type(provider).__name__
