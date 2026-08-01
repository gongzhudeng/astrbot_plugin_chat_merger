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
)

OUTPUT_PROTOCOL = (
    "请保留给出的图片编号，分别说明每张图片。"
    '推荐输出为 {"图1": "...", "图2": "..."}，也可以使用“图1：……”等明确的编号格式。'
    "不要遗漏、改写或合并编号；只有一张图片时可以直接输出描述。"
)

_CHINESE_NUMERALS = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")


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
            if str(payload.get(image_id, "") or "").strip()
        }

    result = _parse_numbered_captions(clean, image_ids)
    if result or len(image_ids) != 1:
        return result
    plain = _clean_lenient_value(clean)
    return {image_ids[0]: plain} if plain else {}


def _parse_numbered_captions(text: str, image_ids: list[str]) -> dict[str, str]:
    aliases = {image_id: _image_id_aliases(image_id) for image_id in image_ids}
    alias_to_id = {
        alias.lower(): image_id
        for image_id, image_aliases in aliases.items()
        for alias in image_aliases
    }
    marker_pattern = re.compile(
        rf"(?:<image_context\s+id=[\"'](?P<xml>{'|'.join(map(re.escape, image_ids))})[\"']>|"
        rf"[\"']?(?P<label>{'|'.join(map(re.escape, alias_to_id))})[\"']?"
        rf"\s*(?:[:：]|(?:是|为|内容是|内容为)\s*))",
        flags=re.IGNORECASE,
    )
    matches = list(marker_pattern.finditer(text))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        image_id = match.group("xml") or alias_to_id.get(
            str(match.group("label") or "").lower()
        )
        if not image_id or image_id in result:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end]
        if match.group("xml"):
            value = value.split("</image_context>", 1)[0]
        value = _clean_lenient_value(value)
        if value:
            result[image_id] = value
    return result


def _image_id_aliases(image_id: str) -> tuple[str, ...]:
    matched = re.search(r"(\d+)$", image_id)
    index = int(matched.group(1)) if matched else 0
    chinese = (
        _CHINESE_NUMERALS[index] if 0 < index < len(_CHINESE_NUMERALS) else str(index)
    )
    return (
        image_id,
        f"图片{index}",
        f"图{chinese}",
        f"图片{chinese}",
        f"第{index}张",
        f"第{chinese}张",
    )


def _clean_lenient_value(value: str) -> str:
    value = value.strip().rstrip(",").strip()
    if value.endswith("}"):
        value = value[:-1].rstrip()
    if value[:1] in {'"', "'"}:
        value = value[1:]
    if value[-1:] in {'"', "'"}:
        value = value[:-1]
    return " ".join(value.split()).strip()


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

    captions: dict[str, str] = {}
    pending_ids = list(prepared_image_ids)
    for provider in providers:
        provider_name = _provider_name(provider)
        provider_content = _content_for_image_ids(content, pending_ids)
        try:
            response = await asyncio.wait_for(
                provider.text_chat(
                    contexts=[{"role": "user", "content": provider_content}]
                ),
                timeout=max(1.0, timeout_seconds),
            )
            text = str(getattr(response, "completion_text", "") or "").strip()
            if not text:
                logger.warning("[消息合并] 图片转述模型 %s 返回空结果", provider_name)
                continue
            if is_refusal_text(text, refusal_keywords):
                logger.warning("[消息合并] 图片转述模型 %s 返回拒绝内容", provider_name)
                continue
            parsed = parse_caption_map(text, pending_ids)
            captions.update(parsed)
            pending_ids = [
                image_id for image_id in pending_ids if not captions.get(image_id)
            ]
            if not pending_ids:
                return captions
            logger.warning(
                "[消息合并] 图片转述模型 %s 缺少图片描述: %s，尝试下一个",
                provider_name,
                ", ".join(pending_ids),
            )
        except Exception as exc:
            logger.warning(
                "[消息合并] 图片转述模型 %s 调用失败，尝试下一个: %s",
                provider_name,
                exc,
            )
    return captions


def _content_for_image_ids(
    content: list[dict[str, Any]], image_ids: list[str]
) -> list[dict[str, Any]]:
    selected = set(image_ids)
    result: list[dict[str, Any]] = []
    for item in content:
        if item.get("type") == "image_url":
            if str(item.get("image_url", {}).get("id", "")) in selected:
                result.append(item)
            continue
        text = str(item.get("text", ""))
        if (
            text.startswith("图片编号：")
            and text.removeprefix("图片编号：") not in selected
        ):
            continue
        result.append(item)
    return result


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
