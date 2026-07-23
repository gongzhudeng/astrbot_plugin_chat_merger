from __future__ import annotations

import re
from typing import Any

IMAGE_CONTEXT_START = "<!-- astrbot-chat-merger:image-context:v1:start -->"
IMAGE_CONTEXT_RETAINED_START = (
    "<!-- astrbot-chat-merger:image-context:v1:retained:start -->"
)
IMAGE_CONTEXT_END = "<!-- astrbot-chat-merger:image-context:v1:end -->"
IMAGE_CONTEXT_PRUNED = "[历史图片：详细解析内容已按上下文保留限制清理]"
IMAGE_CONTEXT_PATTERN = re.compile(
    re.escape(IMAGE_CONTEXT_START) + r".*?" + re.escape(IMAGE_CONTEXT_END),
    flags=re.DOTALL,
)


def wrap_image_context(details: str, *, retained: bool = False) -> str:
    start = IMAGE_CONTEXT_RETAINED_START if retained else IMAGE_CONTEXT_START
    return "\n".join(
        (
            start,
            details.strip(),
            IMAGE_CONTEXT_END,
        )
    )


def count_image_contexts(text: Any) -> int:
    return len(IMAGE_CONTEXT_PATTERN.findall(str(text or "")))


def prune_image_contexts(
    contexts: list[Any],
    *,
    max_details: int,
    incoming_details: int = 0,
) -> int:
    if max_details <= 0:
        return 0

    keep_history = max(0, max_details - max(0, incoming_details))
    locations: list[tuple[Any, int | None, int, int]] = []
    for message in contexts:
        role, content = _message_role_and_content(message)
        if role != "user":
            continue
        if isinstance(content, list):
            for index, part in enumerate(content):
                text = _part_text(part)
                locations.extend(
                    (part, index, match.start(), match.end())
                    for match in IMAGE_CONTEXT_PATTERN.finditer(text)
                )
        else:
            text = str(content or "")
            locations.extend(
                (message, None, match.start(), match.end())
                for match in IMAGE_CONTEXT_PATTERN.finditer(text)
            )

    prune_count = max(0, len(locations) - keep_history)
    selected = locations[:prune_count]
    grouped: dict[
        tuple[int, int | None], tuple[Any, int | None, list[tuple[int, int]]]
    ] = {}
    for target, part_index, start, end in selected:
        key = (id(target), part_index)
        if key not in grouped:
            grouped[key] = (target, part_index, [])
        grouped[key][2].append((start, end))

    for target, part_index, spans in grouped.values():
        text = (
            _part_text(target) if part_index is not None else _message_content(target)
        )
        for start, end in sorted(spans, reverse=True):
            text = text[:start] + IMAGE_CONTEXT_PRUNED + text[end:]
        if part_index is None:
            _set_message_content(target, text)
        else:
            _set_part_text(target, text)
    return prune_count


def _message_role_and_content(message: Any) -> tuple[str, Any]:
    if isinstance(message, dict):
        return str(message.get("role", "")), message.get("content")
    return str(getattr(message, "role", "")), getattr(message, "content", None)


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _part_text(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("text", "") or "")
    return str(getattr(part, "text", "") or "")


def _set_message_content(message: Any, content: str) -> None:
    if isinstance(message, dict):
        message["content"] = content
    else:
        message.content = content


def _set_part_text(part: Any, text: str) -> None:
    if isinstance(part, dict):
        part["text"] = text
    else:
        part.text = text
