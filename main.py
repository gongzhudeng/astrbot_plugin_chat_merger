import asyncio
import html
import importlib
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import (
    ComponentType,
    Image,
    Plain,
    Record,
    Reply,
    Video,
)
from astrbot.core.provider.provider import Provider, STTProvider
from astrbot.core.utils.quoted_message_parser import (
    extract_quoted_message_images,
    extract_quoted_message_text,
)

from .image_caption import DEFAULT_IMAGE_CAPTION_PROMPT, caption_ordered_images
from .image_context import (
    count_image_contexts,
    prune_image_contexts,
    wrap_image_context,
)

try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
        AiocqhttpMessageEvent,
    )

    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False


def count_words(text: str) -> int:
    """Count Chinese characters and English words."""
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    english = len(re.findall(r"[a-zA-Z]+", text))
    return chinese + english


MERGED_FLAG_KEY = "chat_merger_merged"


@register(
    "astrbot_plugin_chat_merger",
    "灵犀 · 消息合并助手",
    '彻底告别一问一答式AI聊天。自动合并连续消息、智能延迟后统一回复，AI思考时显示"对方正在输入…"。支持关键词触发超长等待、图片智能合并、等待时间随机波动、AI忙感知自动排队、LLM智能延迟判断、输入状态感知、撤回消息过滤，让AI对话真正拥有真人聊天的节奏感',
    "2.8.2",
    "https://github.com/gongzhudeng/astrbot_plugin_chat_merger",
)
class ChatMergerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.message_queues: dict[str, list[dict]] = defaultdict(list)
        self.timers: dict[str, asyncio.Task] = {}
        self._event_refs: dict[str, AstrMessageEvent] = {}
        self.infinite_wait: dict[str, bool] = defaultdict(bool)
        self.wait_start_time: dict[str, float] = {}
        self._ai_busy: dict[str, bool] = {}
        self._ai_busy_wait_tasks: dict[str, asyncio.Task] = {}
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._typing_stop_events: dict[str, asyncio.Event] = {}
        # typing detection state
        self._is_typing: dict[str, bool] = {}
        self._timer_end_time: dict[str, float] = {}
        self._calc_delay: dict[str, float] = {}
        self._debug("插件已初始化")

    # ── Utility ──────────────────────────────────────────────

    def _debug(self, msg: str) -> None:
        if self._get_config("debug_mode", False):
            logger.info(f"[消息合并] {msg}")

    def _log(self, msg: str) -> None:
        logger.info(f"[消息合并] {msg}")

    def _get_config(self, key: str, default=None):
        """Read config value. Supports both flat keys and nested keys under UI groups."""
        # Flat key access
        if key in self.config:
            return self.config[key]
        # Nested: search inside "type": "object" groups
        for group_key in self.config:
            group = self.config.get(group_key)
            if isinstance(group, dict) and key in group:
                return group[key]
        return default

    @staticmethod
    def _get_original_text(event: AstrMessageEvent) -> str:
        """从消息链获取原始文本（含 / 前缀），不受 waking_check 剥离影响。"""
        parts = []
        for comp in event.message_obj.message:
            if isinstance(comp, Plain):
                parts.append(comp.text)
        return "".join(parts).strip()

    @staticmethod
    def _raw_message_has_voice(event: AstrMessageEvent) -> bool:
        raw = getattr(event.message_obj, "raw_message", None)
        if raw is None:
            return False
        try:
            message = raw.get("message")
        except (AttributeError, TypeError):
            message = getattr(raw, "message", None)
        if not isinstance(message, list):
            return False
        return any(
            isinstance(segment, dict) and segment.get("type") == "record"
            for segment in message
        )

    @staticmethod
    def _get_record(event: AstrMessageEvent) -> Record | None:
        return next(
            (comp for comp in event.message_obj.message if isinstance(comp, Record)),
            None,
        )

    @staticmethod
    def _format_voice_message(text: str, *, failed: bool = False) -> str:
        source = "speech_to_text_failed" if failed else "speech_to_text"
        content = text.strip() or "[语音识别失败]"
        return (
            f'<voice_message speaker="user" source="{source}">\n'
            f"{html.escape(content)}\n"
            "</voice_message>"
        )

    def _find_config_value(self, key: str):
        if key in self.config:
            return True, self.config[key]
        for group_key in self.config:
            group = self.config.get(group_key)
            if isinstance(group, dict) and key in group:
                return True, group[key]
        return False, None

    def _configured_fallback_ids(
        self, list_key: str, legacy_keys: tuple[str, ...]
    ) -> list[str]:
        exists, configured = self._find_config_value(list_key)
        if exists:
            if isinstance(configured, list):
                return [str(item).strip() for item in configured if str(item).strip()]
            self._log(f"回退 Provider 配置不是列表: {list_key}")
            return []
        return [
            value
            for key in legacy_keys
            if (value := str(self._get_config(key, "") or "").strip())
        ]

    def _stt_providers(self) -> list[STTProvider]:
        provider_ids = [
            str(self._get_config("stt_provider_id", "") or "").strip(),
            *self._configured_fallback_ids(
                "stt_fallback_provider_ids",
                ("stt_fallback_provider_id", "stt_fallback_provider_id_2"),
            ),
        ]
        return self._resolve_providers(provider_ids, STTProvider, "STT")

    def _image_providers(self) -> list[Provider]:
        provider_ids = [
            str(self._get_config("image_caption_provider_id", "") or "").strip(),
            *self._configured_fallback_ids(
                "image_caption_fallback_provider_ids",
                (
                    "image_caption_fallback_provider_id",
                    "image_caption_fallback_provider_id_2",
                ),
            ),
        ]
        return self._resolve_providers(provider_ids, Provider, "图片转述")

    def _resolve_providers(self, provider_ids, provider_type, label):
        providers = []
        seen = set()
        for provider_id in provider_ids:
            if not provider_id:
                continue
            provider = self.context.get_provider_by_id(provider_id)
            if not isinstance(provider, provider_type):
                self._log(f"{label} Provider 不可用或类型不正确: {provider_id}")
                continue
            identity = id(provider)
            if identity in seen:
                continue
            seen.add(identity)
            providers.append(provider)
        return providers

    def _image_refusal_keywords(self) -> list[str]:
        configured = self._get_config(
            "image_caption_refusal_keywords",
            [
                "无法协助",
                "不能协助",
                "无法描述",
                "不能描述",
                "抱歉，我不能",
                "i can't help",
                "i cannot help",
                "unable to assist",
            ],
        )
        return [str(item) for item in configured if str(item).strip()]

    @staticmethod
    def _provider_name(provider) -> str:
        try:
            return str(provider.meta().id)
        except Exception:
            return type(provider).__name__

    async def _transcribe_record(self, record: Record) -> str:
        try:
            audio_path = await record.convert_to_file_path()
        except Exception as e:
            self._log(f"获取语音文件失败: {e}")
            return ""

        retries = max(1, int(self._get_config("stt_file_ready_retries", 5)))
        interval = max(0.0, float(self._get_config("stt_file_ready_interval", 0.5)))
        for provider in self._stt_providers():
            for attempt in range(retries):
                try:
                    text = str(
                        await provider.get_text(audio_url=audio_path) or ""
                    ).strip()
                    if text:
                        self._debug(
                            f"STT Provider {self._provider_name(provider)} 识别成功"
                        )
                        return text
                    self._log(
                        f"STT Provider {self._provider_name(provider)} 返回空结果，尝试下一个"
                    )
                    break
                except FileNotFoundError:
                    if attempt + 1 >= retries:
                        self._log(
                            f"语音文件未就绪，STT Provider {self._provider_name(provider)} 重试耗尽"
                        )
                        break
                    await asyncio.sleep(interval)
                except Exception as e:
                    self._log(
                        f"STT Provider {self._provider_name(provider)} 识别失败，尝试下一个: {e}"
                    )
                    break
        return ""

    async def _resolve_voice(
        self, event: AstrMessageEvent, text: str
    ) -> tuple[bool, str]:
        if not self._get_config("voice_message_enabled", True):
            return False, text

        record = self._get_record(event)
        raw_has_voice = self._raw_message_has_voice(event)
        if record is None and not raw_has_voice:
            return False, text

        if record is not None:
            transcript = await self._transcribe_record(record)
        else:
            transcript = text.strip()

        if transcript:
            return True, transcript
        return True, ""

    @staticmethod
    def _is_contains_mode(mode_str: str) -> bool:
        return mode_str in ("contains", "包含")

    @classmethod
    def _is_image_only(cls, event: AstrMessageEvent) -> bool:
        """Check if message contains only images (no text or voice)."""
        has_image = False
        has_text = False
        has_other_content = False
        for comp in event.message_obj.message:
            if cls._is_image_component(comp):
                has_image = True
            elif isinstance(comp, Plain) and comp.text.strip():
                has_text = True
            elif isinstance(comp, Record) or cls._is_video_component(comp):
                has_other_content = True
        return has_image and not has_text and not has_other_content

    @staticmethod
    def _component_type_is(component, expected: ComponentType) -> bool:
        value = getattr(component, "type", None)
        if value == expected:
            return True
        return (
            str(getattr(value, "value", value) or "").lower() == expected.value.lower()
        )

    @classmethod
    def _is_video_component(cls, component) -> bool:
        return isinstance(component, Video) or cls._component_type_is(
            component, ComponentType.Video
        )

    @classmethod
    def _is_image_component(cls, component) -> bool:
        return isinstance(component, Image) or cls._component_type_is(
            component, ComponentType.Image
        )

    @staticmethod
    def _raw_message_has_video(event: AstrMessageEvent) -> bool:
        raw = getattr(event.message_obj, "raw_message", None)
        if raw is None:
            return False
        try:
            message = raw.get("message")
        except (AttributeError, TypeError):
            message = getattr(raw, "message", None)
        return isinstance(message, list) and any(
            isinstance(segment, dict) and segment.get("type") == "video"
            for segment in message
        )

    @classmethod
    def _has_direct_video(cls, event: AstrMessageEvent) -> bool:
        return any(cls._is_video_component(comp) for comp in event.message_obj.message)

    @classmethod
    def _reply_contains_video(cls, reply: Reply, quoted_text: str) -> bool:
        if any(cls._is_video_component(comp) for comp in list(reply.chain or [])):
            return True
        candidates = (quoted_text, str(reply.message_str or ""))
        return any(
            marker in candidate.lower()
            for candidate in candidates
            for marker in ("[video]", "[视频]", "[引用视频消息]")
        )

    @classmethod
    def _ordered_message_parts(
        cls,
        event: AstrMessageEvent,
        text: str,
        *,
        source_type: str,
        quote_contexts: list[str],
        image_only: bool,
        video_only: bool,
    ) -> list[dict]:
        parts = [{"kind": "text", "text": value} for value in quote_contexts]
        if source_type == "voice":
            if text:
                parts.append({"kind": "text", "text": text})
            return parts

        saw_text = False
        for component in event.message_obj.message:
            if isinstance(component, (Reply, Record)):
                continue
            if isinstance(component, Plain):
                value = component.text.strip()
                if value:
                    parts.append({"kind": "text", "text": value})
                    saw_text = True
            elif cls._is_image_component(component):
                parts.append({"kind": "image", "component": component})
            elif cls._is_video_component(component):
                parts.append({"kind": "video", "component": component})
            else:
                parts.append({"kind": "component", "component": component})

        if text and not saw_text:
            parts.append({"kind": "text", "text": text})
        elif image_only and not any(part["kind"] == "image" for part in parts):
            parts.append({"kind": "text", "text": "[图片]"})
        elif video_only and not any(part["kind"] == "video" for part in parts):
            parts.append({"kind": "text", "text": "[视频]"})
        return parts

    @staticmethod
    def _number_media_parts(parts: list[dict]) -> None:
        image_index = 0
        video_index = 0
        for part in parts:
            if part["kind"] == "image":
                image_index += 1
                part["id"] = f"图{image_index}"
            elif part["kind"] == "video":
                video_index += 1
                part["id"] = f"视频{video_index}"

    @staticmethod
    async def _snapshot_image_component(component) -> Image:
        """Detach an image from the source event's temporary-file lifecycle."""
        convert_to_base64 = getattr(component, "convert_to_base64", None)
        if callable(convert_to_base64):
            encoded = str(await convert_to_base64() or "").strip()
            if encoded:
                return Image.fromBase64(encoded)

        image_path = await component.convert_to_file_path()
        image_bytes = await asyncio.to_thread(Path(image_path).read_bytes)
        if not image_bytes:
            raise ValueError("resolved image is empty")
        return Image.fromBytes(image_bytes)

    async def _snapshot_image_parts(self, parts: list[dict]) -> None:
        for part in parts:
            component = part.get("component")
            if component is None or not self._is_image_component(component):
                continue
            try:
                part["component"] = await self._snapshot_image_component(component)
            except Exception as exc:
                self._log(f"图片入队快照失败，后续将使用失败占位: {exc}")

    @staticmethod
    def _emoji_library_service():
        try:
            module = importlib.import_module(
                "data.plugins.astrbot_plugin_emoji_library.service"
            )
            return module.get_service()
        except (ImportError, AttributeError):
            return None

    async def _caption_images(self, parts: list[dict]) -> dict[str, str]:
        image_parts = [part for part in parts if part["kind"] == "image"]
        if not image_parts:
            return {}
        has_cached_match = any(
            str((part.get("emoji_library_match") or {}).get("explanation", "")).strip()
            for part in image_parts
        )
        if (
            not self._get_config("image_caption_enabled", False)
            and not has_cached_match
        ):
            return {}

        captions: dict[str, str] = {}
        unresolved: list[dict] = []
        service = self._emoji_library_service()
        for part in image_parts:
            match = part.get("emoji_library_match")
            if not part.get("emoji_library_checked") and service is not None:
                try:
                    match = await service.resolve_component(part["component"])
                except Exception as e:
                    self._debug(f"表情包解释库查询失败，回退图片转述: {e}")
            explanation = str((match or {}).get("explanation", "")).strip()
            if explanation:
                image_id = str(part["id"])
                captions[image_id] = explanation
                part["preserve_image_context"] = bool(
                    (match or {}).get("preserve_context", False)
                )
                self._log(f"{image_id} 命中表情包解释库，跳过图片模型")
            else:
                unresolved.append(part)

        if not unresolved or not self._get_config("image_caption_enabled", False):
            return captions
        providers = self._image_providers()
        if not providers:
            self._log("未配置可用的图片转述模型，图片将以失败占位交给主模型")
            return captions

        unresolved_ids = {id(part) for part in unresolved}
        caption_parts = [
            part
            for part in parts
            if part["kind"] != "image" or id(part) in unresolved_ids
        ]
        generated = await caption_ordered_images(
            caption_parts,
            providers=providers,
            prompt=str(
                self._get_config("image_caption_prompt", DEFAULT_IMAGE_CAPTION_PROMPT)
                or DEFAULT_IMAGE_CAPTION_PROMPT
            ),
            refusal_keywords=self._image_refusal_keywords(),
            timeout_seconds=float(
                self._get_config("image_caption_timeout_seconds", 60) or 60
            ),
            max_images=max(
                1, int(self._get_config("image_caption_max_images", 9) or 9)
            ),
            compress_enabled=bool(
                self._get_config("image_caption_compress_enabled", False)
            ),
            compress_max_size=max(
                1,
                int(self._get_config("image_caption_compress_max_size", 1280) or 1280),
            ),
            compress_quality=min(
                100,
                max(
                    50,
                    int(self._get_config("image_caption_compress_quality", 85) or 85),
                ),
            ),
        )
        captions.update(generated)

        if service is not None:
            for part in unresolved:
                explanation = generated.get(str(part["id"]), "").strip()
                if not explanation:
                    continue
                try:
                    await service.record_component_analysis(
                        part["component"], explanation
                    )
                except Exception as e:
                    self._debug(f"表情包解释候选保存失败: {e}")
        return captions

    @staticmethod
    def _render_parts(
        parts: list[dict],
        image_captions: dict[str, str],
        *,
        preserve_images: bool = False,
    ) -> tuple[str, list]:
        text_lines: list[str] = []
        replay_components: list = []
        for part in parts:
            kind = part["kind"]
            if kind == "text":
                value = str(part.get("text", "")).strip()
                if value:
                    text_lines.append(value)
                    replay_components.append(Plain(value))
            elif kind == "image":
                image_id = str(part["id"])
                description = image_captions.get(image_id, "").strip()
                if preserve_images and not description:
                    placeholder = f"[{image_id}]"
                    text_lines.append(placeholder)
                    replay_components.extend([Plain(placeholder), part["component"]])
                    continue
                if description:
                    value = wrap_image_context(
                        f'<image_context id="{image_id}">{html.escape(description)}'
                        "</image_context>",
                        retained=bool(part.get("preserve_image_context", False)),
                    )
                else:
                    value = (
                        f'<image_context id="{image_id}" status="failed">'
                        "图片转述失败</image_context>"
                    )
                text_lines.append(value)
                replay_components.append(Plain(value))
            elif kind == "video":
                video_id = str(part["id"])
                placeholder = f"[{video_id}]"
                text_lines.append(placeholder)
                replay_components.extend([Plain(placeholder), part["component"]])
            else:
                replay_components.append(part["component"])
        return "\n".join(text_lines), replay_components

    @staticmethod
    def _format_quoted_message(
        event: AstrMessageEvent,
        reply: Reply,
        text: str,
        image_count: int,
        *,
        has_video: bool = False,
    ) -> str:
        sender = reply.sender_nickname or (
            f"user_id:{reply.sender_id}" if reply.sender_id else "未知发送者"
        )
        role = (
            "assistant"
            if reply.sender_id and str(reply.sender_id) == str(event.get_self_id())
            else "user"
        )
        content_parts = []
        if text and text != "[Empty Text]":
            normalized_text = text.replace("[Video]", "").replace("[视频]", "").strip()
            if normalized_text:
                content_parts.append(normalized_text)
        if has_video:
            content_parts.append("[引用视频消息]")
        if image_count:
            content_parts.append(f"[引用图片: {image_count}张]")
        if not content_parts:
            content_parts.append("[空引用消息]")
        content = "\n".join(content_parts)
        return (
            f'<quoted_message sender="{html.escape(str(sender), quote=True)}" '
            f'role="{role}">'
            f"{html.escape(content)}"
            "</quoted_message>"
        )

    async def _build_queue_item(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        source_type: str = "text",
        delay_text: str | None = None,
        wait_trigger: bool = False,
        image_only: bool = False,
        video_only: bool = False,
    ) -> dict:
        quote_contexts = []
        quote_images = []
        for comp in event.message_obj.message:
            if not isinstance(comp, Reply):
                continue
            try:
                quoted_text = await extract_quoted_message_text(event, comp) or ""
                quoted_images = await extract_quoted_message_images(event, comp)
            except Exception as e:
                self._log(f"引用消息解析失败 | reply_id={comp.id} | {e}")
                quoted_text = comp.message_str or ""
                quoted_images = []
            has_video = self._reply_contains_video(comp, quoted_text)
            quote_contexts.append(
                self._format_quoted_message(
                    event,
                    comp,
                    quoted_text,
                    len(quoted_images),
                    has_video=has_video,
                )
            )
            quote_images.extend(Image(file=image_ref) for image_ref in quoted_images)

        parts = [
            {"kind": "component", "component": image} for image in quote_images
        ] + self._ordered_message_parts(
            event,
            text,
            source_type=source_type,
            quote_contexts=quote_contexts,
            image_only=image_only,
            video_only=video_only,
        )
        await self._snapshot_image_parts(parts)
        preview_parts = []
        for part in parts:
            if part["kind"] == "text":
                preview_parts.append(str(part.get("text", "")))
            elif part["kind"] == "image":
                preview_parts.append("[图片]")
            elif part["kind"] == "video":
                preview_parts.append("[视频]")
        preview_text = "\n".join(value for value in preview_parts if value).strip()
        return {
            "message_id": self._get_message_id(event),
            "text": preview_text,
            "delay_text": preview_text if delay_text is None else delay_text,
            "source_type": source_type,
            "wait_trigger": wait_trigger,
            "parts": parts,
            "event": event,
        }

    async def _enqueue_message(
        self,
        user_id: str,
        event: AstrMessageEvent,
        text: str,
        *,
        source_type: str = "text",
        delay_text: str | None = None,
        wait_trigger: bool = False,
        image_only: bool = False,
        video_only: bool = False,
    ) -> dict:
        item = await self._build_queue_item(
            event,
            text,
            source_type=source_type,
            delay_text=delay_text,
            wait_trigger=wait_trigger,
            image_only=image_only,
            video_only=video_only,
        )
        item["emoji_library_matched"] = await self._prepare_emoji_library_matches(item)
        self.message_queues[user_id].append(item)
        self._event_refs[user_id] = event
        return item

    async def _prepare_emoji_library_matches(self, item: dict) -> bool:
        if not self._get_config("matched_emoji_skip_wait", False):
            return False
        service = self._emoji_library_service()
        if service is None:
            return False
        matched = False
        for part in item.get("parts", []):
            if part.get("kind") != "image":
                continue
            try:
                result = await service.resolve_component(part["component"])
            except Exception as exc:
                self._debug(f"表情包解释库提前查询失败: {exc}")
                continue
            part["emoji_library_checked"] = True
            part["emoji_library_match"] = result
            if str((result or {}).get("explanation", "")).strip():
                matched = True
        return matched

    # ── Typing state (NapCat input status) ───────────────────

    async def _show_input_status(self, event: AstrMessageEvent) -> None:
        """Show typing indicator via NapCat set_input_status API."""
        try:
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                AiocqhttpMessageEvent,
            )

            if not isinstance(event, AiocqhttpMessageEvent):
                return
            if event.get_group_id():
                return  # Only for private chat
            client = event.bot
            user_id = event.get_sender_id()
            await client.api.call_action(
                "set_input_status", user_id=user_id, event_type=1
            )
        except Exception as e:
            self._debug(f"设置输入状态失败: {e}")

    async def _typing_loop(self, user_id: str, event: AstrMessageEvent) -> None:
        """Periodically show typing status while waiting."""
        interval = self._get_config("typing_interval", 0.5)
        stop_event = self._typing_stop_events.get(user_id)
        if not stop_event:
            return
        while not stop_event.is_set():
            await self._show_input_status(event)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                continue

    def _start_typing(self, user_id: str, event: AstrMessageEvent) -> None:
        """Start typing indicator loop for a user."""
        self._stop_typing(user_id)
        stop_event = asyncio.Event()
        self._typing_stop_events[user_id] = stop_event
        task = asyncio.create_task(self._typing_loop(user_id, event))
        self._typing_tasks[user_id] = task

    def _stop_typing(self, user_id: str) -> None:
        """Stop typing indicator loop for a user."""
        if user_id in self._typing_stop_events:
            self._typing_stop_events[user_id].set()
        self._typing_stop_events.pop(user_id, None)
        task = self._typing_tasks.pop(user_id, None)
        if task and not task.done():
            task.cancel()

    # ── User typing-state detection (NapCat input_status) ────

    @staticmethod
    def _is_typing_event(event: AstrMessageEvent) -> bool:
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return False
        try:
            raw = getattr(event.message_obj, "raw_message", None)
            return (
                isinstance(raw, dict)
                and raw.get("post_type") == "notice"
                and raw.get("sub_type") == "input_status"
            )
        except Exception:
            return False

    @staticmethod
    def _is_recall_event(event: AstrMessageEvent) -> bool:
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return False
        try:
            raw = getattr(event.message_obj, "raw_message", None)
            return (
                isinstance(raw, dict)
                and raw.get("post_type") == "notice"
                and raw.get("notice_type") in ("friend_recall", "group_recall")
            )
        except Exception:
            return False

    @staticmethod
    def _get_message_id(event: AstrMessageEvent):
        try:
            mid = getattr(event.message_obj, "message_id", None)
            if mid is not None:
                return mid
            raw = getattr(event.message_obj, "raw_message", None)
            if isinstance(raw, dict):
                return raw.get("message_id")
        except Exception:
            pass
        return None

    # ── Keyword checks ───────────────────────────────────────

    def _check_skip_words(self, text: str) -> bool:
        skip_words = self._get_config("skip_words", [])
        mode = self._get_config("skip_words_mode", "包含")
        contains = self._is_contains_mode(mode)
        require_message_end = bool(
            self._get_config("skip_words_require_message_end", False)
        )
        stripped = text.strip()
        for word in skip_words:
            if not isinstance(word, str) or not word:
                continue
            if not contains and stripped == word:
                return True
            if contains and require_message_end and stripped.endswith(word):
                return True
            if contains and not require_message_end and word in text:
                return True
        return False

    def _check_wait_keywords(self, text: str) -> bool:
        if not self._get_config("wait_keyword_enabled", True):
            return False
        keywords = self._get_config("wait_keywords", ["等一下"])
        mode = self._get_config("wait_keyword_mode", "完全匹配")
        contains = self._is_contains_mode(mode)
        stripped = text.strip()
        for keyword in keywords:
            if not contains and stripped == keyword:
                return True
            if contains and keyword in text:
                return True
        return False

    # ── Delay calculation ────────────────────────────────────

    def _calc_delay_for_text(self, text: str) -> float:
        word_count = count_words(text)
        long_threshold = self._get_config("long_msg_threshold", 50)
        if word_count >= long_threshold:
            return self._get_config("long_msg_delay_seconds", 2)
        min_delay = self._get_config("min_delay_seconds", 2)
        max_delay = self._get_config("max_delay_seconds", 10)
        short_threshold = self._get_config("short_msg_threshold", 10)
        if word_count <= short_threshold:
            return max_delay
        ratio = (word_count - short_threshold) / (long_threshold - short_threshold)
        delay = max_delay - (max_delay - min_delay) * ratio
        return max(min_delay, min(max_delay, delay))

    def _calc_queue_delay(self, user_id: str) -> float:
        items = self.message_queues[user_id]
        if not items:
            return 0
        total_text = "\n".join(
            item.get("delay_text", item["text"])
            for item in items
            if item.get("delay_text", item["text"])
        )
        return self._calc_delay_for_text(total_text)

    # ── Timer management ─────────────────────────────────────

    def _cancel_timer(self, user_id: str) -> None:
        if user_id in self.timers:
            self.timers[user_id].cancel()
            del self.timers[user_id]

    def _start_timer(self, user_id: str, event: AstrMessageEvent, delay: float) -> None:
        self._cancel_timer(user_id)
        if delay <= 0:
            asyncio.create_task(self._send_merged(user_id))
            return
        self._calc_delay[user_id] = delay
        self._timer_end_time[user_id] = time.time() + delay
        task = asyncio.create_task(self._timer_callback(user_id, delay))
        self.timers[user_id] = task

    async def _timer_callback(self, user_id: str, delay: float) -> None:
        await asyncio.sleep(delay)
        if self.infinite_wait.get(user_id, False):
            return
        await self._send_merged(user_id)

    def _restore_timer_from_last_item(self, user_id: str) -> None:
        items = self.message_queues[user_id]
        if not items:
            self._cancel_timer(user_id)
            self.infinite_wait[user_id] = False
            self.wait_start_time.pop(user_id, None)
            return

        last_item = items[-1]
        event = last_item["event"]
        self._cancel_timer(user_id)
        if last_item.get("wait_trigger", False):
            wait_sec = self._get_config("wait_keyword_seconds", 300)
            self.wait_start_time[user_id] = time.time()
            if wait_sec == 0:
                self.infinite_wait[user_id] = True
                return
            random_range = self._get_config("wait_keyword_random_range", 0)
            if random_range > 0:
                wait_sec = max(
                    1, wait_sec + random.randint(-random_range, random_range)
                )
            self.infinite_wait[user_id] = False
            self._start_timer(user_id, event, wait_sec)
            return

        self.infinite_wait[user_id] = False
        delay = self._calc_queue_delay(user_id) or self._get_config(
            "min_delay_seconds", 2
        )
        self._start_timer(user_id, event, delay)

    # ── Core: send merged message via re-injection ───────────

    async def _send_merged(self, user_id: str) -> None:
        items = self.message_queues[user_id]
        if not items:
            return

        self._stop_typing(user_id)

        # AI busy check: wait for AI to finish before injecting
        if self._get_config("ai_busy_wait_enabled", True) and self._ai_busy.get(
            user_id, False
        ):
            # Cancel existing wait task to avoid duplicates
            if user_id in self._ai_busy_wait_tasks:
                self._ai_busy_wait_tasks[user_id].cancel()
            self._log(f"[{user_id}] AI正在处理中，等待完成后再发送")
            task = asyncio.create_task(self._wait_ai_free(user_id))
            self._ai_busy_wait_tasks[user_id] = task
            return

        all_parts = [part for item in items for part in item.get("parts", [])]
        self._number_media_parts(all_parts)
        image_caption_enabled = self._get_config("image_caption_enabled", False)
        image_captions = await self._caption_images(all_parts)
        merged, replay_components = self._render_parts(
            all_parts,
            image_captions,
            preserve_images=not image_caption_enabled,
        )
        event = self._event_refs.get(user_id)
        if not event:
            self._log(f"[{user_id}] 未找到事件引用，丢弃 {len(items)} 条消息")
            self.message_queues[user_id] = []
            return

        word_count = count_words(merged)
        self._log(
            f"[{user_id}] >>> 发送合并消息: {len(items)}条, {word_count}字, 内容: {merged[:80]}"
        )

        event.message_str = merged
        event.message_obj.message_str = merged
        event.message_obj.message = replay_components
        self._debug(f"[{user_id}] 消息链: {len(replay_components)} ordered components")

        event._force_stopped = False
        event._result = None
        event._has_send_oper = True
        event.call_llm = False
        event.is_at_or_wake_command = True
        event.is_wake = True

        event.set_extra(MERGED_FLAG_KEY, True)

        self.message_queues[user_id] = []
        self._event_refs.pop(user_id, None)
        self.infinite_wait[user_id] = False
        self.wait_start_time.pop(user_id, None)
        self._is_typing.pop(user_id, None)
        self._timer_end_time.pop(user_id, None)
        self._calc_delay.pop(user_id, None)

        try:
            self.context.get_event_queue().put_nowait(event)
        except Exception as e:
            self._log(f"[{user_id}] 重新注入事件失败: {e}")

    @filter.on_llm_request(priority=9_999)
    async def prune_image_context_history(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        del event
        limit = max(0, int(self._get_config("max_image_context_details", 3) or 0))
        incoming_details = count_image_contexts(req.prompt)
        pruned = prune_image_contexts(
            list(req.contexts or []),
            max_details=limit,
            incoming_details=incoming_details,
        )
        if pruned:
            self._log(f"已清理 {pruned} 个旧图片解析上下文")

    # ── Main message handler ─────────────────────────────────

    @filter.event_message_type(EventMessageType.PRIVATE_MESSAGE)
    async def on_message(self, event: AstrMessageEvent) -> None:
        if not self._get_config("enabled", True):
            return

        # ── Typing-state notification (NapCat input_status) ──────────────
        if self._is_typing_event(event):
            if not self._get_config("enable_typing_detection", False):
                event.stop_event()
                return
            raw = event.message_obj.raw_message
            user_id = event.get_sender_id()
            is_typing = "正在输入" in raw.get("status_text", "")
            has_active_queue = bool(self.message_queues.get(user_id))
            if has_active_queue:
                if is_typing:
                    self._is_typing[user_id] = True
                    self._cancel_timer(user_id)
                    # Reset with timeout-protection timer
                    max_wait = float(self._get_config("max_typing_wait", 60.0))
                    self._timer_end_time[user_id] = time.time() + max_wait
                    task = asyncio.create_task(self._timer_callback(user_id, max_wait))
                    self.timers[user_id] = task
                    self._debug(
                        f"[{user_id}] 用户正在输入，重置倒计时（超时保护 {max_wait}s）"
                    )
                elif self._is_typing.get(user_id):
                    self._is_typing[user_id] = False
                    self._cancel_timer(user_id)
                    # Recalculate delay from current queue to avoid inheriting wait_keyword delay
                    delay = self._calc_queue_delay(user_id) or self._get_config(
                        "min_delay_seconds", 2
                    )
                    self._timer_end_time[user_id] = time.time() + delay
                    task = asyncio.create_task(self._timer_callback(user_id, delay))
                    self.timers[user_id] = task
                    self._debug(f"[{user_id}] 用户停止输入，重置倒计时 {delay:.1f}s")
            event.stop_event()
            return

        # ── Recall-message filter ─────────────────────────────────────────
        if self._get_config("enable_recall_filter", True) and self._is_recall_event(
            event
        ):
            try:
                raw = event.message_obj.raw_message
                recalled_mid = raw.get("message_id") if isinstance(raw, dict) else None
            except Exception:
                recalled_mid = None
            user_id = event.get_sender_id()
            if recalled_mid is not None and user_id in self.message_queues:
                before = len(self.message_queues[user_id])
                self.message_queues[user_id] = [
                    item
                    for item in self.message_queues[user_id]
                    if str(item["message_id"]) != str(recalled_mid)
                ]
                remaining = len(self.message_queues[user_id])
                if remaining < before:
                    self._log(
                        f"[{user_id}] 撤回消息已移除 | mid={recalled_mid} | 剩余 {remaining} 条"
                    )
                    if not remaining:
                        self._restore_timer_from_last_item(user_id)
                        self._event_refs.pop(user_id, None)
                    else:
                        self._event_refs[user_id] = self.message_queues[user_id][-1][
                            "event"
                        ]
                        self._restore_timer_from_last_item(user_id)
            event.stop_event()
            return

        # ── Skip messages that start with a command prefix ────────────────
        original_text = self._get_original_text(event)
        command_prefixes = self._get_config("command_prefixes", ["/"])
        if any(original_text.startswith(p) for p in command_prefixes):
            return

        user_id = event.get_sender_id()
        text = event.message_str.strip()
        has_reply = any(isinstance(comp, Reply) for comp in event.message_obj.message)
        has_direct_video = self._has_direct_video(event)
        raw_has_video = self._raw_message_has_video(event)

        if event.get_extra(MERGED_FLAG_KEY):
            # Route through Path A (handler yields ProviderRequest) to set _has_send_oper.
            # Without this, Path B fallback fires and the merged event gets LLM'd twice.
            conv_mgr = self.context.conversation_manager
            umo = event.unified_msg_origin
            cid = await conv_mgr.get_curr_conversation_id(umo)
            conversation = await conv_mgr.get_conversation(umo, cid) if cid else None
            image_urls = []
            for comp in event.message_obj.message:
                if isinstance(comp, Image):
                    try:
                        image_urls.append(await comp.convert_to_file_path())
                    except Exception:
                        pass
            yield event.request_llm(
                prompt=event.message_str,
                conversation=conversation,
                image_urls=image_urls or None,
            )
            event.stop_event()
            return

        is_voice, voice_text = await self._resolve_voice(event, text)
        if is_voice:
            text = self._format_voice_message(voice_text, failed=not voice_text)

        # Image-only message: treat as wait keyword (long wait)
        is_image_only = (
            not is_voice and not text and not has_reply and self._is_image_only(event)
        )

        is_video_only = not is_voice and not text and not has_reply and has_direct_video
        if raw_has_video and not has_direct_video:
            self._log(
                f"[{user_id}] 原始消息包含视频，但未找到可重放 Video 组件，已停止事件避免生成错误回复"
            )
            event.stop_event()
            return

        if not text and not is_image_only and not is_video_only and not has_reply:
            return

        queue_len = len(self.message_queues[user_id])
        voice_wait = is_voice and self._get_config("voice_wait_enabled", True)
        image_wait = is_image_only and self._get_config("image_wait_enabled", True)
        video_wait = is_video_only and self._get_config("video_wait_enabled", True)

        # The current message decides the timer mode. Media wait wins only for
        # this message; a later normal message restores regular routing.
        is_wait_keyword = (
            not is_voice and bool(text) and self._check_wait_keywords(text)
        )
        is_wait = voice_wait or image_wait or video_wait or is_wait_keyword
        if is_wait:
            event.stop_event()
            source_type = (
                "voice"
                if is_voice
                else "image"
                if is_image_only
                else "video"
                if is_video_only
                else "text"
            )
            delay_text = voice_text if is_voice else None
            display_text = (
                voice_text
                if is_voice
                else text
                if text
                else "[视频]"
                if is_video_only
                else "[图片]"
            )
            item = await self._enqueue_message(
                user_id,
                event,
                text,
                source_type=source_type,
                delay_text=delay_text,
                wait_trigger=True,
                image_only=is_image_only,
                video_only=is_video_only,
            )
            self._cancel_timer(user_id)
            self.infinite_wait[user_id] = False
            if item.get("emoji_library_matched", False):
                self._log(f"[{user_id}] 命中表情包解释库，跳过等待并立即发送当前队列")
                await self._send_merged(user_id)
                return
            wait_sec = self._get_config("wait_keyword_seconds", 300)
            trigger = (
                "语音"
                if is_voice
                else "图片"
                if is_image_only
                else "视频"
                if is_video_only
                else "关键词"
            )
            if wait_sec == 0:
                self.infinite_wait[user_id] = True
                self.wait_start_time[user_id] = time.time()
                self._log(
                    f'[{user_id}] 触发无限等待({trigger}): "{display_text[:30]}" | 队列: {queue_len + 1}条'
                )
            else:
                random_range = self._get_config("wait_keyword_random_range", 0)
                if random_range > 0:
                    wait_sec = max(
                        1, wait_sec + random.randint(-random_range, random_range)
                    )
                self.wait_start_time[user_id] = time.time()
                self._log(
                    f'[{user_id}] 触发等待({trigger}): "{display_text[:30]}" | 等待: {wait_sec}秒 | 队列: {queue_len + 1}条'
                )
                self._start_timer(user_id, event, wait_sec)
            return

        # Any non-wait message exits a previous infinite wait before normal routing.
        self.infinite_wait[user_id] = False

        # Skip keyword (new text only): flush the queue immediately or after jitter.
        if not is_voice and text and self._check_skip_words(text):
            event.stop_event()
            await self._enqueue_message(user_id, event, text)
            self._cancel_timer(user_id)
            if self._get_config("skip_words_random_delay_enabled", False):
                delay_min = float(self._get_config("skip_words_random_delay_min", 0.5))
                delay_max = float(self._get_config("skip_words_random_delay_max", 3.0))
                delay_min, delay_max = (
                    min(delay_min, delay_max),
                    max(delay_min, delay_max),
                )
                rand_delay = random.uniform(delay_min, delay_max)
                self._log(
                    f'[{user_id}] 命中跳过词: "{text[:30]}" | 队列: {len(self.message_queues[user_id])}条, 随机等待 {rand_delay:.2f}s 后发送'
                )
                self._start_timer(user_id, event, rand_delay)
            else:
                self._log(
                    f'[{user_id}] 命中跳过词: "{text[:30]}" | 队列: {len(self.message_queues[user_id])}条, 立即发送'
                )
                await self._send_merged(user_id)
            return

        # Normal message: stop event, queue it.
        event.stop_event()
        item = await self._enqueue_message(
            user_id,
            event,
            text,
            source_type=("voice" if is_voice else "video" if is_video_only else "text"),
            delay_text=voice_text if is_voice else text,
            video_only=is_video_only,
        )
        if item.get("emoji_library_matched", False):
            self._cancel_timer(user_id)
            self._log(f"[{user_id}] 命中表情包解释库，跳过普通延迟并立即发送当前队列")
            await self._send_merged(user_id)
            return
        if user_id not in self.wait_start_time:
            self.wait_start_time[user_id] = time.time()

        new_queue_len = len(self.message_queues[user_id])

        # A voice that triggered long wait returned above. Other messages use the
        # regular count threshold, including voice when voice wait is disabled.
        max_count = self._get_config("max_message_count", 10)
        if new_queue_len >= max_count:
            self._log(
                f"[{user_id}] 达到条数阈值({new_queue_len}/{max_count}), 立即发送"
            )
            self._cancel_timer(user_id)
            await self._send_merged(user_id)
            return

        delay = self._calc_queue_delay(user_id)
        display_text = voice_text if is_voice else "[视频]" if is_video_only else text
        self._log(
            f'[{user_id}] 收到消息: "{display_text[:30]}" | 队列: {new_queue_len}条 | 等待: {delay:.0f}秒后发送'
        )
        self._start_timer(user_id, event, delay)

    # ── Plugin commands ───────────────────────────────────────

    @filter.command("合并帮助", desc="显示消息合并插件的帮助信息")
    async def cmd_help(self, event: AstrMessageEvent):
        text = (
            "[消息合并] 可用命令:\n"
            "/合并帮助 - 显示此帮助\n"
            "/合并状态 - 查看当前队列状态\n"
            "/立即发送 - 立即发送队列中的消息\n"
            "/清空队列 - 清空消息队列\n"
            "/合并配置 - 查看当前配置\n"
            "/合并调试 - 切换调试模式"
        )
        yield event.plain_result(text)

    @filter.command("合并状态", desc="查看当前消息队列状态")
    async def cmd_status(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        q = len(self.message_queues[user_id])
        w = sum(
            count_words(item.get("delay_text", item["text"]))
            for item in self.message_queues[user_id]
            if item.get("delay_text", item["text"])
        )
        inf = "是" if self.infinite_wait.get(user_id, False) else "否"
        elapsed = ""
        start = self.wait_start_time.get(user_id)
        if start:
            e = time.time() - start
            elapsed = f" | 已等待: {e:.0f}秒"
        yield event.plain_result(
            f"[消息合并] 队列: {q}条消息({w}字) | 无限等待: {inf}{elapsed}"
        )

    @filter.command("立即发送", desc="立即发送队列中的所有消息")
    async def cmd_send_now(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        if not self.message_queues[user_id]:
            yield event.plain_result("[消息合并] 当前没有待发送的消息")
            return
        self._cancel_timer(user_id)
        await self._send_merged(user_id)
        yield event.plain_result("[消息合并] 已立即发送")

    @filter.command("清空队列", desc="清空当前消息队列")
    async def cmd_clear(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        self._cancel_timer(user_id)
        self.message_queues[user_id] = []
        self.infinite_wait[user_id] = False
        self.wait_start_time.pop(user_id, None)
        self._event_refs.pop(user_id, None)
        yield event.plain_result("[消息合并] 已清空消息队列")

    @filter.command("合并配置", desc="查看当前合并配置")
    async def cmd_config(self, event: AstrMessageEvent):
        lines = [
            f"启用: {self._get_config('enabled', True)}",
            f"调试模式: {self._get_config('debug_mode', False)}",
            f"最短延迟: {self._get_config('min_delay_seconds', 2)}秒",
            f"最长延迟: {self._get_config('max_delay_seconds', 10)}秒",
            f"短消息阈值: {self._get_config('short_msg_threshold', 10)}字",
            f"长消息阈值: {self._get_config('long_msg_threshold', 50)}字",
            f"长消息延迟: {self._get_config('long_msg_delay_seconds', 2)}秒",
            f"消息条数阈值: {self._get_config('max_message_count', 10)}条",
            f"跳过关键词: {self._get_config('skip_words', [])}",
            f"跳过词模式: {self._get_config('skip_words_mode', '包含')}",
            f"跳过词须在消息末尾: {self._get_config('skip_words_require_message_end', False)}",
            f"等待关键词: {self._get_config('wait_keywords', ['等一下'])}",
            f"等待词模式: {self._get_config('wait_keyword_mode', '完全匹配')}",
            f"等待时间: {self._get_config('wait_keyword_seconds', 300)}秒 (0=无限)",
            f"LLM判断: {self._get_config('llm_judge_enabled', False)}",
            f"AI忙感知: {self._get_config('ai_busy_wait_enabled', True)}",
            f"AI忙检查间隔: {self._get_config('ai_busy_check_interval', 3)}秒",
            f"AI忙最大等待: {self._get_config('ai_busy_max_wait', 120)}秒",
            f"输入状态显示: {self._get_config('typing_status_enabled', True)}",
            f"输入状态间隔: {self._get_config('typing_interval', 0.5)}秒",
            f"图片触发超长等待: {self._get_config('image_wait_enabled', True)}",
            f"视频触发超长等待: {self._get_config('video_wait_enabled', True)}",
            f"插件级图片转述: {self._get_config('image_caption_enabled', False)}",
            f"单轮图片预处理: {self._get_config('image_caption_compress_enabled', False)}",
            f"图片预处理最长边: {self._get_config('image_caption_compress_max_size', 1280)}px",
            f"图片预处理质量: {self._get_config('image_caption_compress_quality', 85)}",
            f"图片上下文完整保留: {self._get_config('max_image_context_details', 3)}张 (0=不限制)",
            f"首选图片转述模型: {self._get_config('image_caption_provider_id', '') or '未配置'}",
            f"图片转述回退列表: {self._configured_fallback_ids('image_caption_fallback_provider_ids', ('image_caption_fallback_provider_id', 'image_caption_fallback_provider_id_2')) or '未配置'}",
            f"语音消息感知: {self._get_config('voice_message_enabled', True)}",
            f"语音触发超长等待: {self._get_config('voice_wait_enabled', True)}",
            f"首选 STT: {self._get_config('stt_provider_id', '') or '未配置'}",
            f"STT 回退列表: {self._configured_fallback_ids('stt_fallback_provider_ids', ('stt_fallback_provider_id', 'stt_fallback_provider_id_2')) or '未配置'}",
            f"等待随机变化: ±{self._get_config('wait_keyword_random_range', 0)}秒",
        ]
        yield event.plain_result("[消息合并] 当前配置:\n" + "\n".join(lines))

    @filter.command("合并调试", desc="切换调试模式开关")
    async def cmd_debug(self, event: AstrMessageEvent):
        current = self._get_config("debug_mode", False)
        new_value = not current
        self.config["debug_mode"] = new_value
        state = "开启" if new_value else "关闭"
        yield event.plain_result(f"[消息合并] 调试模式已{state}")

    # ── Cleanup ──────────────────────────────────────────────

    async def terminate(self):
        for uid in list(self.timers.keys()):
            self._cancel_timer(uid)
        for uid in list(self._ai_busy_wait_tasks.keys()):
            self._ai_busy_wait_tasks[uid].cancel()
        for uid in list(self._typing_tasks.keys()):
            self._stop_typing(uid)
        self.message_queues.clear()
        self._event_refs.clear()
        self.infinite_wait.clear()
        self.wait_start_time.clear()
        self._ai_busy.clear()
        self._ai_busy_wait_tasks.clear()
        self._is_typing.clear()
        self._timer_end_time.clear()
        self._calc_delay.clear()
        self._log("插件已卸载")

    # ── AI busy hooks ───────────────────────────────────────

    def _mark_ai_idle(self, user_id: str, source: str) -> None:
        """Clear busy and typing state after AI processing finishes."""
        was_busy = self._ai_busy.get(user_id, False)
        had_typing_task = user_id in self._typing_tasks
        self._ai_busy[user_id] = False
        self._stop_typing(user_id)
        if was_busy or had_typing_task:
            self._debug(f"[{user_id}] AI处理完成 ({source})")

    @filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req) -> None:
        user_id = event.get_sender_id()
        self._ai_busy[user_id] = True
        if self._get_config("typing_status_enabled", True):
            self._start_typing(user_id, event)
        self._debug(f"[{user_id}] AI开始处理")

    @filter.on_llm_response()
    async def _on_llm_response(self, event: AstrMessageEvent, resp) -> None:
        self._mark_ai_idle(event.get_sender_id(), "LLM响应")

    @filter.on_agent_done()
    async def _on_agent_done(self, event: AstrMessageEvent, *args) -> None:
        self._mark_ai_idle(event.get_sender_id(), "Agent结束")

    @filter.on_llm_tool_respond()
    async def _on_llm_tool_respond(
        self, event: AstrMessageEvent, tool, tool_args, tool_result
    ) -> None:
        if tool_result is None:
            self._mark_ai_idle(event.get_sender_id(), "工具直发结束")

    async def _wait_ai_free(self, user_id: str) -> None:
        """Wait until AI is free, then send merged message."""
        interval = self._get_config("ai_busy_check_interval", 3)
        max_wait = self._get_config("ai_busy_max_wait", 120)
        waited = 0
        while self._ai_busy.get(user_id, False) and waited < max_wait:
            await asyncio.sleep(interval)
            waited += interval
        self._ai_busy_wait_tasks.pop(user_id, None)
        if not self.message_queues.get(user_id):
            return
        if waited >= max_wait:
            self._log(f"[{user_id}] AI忙等待超时({max_wait}秒), 强制发送")
        else:
            self._debug(f"[{user_id}] AI空闲, 发送合并消息")
        await self._send_merged(user_id)
