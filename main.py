import asyncio
import random
import re
import time
from collections import defaultdict

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Image, Plain

try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
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
    "彻底告别一问一答式AI聊天。自动合并连续消息、智能延迟后统一回复，AI思考时显示\"对方正在输入…\"。支持关键词触发超长等待、图片智能合并、等待时间随机波动、AI忙感知自动排队、LLM智能延迟判断、输入状态感知、撤回消息过滤，让AI对话真正拥有真人聊天的节奏感",
    "2.0.0",
    "https://github.com/gongzhudeng/astrbot_plugin_chat_merger",
)
class ChatMergerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.message_queues: dict[str, list[str]] = defaultdict(list)
        self.timers: dict[str, asyncio.Task] = {}
        self._event_refs: dict[str, AstrMessageEvent] = {}
        self.infinite_wait: dict[str, bool] = defaultdict(bool)
        self.wait_start_time: dict[str, float] = {}
        self._ai_busy: dict[str, bool] = {}
        self._ai_busy_wait_tasks: dict[str, asyncio.Task] = {}
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._typing_stop_events: dict[str, asyncio.Event] = {}
        self._extra_components: dict[str, list] = defaultdict(list)
        # typing detection state
        self._is_typing: dict[str, bool] = {}
        self._timer_end_time: dict[str, float] = {}
        self._typing_paused_deadline: dict[str, float] = {}
        # recall filter: per-user ordered list of {message_id, text}
        self._message_items: dict[str, list[dict]] = defaultdict(list)
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
    def _is_contains_mode(mode_str: str) -> bool:
        return mode_str in ("contains", "包含")

    @staticmethod
    def _is_image_only(event: AstrMessageEvent) -> bool:
        """Check if message contains only images (no text)."""
        has_image = False
        has_text = False
        for comp in event.message_obj.message:
            if isinstance(comp, Image):
                has_image = True
            elif isinstance(comp, Plain) and comp.text.strip():
                has_text = True
        return has_image and not has_text

    @staticmethod
    def _extract_extra_components(event: AstrMessageEvent) -> list:
        """Extract non-Plain message components (Image, etc.) for preservation."""
        return [comp for comp in event.message_obj.message if not isinstance(comp, Plain)]

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
        stripped = text.strip()
        for word in skip_words:
            if not contains and stripped == word:
                return True
            if contains and word in text:
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
        messages = self.message_queues[user_id]
        if not messages:
            return 0
        total_text = "\n".join(messages)
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
        self._timer_end_time[user_id] = time.time() + delay
        task = asyncio.create_task(self._timer_callback(user_id, delay))
        self.timers[user_id] = task

    async def _timer_callback(self, user_id: str, delay: float) -> None:
        await asyncio.sleep(delay)
        if self.infinite_wait.get(user_id, False):
            return
        await self._send_merged(user_id)

    # ── Core: send merged message via re-injection ───────────

    async def _send_merged(self, user_id: str) -> None:
        messages = self.message_queues[user_id]
        if not messages:
            return

        self._stop_typing(user_id)

        # AI busy check: wait for AI to finish before injecting
        if self._get_config("ai_busy_wait_enabled", True) and self._ai_busy.get(user_id, False):
            # Cancel existing wait task to avoid duplicates
            if user_id in self._ai_busy_wait_tasks:
                self._ai_busy_wait_tasks[user_id].cancel()
            self._log(f"[{user_id}] AI正在处理中，等待完成后再发送")
            task = asyncio.create_task(self._wait_ai_free(user_id))
            self._ai_busy_wait_tasks[user_id] = task
            return

        merged = "\n".join(messages)
        event = self._event_refs.get(user_id)
        if not event:
            self._log(f"[{user_id}] 未找到事件引用，丢弃 {len(messages)} 条消息")
            self.message_queues[user_id] = []
            return

        word_count = count_words(merged)
        self._log(
            f"[{user_id}] >>> 发送合并消息: {len(messages)}条, {word_count}字, 内容: {merged[:80]}"
        )

        event.message_str = merged
        event.message_obj.message_str = merged
        extra = self._extra_components.get(user_id, [])
        event.message_obj.message = [Plain(merged)] + extra
        self._debug(f"[{user_id}] 消息链: 1 Plain + {len(extra)} extra components")

        event._force_stopped = False
        event._result = None
        event._has_send_oper = True
        event.call_llm = False
        event.is_at_or_wake_command = True
        event.is_wake = True

        event.set_extra(MERGED_FLAG_KEY, True)

        self.message_queues[user_id] = []
        self._message_items[user_id] = []
        self._event_refs.pop(user_id, None)
        self._extra_components.pop(user_id, None)
        self.infinite_wait[user_id] = False
        self.wait_start_time.pop(user_id, None)
        self._is_typing.pop(user_id, None)
        self._timer_end_time.pop(user_id, None)

        try:
            self.context.get_event_queue().put_nowait(event)
        except Exception as e:
            self._log(f"[{user_id}] 重新注入事件失败: {e}")

    # ── Main message handler ─────────────────────────────────

    @filter.event_message_type(EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent) -> None:
        if not self._get_config("enabled", True):
            return

        # ── Typing-state notification (NapCat input_status) ──────────────
        if self._get_config("enable_typing_detection", False) and self._is_typing_event(event):
            raw = event.message_obj.raw_message
            user_id = event.get_sender_id()
            is_typing = "正在输入" in raw.get("status_text", "")
            has_active_queue = bool(self.message_queues.get(user_id))
            if has_active_queue:
                if is_typing and not self._is_typing.get(user_id):
                    self._is_typing[user_id] = True
                    # Save remaining time before cancelling
                    saved = max(0.5, self._timer_end_time.get(user_id, time.time()) - time.time())
                    self._typing_paused_deadline[user_id] = saved
                    self._cancel_timer(user_id)
                    # Timeout-protection timer so we don't wait forever
                    max_wait = float(self._get_config("max_typing_wait", 60.0))
                    task = asyncio.create_task(self._timer_callback(user_id, max_wait))
                    self.timers[user_id] = task
                    self._debug(f"[{user_id}] 用户正在输入，暂停倒计时（超时保护 {max_wait}s）")
                elif not is_typing and self._is_typing.get(user_id):
                    self._is_typing[user_id] = False
                    self._cancel_timer(user_id)
                    remaining = self._typing_paused_deadline.pop(user_id, self._get_config("min_delay_seconds", 2))
                    self._timer_end_time[user_id] = time.time() + remaining
                    task = asyncio.create_task(self._timer_callback(user_id, remaining))
                    self.timers[user_id] = task
                    self._debug(f"[{user_id}] 用户停止输入，恢复倒计时 {remaining:.1f}s")
            event.stop_event()
            return

        # ── Recall-message filter ─────────────────────────────────────────
        if self._get_config("enable_recall_filter", True) and self._is_recall_event(event):
            try:
                raw = event.message_obj.raw_message
                recalled_mid = raw.get("message_id") if isinstance(raw, dict) else None
            except Exception:
                recalled_mid = None
            user_id = event.get_sender_id()
            if recalled_mid is not None and user_id in self._message_items:
                before = len(self._message_items[user_id])
                self._message_items[user_id] = [
                    it for it in self._message_items[user_id]
                    if str(it["message_id"]) != str(recalled_mid)
                ]
                if len(self._message_items[user_id]) < before:
                    self.message_queues[user_id] = [
                        it["text"] for it in self._message_items[user_id] if it["text"]
                    ]
                    self._log(f"[{user_id}] 撤回消息已移除 | mid={recalled_mid} | 剩余 {len(self._message_items[user_id])} 条")
                    if not self._message_items[user_id]:
                        self._cancel_timer(user_id)
                        self.message_queues[user_id] = []
                        self._event_refs.pop(user_id, None)
                        self._message_items[user_id] = []
            event.stop_event()
            return

        # ── Skip messages that start with a command prefix ────────────────
        original_text = self._get_original_text(event)
        command_prefixes = self._get_config("command_prefixes", ["/"])
        if any(original_text.startswith(p) for p in command_prefixes):
            return

        user_id = event.get_sender_id()
        text = event.message_str.strip()

        # Image-only message: treat as wait keyword (long wait)
        is_image_only = (
            not text
            and self._get_config("image_wait_enabled", True)
            and self._is_image_only(event)
        )

        if not text and not is_image_only:
            return

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

        queue_len = len(self.message_queues[user_id])

        # Skip keyword (text only): flush entire queue + skip keyword immediately
        if text and self._check_skip_words(text):
            event.stop_event()
            self.message_queues[user_id].append(text)
            self._message_items[user_id].append({"message_id": self._get_message_id(event), "text": text})
            self._event_refs[user_id] = event
            self._extra_components[user_id].extend(self._extract_extra_components(event))
            self._log(
                f"[{user_id}] 命中跳过词: \"{text[:30]}\" | 队列: {len(self.message_queues[user_id])}条, 立即发送"
            )
            self._cancel_timer(user_id)
            await self._send_merged(user_id)
            return

        # Wait keyword or image-only
        is_wait = (text and self._check_wait_keywords(text)) or is_image_only
        if is_wait:
            event.stop_event()
            display_text = text if text else "[图片]"
            queued_text = text if text else "[图片]"
            self.message_queues[user_id].append(queued_text)
            self._message_items[user_id].append({"message_id": self._get_message_id(event), "text": queued_text})
            self._event_refs[user_id] = event
            self._extra_components[user_id].extend(self._extract_extra_components(event))
            wait_sec = self._get_config("wait_keyword_seconds", 300)
            if wait_sec == 0:
                self.infinite_wait[user_id] = True
                self.wait_start_time[user_id] = time.time()
                trigger = "图片" if is_image_only else "关键词"
                self._log(
                    f"[{user_id}] 触发无限等待({trigger}): \"{display_text[:30]}\" | 队列: {queue_len + 1}条"
                )
            else:
                random_range = self._get_config("wait_keyword_random_range", 0)
                if random_range > 0:
                    wait_sec = max(1, wait_sec + random.randint(-random_range, random_range))
                self.wait_start_time[user_id] = time.time()
                trigger = "图片" if is_image_only else "关键词"
                self._log(
                    f"[{user_id}] 触发等待({trigger}): \"{display_text[:30]}\" | 等待: {wait_sec}秒 | 队列: {queue_len + 1}条"
                )
                self._start_timer(user_id, event, wait_sec)
            return

        # Normal message: stop event, queue it
        event.stop_event()
        self.message_queues[user_id].append(text)
        self._message_items[user_id].append({"message_id": self._get_message_id(event), "text": text})
        self._event_refs[user_id] = event
        self._extra_components[user_id].extend(self._extract_extra_components(event))
        if user_id not in self.wait_start_time:
            self.wait_start_time[user_id] = time.time()

        new_queue_len = len(self.message_queues[user_id])

        # Check message count threshold
        max_count = self._get_config("max_message_count", 10)
        if new_queue_len >= max_count:
            self._log(
                f"[{user_id}] 达到条数阈值({new_queue_len}/{max_count}), 立即发送"
            )
            self._cancel_timer(user_id)
            await self._send_merged(user_id)
            return

        # Determine delay
        delay = self._calc_queue_delay(user_id)
        self._log(
            f"[{user_id}] 收到消息: \"{text[:30]}\" | 队列: {new_queue_len}条 | 等待: {delay:.0f}秒后发送"
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
        w = sum(count_words(m) for m in self.message_queues[user_id])
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
        self._extra_components[user_id] = []
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
        self._message_items.clear()
        self._event_refs.clear()
        self.infinite_wait.clear()
        self.wait_start_time.clear()
        self._ai_busy.clear()
        self._ai_busy_wait_tasks.clear()
        self._extra_components.clear()
        self._is_typing.clear()
        self._timer_end_time.clear()
        self._typing_paused_deadline.clear()
        self._log("插件已卸载")

    # ── AI busy hooks ───────────────────────────────────────

    @filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req) -> None:
        user_id = event.get_sender_id()
        self._ai_busy[user_id] = True
        if self._get_config("typing_status_enabled", True):
            self._start_typing(user_id, event)
        self._debug(f"[{user_id}] AI开始处理")

    @filter.on_llm_response()
    async def _on_llm_response(self, event: AstrMessageEvent, resp) -> None:
        user_id = event.get_sender_id()
        self._ai_busy[user_id] = False
        self._stop_typing(user_id)
        self._debug(f"[{user_id}] AI处理完成")

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