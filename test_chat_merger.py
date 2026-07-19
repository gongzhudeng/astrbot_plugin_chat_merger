from __future__ import annotations

# ruff: noqa: E402, I001

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

PLUGIN_DIR = Path(__file__).resolve().parent
ASTRBOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(ASTRBOT_DIR))

from astrbot.core.message.components import Image, Plain, Reply, Video
from astrbot_plugin_chat_merger.image_caption import (
    caption_ordered_images,
    is_refusal_text,
    parse_caption_map,
)
from astrbot_plugin_chat_merger.main import ChatMergerPlugin, MERGED_FLAG_KEY


class _FakeQueue:
    def __init__(self) -> None:
        self.items = []

    def put_nowait(self, item) -> None:
        self.items.append(item)


class _FakeContext:
    def __init__(self) -> None:
        self.queue = _FakeQueue()

    def get_event_queue(self):
        return self.queue


class _FakeEvent:
    def __init__(
        self,
        message,
        *,
        message_str: str = "",
        message_id: int = 1,
    ) -> None:
        self.message_str = message_str
        self.message_obj = type(
            "MessageObject",
            (),
            {
                "message": list(message),
                "message_str": message_str,
                "message_id": message_id,
                "raw_message": {},
            },
        )()
        self._extras = {}
        self._force_stopped = True
        self._result = object()
        self._has_send_oper = False
        self.call_llm = True
        self.is_at_or_wake_command = False
        self.is_wake = False

    def get_self_id(self):
        return "bot"

    def set_extra(self, key, value) -> None:
        self._extras[key] = value


class ChatMergerVideoTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _plugin() -> ChatMergerPlugin:
        plugin = object.__new__(ChatMergerPlugin)
        plugin.context = _FakeContext()
        plugin.config = {"ai_busy_wait_enabled": False}
        plugin.message_queues = {}
        plugin.timers = {}
        plugin._event_refs = {}
        plugin.infinite_wait = {}
        plugin.wait_start_time = {}
        plugin._ai_busy = {}
        plugin._ai_busy_wait_tasks = {}
        plugin._typing_tasks = {}
        plugin._typing_stop_events = {}
        plugin._is_typing = {}
        plugin._timer_end_time = {}
        plugin._calc_delay = {}
        return plugin

    async def test_video_only_queue_item_preserves_component_and_placeholder(
        self,
    ) -> None:
        plugin = self._plugin()
        video = Video(file="D:/cache/video.mp4", path="D:/cache/video.mp4")
        event = _FakeEvent([video])

        item = await plugin._build_queue_item(event, "", video_only=True)

        self.assertEqual(item["text"], "[视频]")
        self.assertEqual(item["source_type"], "text")
        self.assertEqual(
            [part["component"] for part in item["parts"] if part["kind"] == "video"],
            [video],
        )

    async def test_text_video_text_replay_keeps_order_and_video(self) -> None:
        plugin = self._plugin()
        first = _FakeEvent([Plain("给你看看")], message_str="给你看看", message_id=1)
        video = Video(file="D:/cache/video.mp4", path="D:/cache/video.mp4")
        middle = _FakeEvent([video], message_id=2)
        last = _FakeEvent([Plain("能看到不")], message_str="能看到不", message_id=3)
        user_id = "user"
        plugin.message_queues[user_id] = []

        await plugin._enqueue_message(user_id, first, "给你看看")
        await plugin._enqueue_message(
            user_id,
            middle,
            "",
            source_type="video",
            video_only=True,
        )
        await plugin._enqueue_message(user_id, last, "能看到不")
        await plugin._send_merged(user_id)
        await asyncio.sleep(0)

        replayed = plugin.context.queue.items[0]
        self.assertEqual(replayed.message_str, "给你看看\n[视频1]\n能看到不")
        self.assertEqual(
            [
                part.text
                for part in replayed.message_obj.message
                if isinstance(part, Plain)
            ],
            ["给你看看", "[视频1]", "能看到不"],
        )
        self.assertIn(video, replayed.message_obj.message)
        self.assertTrue(replayed._extras[MERGED_FLAG_KEY])

    async def test_quoted_video_becomes_semantic_placeholder_only(self) -> None:
        plugin = self._plugin()
        quoted_video = Video(file="D:/cache/quoted.mp4")
        reply = Reply(
            id="10",
            chain=[quoted_video],
            sender_id="user",
            sender_nickname="Mando",
            message_str="[Video]",
        )
        event = _FakeEvent([reply, Plain("这是啥")], message_str="这是啥")

        with (
            patch(
                "astrbot_plugin_chat_merger.main.extract_quoted_message_text",
                new=AsyncMock(return_value="[Video]"),
            ),
            patch(
                "astrbot_plugin_chat_merger.main.extract_quoted_message_images",
                new=AsyncMock(return_value=[]),
            ),
        ):
            item = await plugin._build_queue_item(event, "这是啥")

        self.assertIn("[引用视频消息]", item["text"])
        self.assertNotIn("[Video]", item["text"])
        self.assertNotIn(
            quoted_video,
            [part.get("component") for part in item["parts"]],
        )

    def test_component_type_video_is_detected_without_class_identity(self) -> None:
        plugin = self._plugin()
        foreign_video = type(
            "ForeignVideo",
            (),
            {"type": type("ForeignType", (), {"value": "Video"})()},
        )()

        self.assertTrue(plugin._is_video_component(foreign_video))

    def test_regular_quote_and_images_keep_existing_semantics(self) -> None:
        plugin = self._plugin()
        reply = Reply(id="11", sender_id="user", sender_nickname="Mando")
        event = _FakeEvent([])

        formatted = plugin._format_quoted_message(
            event,
            reply,
            "原消息",
            2,
        )

        self.assertIn("原消息", formatted)
        self.assertIn("[引用图片: 2张]", formatted)
        self.assertNotIn("[引用视频消息]", formatted)


class _FakeVisionResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class _FakeVisionProvider:
    def __init__(self, provider_id: str, result) -> None:
        self.provider_id = provider_id
        self.result = result
        self.calls = 0

    def meta(self):
        return type("Meta", (), {"id": self.provider_id})()

    async def text_chat(self, **kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return _FakeVisionResponse(self.result)


class _FakeImage:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def convert_to_file_path(self) -> str:
        return str(self.path)


class ChatMergerImageCaptionTests(unittest.IsolatedAsyncioTestCase):
    def test_caption_parser_and_refusal_detection(self) -> None:
        self.assertEqual(
            parse_caption_map('{"图1":"第一张","图2":"第二张"}', ["图1", "图2"]),
            {"图1": "第一张", "图2": "第二张"},
        )
        self.assertTrue(is_refusal_text("抱歉，我不能描述这张图", ["我不能描述"]))

    async def test_refusal_uses_fallback_and_keeps_ordered_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.jpg"
            image_path.write_bytes(b"jpeg")
            primary = _FakeVisionProvider("primary", "抱歉，我不能描述这张图")
            fallback = _FakeVisionProvider("fallback", '{"图1":"一个水池"}')
            parts = [
                {"kind": "text", "text": "这个好恶心"},
                {"kind": "image", "id": "图1", "component": _FakeImage(image_path)},
            ]

            result = await caption_ordered_images(
                parts,
                providers=[primary, fallback],
                prompt="结合用户文字转述",
                refusal_keywords=["我不能描述"],
                timeout_seconds=5,
                max_images=9,
            )

        self.assertEqual(result, {"图1": "一个水池"})
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)

    async def test_disabled_caption_preserves_original_image(self) -> None:
        plugin = ChatMergerVideoTests._plugin()
        image = Image(file="D:/cache/image.jpg")
        parts = [{"kind": "image", "id": "图1", "component": image}]

        merged, components = plugin._render_parts(
            parts,
            {},
            preserve_images=True,
        )

        self.assertEqual(merged, "[图1]")
        self.assertIn(image, components)

    async def test_rendered_images_become_text_without_original_attachment(
        self,
    ) -> None:
        plugin = ChatMergerVideoTests._plugin()
        image = Image(file="D:/cache/image.jpg")
        parts = [
            {"kind": "image", "id": "图1", "component": image},
            {"kind": "text", "text": "你看这个"},
        ]

        merged, components = plugin._render_parts(parts, {"图1": "一只猫"})

        self.assertEqual(
            merged,
            '<image_context id="图1">一只猫</image_context>\n你看这个',
        )
        self.assertNotIn(image, components)
