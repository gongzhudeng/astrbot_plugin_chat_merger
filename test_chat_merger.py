from __future__ import annotations

# ruff: noqa: E402, I001

import asyncio
import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PLUGIN_DIR = Path(__file__).resolve().parent
ASTRBOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(ASTRBOT_DIR))

from astrbot.core.message.components import Image, Plain, Reply, Video
from astrbot_plugin_chat_merger.image_caption import (
    _path_to_data_url,
    caption_ordered_images,
    is_refusal_text,
    parse_caption_map,
)
from astrbot_plugin_chat_merger.image_preprocess import prepare_image_bytes
from astrbot_plugin_chat_merger.image_context import (
    IMAGE_CONTEXT_PRUNED,
    count_image_contexts,
    prune_image_contexts,
    wrap_image_context,
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


class ChatMergerSkipWordTests(unittest.TestCase):
    @staticmethod
    def _plugin(config: dict) -> ChatMergerPlugin:
        plugin = object.__new__(ChatMergerPlugin)
        plugin.config = config
        return plugin

    def test_end_required_rejects_middle_match_and_accepts_trailing_whitespace(
        self,
    ) -> None:
        plugin = self._plugin(
            {
                "skip_words": ["。"],
                "skip_words_mode": "包含",
                "skip_words_require_message_end": True,
            }
        )

        self.assertFalse(plugin._check_skip_words("标题。后面还有分享链接！"))
        self.assertTrue(plugin._check_skip_words("再发一遍给你。  \n"))
        self.assertFalse(plugin._check_skip_words("结尾不是句号。！"))

    def test_end_required_supports_multi_character_keywords(self) -> None:
        plugin = self._plugin(
            {
                "skip_words": ["马上"],
                "skip_words_mode": "包含",
                "skip_words_require_message_end": True,
            }
        )

        self.assertFalse(plugin._check_skip_words("马上处理这个问题"))
        self.assertTrue(plugin._check_skip_words("这个问题要处理，马上"))

    def test_disabled_end_requirement_preserves_contains_mode(self) -> None:
        plugin = self._plugin(
            {
                "skip_words": ["。"],
                "skip_words_mode": "包含",
                "skip_words_require_message_end": False,
            }
        )

        self.assertTrue(plugin._check_skip_words("标题。后面还有分享链接！"))

    def test_exact_mode_is_unchanged_when_end_requirement_is_enabled(self) -> None:
        plugin = self._plugin(
            {
                "skip_words": ["立刻"],
                "skip_words_mode": "完全匹配",
                "skip_words_require_message_end": True,
            }
        )

        self.assertTrue(plugin._check_skip_words("  立刻  "))
        self.assertFalse(plugin._check_skip_words("请立刻"))

    def test_empty_skip_words_never_match(self) -> None:
        plugin = self._plugin(
            {
                "skip_words": [""],
                "skip_words_mode": "包含",
                "skip_words_require_message_end": True,
            }
        )

        self.assertFalse(plugin._check_skip_words("任意消息"))


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
        self.requests = []

    def meta(self):
        return type("Meta", (), {"id": self.provider_id})()

    async def text_chat(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return _FakeVisionResponse(self.result)


class _FakeImage:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def convert_to_file_path(self) -> str:
        return str(self.path)


class ChatMergerImagePreprocessTests(unittest.TestCase):
    @staticmethod
    def _encode_image(
        size: tuple[int, int],
        *,
        mode: str = "RGB",
        color="white",
        image_format: str = "PNG",
    ) -> bytes:
        from PIL import Image as PillowImage

        image = PillowImage.new(mode, size, color)
        output = io.BytesIO()
        image.save(output, format=image_format)
        return output.getvalue()

    @staticmethod
    def _decode_data_url(data_url: str) -> tuple[str, bytes]:
        header, payload = data_url.split(",", 1)
        return header, base64.b64decode(payload)

    def test_disabled_preprocess_keeps_original_png_and_mime(self) -> None:
        original = self._encode_image((320, 180))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "small.png"
            path.write_bytes(original)

            header, payload = self._decode_data_url(
                _path_to_data_url(path, compress_enabled=False)
            )

        self.assertEqual(header, "data:image/png;base64")
        self.assertEqual(payload, original)

    def test_large_image_is_resized_without_upscaling(self) -> None:
        from PIL import Image as PillowImage

        large = self._encode_image((2400, 1200))
        prepared = prepare_image_bytes(large, max_size=1280, quality=85)
        with PillowImage.open(io.BytesIO(prepared)) as image:
            self.assertEqual(image.size, (1280, 640))

        small = self._encode_image((640, 320))
        self.assertEqual(
            prepare_image_bytes(small, max_size=1280, quality=85),
            small,
        )

    def test_transparent_png_uses_white_jpeg_background_when_resized(self) -> None:
        from PIL import Image as PillowImage

        transparent = self._encode_image(
            (1600, 800),
            mode="RGBA",
            color=(255, 0, 0, 0),
        )
        prepared = prepare_image_bytes(transparent, max_size=1280, quality=85)

        self.assertTrue(prepared.startswith(b"\xff\xd8\xff"))
        with PillowImage.open(io.BytesIO(prepared)) as image:
            pixel = image.convert("RGB").getpixel((0, 0))
        self.assertTrue(all(channel >= 245 for channel in pixel))

    def test_large_file_keeps_original_when_reencoding_is_not_smaller(self) -> None:
        original = b"x" * (1024 * 1024 + 1)
        image = MagicMock()
        image.size = (1280, 960)
        image.mode = "RGB"
        image.getbands.return_value = ("R", "G", "B")
        source = MagicMock()
        source.__enter__.return_value = image

        def write_larger_result(target, **kwargs) -> None:
            del kwargs
            target.write(original + b"larger")

        image.save.side_effect = write_larger_result
        with (
            patch(
                "astrbot_plugin_chat_merger.image_preprocess.Image.open",
                return_value=source,
            ),
            patch(
                "astrbot_plugin_chat_merger.image_preprocess.ImageOps.exif_transpose",
                return_value=image,
            ),
        ):
            prepared = prepare_image_bytes(original, max_size=1280, quality=85)

        self.assertEqual(prepared, original)

    def test_small_webp_keeps_original_bytes_and_mime(self) -> None:
        original = self._encode_image((320, 180), image_format="WEBP")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "small.webp"
            path.write_bytes(original)

            header, payload = self._decode_data_url(
                _path_to_data_url(path, compress_enabled=True)
            )

        self.assertEqual(header, "data:image/webp;base64")
        self.assertEqual(payload, original)


class ChatMergerImageCaptionTests(unittest.IsolatedAsyncioTestCase):
    def test_ordered_fallback_config_prefers_new_list_and_supports_legacy(self):
        plugin = ChatMergerVideoTests._plugin()
        plugin.config = {
            "图片消息": {
                "image_caption_fallback_provider_ids": ["second", "third"],
                "image_caption_fallback_provider_id": "legacy-one",
                "image_caption_fallback_provider_id_2": "legacy-two",
            }
        }
        self.assertEqual(
            plugin._configured_fallback_ids(
                "image_caption_fallback_provider_ids",
                (
                    "image_caption_fallback_provider_id",
                    "image_caption_fallback_provider_id_2",
                ),
            ),
            ["second", "third"],
        )

        plugin.config = {
            "图片消息": {
                "image_caption_fallback_provider_ids": [],
                "image_caption_fallback_provider_id": "legacy-one",
            }
        }
        self.assertEqual(
            plugin._configured_fallback_ids(
                "image_caption_fallback_provider_ids",
                (
                    "image_caption_fallback_provider_id",
                    "image_caption_fallback_provider_id_2",
                ),
            ),
            [],
        )

        plugin.config = {
            "图片消息": {
                "image_caption_fallback_provider_id": "legacy-one",
                "image_caption_fallback_provider_id_2": "legacy-two",
            }
        }
        self.assertEqual(
            plugin._configured_fallback_ids(
                "image_caption_fallback_provider_ids",
                (
                    "image_caption_fallback_provider_id",
                    "image_caption_fallback_provider_id_2",
                ),
            ),
            ["legacy-one", "legacy-two"],
        )

    def test_caption_parser_and_refusal_detection(self) -> None:
        self.assertEqual(
            parse_caption_map('{"图1":"第一张","图2":"第二张"}', ["图1", "图2"]),
            {"图1": "第一张", "图2": "第二张"},
        )
        self.assertEqual(
            parse_caption_map('{"图1": "第一张", "图2": "第二张"}', ["图1", "图2"]),
            {"图1": "第一张", "图2": "第二张"},
        )
        self.assertEqual(
            parse_caption_map(
                '{"图1": "设备形似"眼睛"造型，品牌为"JISSBON"。"}',
                ["图1"],
            ),
            {"图1": '设备形似"眼睛"造型，品牌为"JISSBON"。'},
        )
        self.assertEqual(
            parse_caption_map("这是一个黑色电子玩具。", ["图1"]),
            {"图1": "这是一个黑色电子玩具。"},
        )
        self.assertEqual(
            parse_caption_map(
                "图一是黑色电子玩具。\n第二张为白色包装盒。",
                ["图1", "图2"],
            ),
            {"图1": "黑色电子玩具。", "图2": "白色包装盒。"},
        )
        self.assertEqual(
            parse_caption_map("两张图都是电子玩具。", ["图1", "图2"]),
            {},
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

    async def test_partial_mapping_only_retries_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.jpg"
            second_path = Path(temp_dir) / "second.jpg"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            primary = _FakeVisionProvider("primary", "图一：黑色玩具")
            fallback = _FakeVisionProvider("fallback", "白色包装盒")
            parts = [
                {"kind": "image", "id": "图1", "component": _FakeImage(first_path)},
                {"kind": "image", "id": "图2", "component": _FakeImage(second_path)},
            ]

            result = await caption_ordered_images(
                parts,
                providers=[primary, fallback],
                prompt="转述图片",
                refusal_keywords=[],
                timeout_seconds=5,
                max_images=9,
            )

        self.assertEqual(result, {"图1": "黑色玩具", "图2": "白色包装盒"})
        fallback_content = fallback.requests[0]["contexts"][0]["content"]
        fallback_image_ids = [
            item["image_url"]["id"]
            for item in fallback_content
            if item.get("type") == "image_url"
        ]
        self.assertEqual(fallback_image_ids, ["图2"])

    async def test_failed_image_preparation_does_not_require_missing_mapping(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid.jpg"
            valid_path.write_bytes(b"jpeg")
            missing_path = Path(temp_dir) / "missing.jpg"
            provider = _FakeVisionProvider("primary", '{"图2":"有效图片"}')
            parts = [
                {"kind": "image", "id": "图1", "component": _FakeImage(missing_path)},
                {"kind": "image", "id": "图2", "component": _FakeImage(valid_path)},
            ]

            result = await caption_ordered_images(
                parts,
                providers=[provider],
                prompt="转述图片",
                refusal_keywords=[],
                timeout_seconds=5,
                max_images=9,
            )

        self.assertEqual(result, {"图2": "有效图片"})
        self.assertEqual(provider.calls, 1)

    async def test_queued_image_survives_source_event_cleanup(self) -> None:
        plugin = ChatMergerVideoTests._plugin()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "event-owned.jpg"
            image_path.write_bytes(b"jpeg")
            image = Image(file=str(image_path), url=str(image_path))
            event = _FakeEvent([image])

            item = await plugin._build_queue_item(event, "", image_only=True)
            image_path.unlink()
            plugin._number_media_parts(item["parts"])
            provider = _FakeVisionProvider("primary", '{"图1":"快照仍然可读"}')

            result = await caption_ordered_images(
                item["parts"],
                providers=[provider],
                prompt="转述图片",
                refusal_keywords=[],
                timeout_seconds=5,
                max_images=9,
            )

        queued_image = next(
            part["component"] for part in item["parts"] if part["kind"] == "image"
        )
        self.assertIsNot(queued_image, image)
        self.assertTrue(str(queued_image.file).startswith("base64://"))
        self.assertEqual(result, {"图1": "快照仍然可读"})
        self.assertEqual(provider.calls, 1)

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

        self.assertIn(
            '<image_context id="图1">一只猫</image_context>',
            merged,
        )
        self.assertEqual(count_image_contexts(merged), 1)
        self.assertTrue(merged.endswith("\n你看这个"))
        self.assertNotIn(image, components)

    async def test_early_library_match_is_cached_and_reused(self) -> None:
        plugin = ChatMergerVideoTests._plugin()
        plugin.config.update(
            {
                "matched_emoji_skip_wait": True,
                "image_caption_enabled": True,
            }
        )
        image = Image(file="D:/cache/image.jpg")
        item = {"parts": [{"kind": "image", "id": "图1", "component": image}]}
        service = type(
            "EmojiService",
            (),
            {"resolve_component": AsyncMock(return_value={"explanation": "固定解释"})},
        )()

        with patch.object(plugin, "_emoji_library_service", return_value=service):
            matched = await plugin._prepare_emoji_library_matches(item)
            captions = await plugin._caption_images(item["parts"])

        self.assertTrue(matched)
        self.assertEqual(captions, {"图1": "固定解释"})
        service.resolve_component.assert_awaited_once_with(image)

    async def test_cached_match_renders_when_general_caption_is_disabled(self) -> None:
        plugin = ChatMergerVideoTests._plugin()
        plugin.config.update(
            {
                "matched_emoji_skip_wait": True,
                "image_caption_enabled": False,
            }
        )
        image = Image(file="D:/cache/image.jpg")
        parts = [
            {
                "kind": "image",
                "id": "图1",
                "component": image,
                "emoji_library_checked": True,
                "emoji_library_match": {"explanation": "固定解释"},
            }
        ]

        captions = await plugin._caption_images(parts)
        merged, components = plugin._render_parts(
            parts,
            captions,
            preserve_images=True,
        )

        self.assertEqual(captions, {"图1": "固定解释"})
        self.assertIn("固定解释", merged)
        self.assertNotIn(image, components)

    async def test_disabled_skip_wait_does_not_query_library_early(self) -> None:
        plugin = ChatMergerVideoTests._plugin()
        plugin.config["matched_emoji_skip_wait"] = False
        service = type(
            "EmojiService",
            (),
            {"resolve_component": AsyncMock(return_value={"explanation": "固定解释"})},
        )()
        item = {
            "parts": [
                {
                    "kind": "image",
                    "component": Image(file="D:/cache/image.jpg"),
                }
            ]
        }

        with patch.object(plugin, "_emoji_library_service", return_value=service):
            matched = await plugin._prepare_emoji_library_matches(item)

        self.assertFalse(matched)
        service.resolve_component.assert_not_awaited()


class ImageContextLimitTests(unittest.IsolatedAsyncioTestCase):
    def test_many_image_contexts_keep_only_latest_details(self) -> None:
        contexts = []
        for index in range(8):
            contexts.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"用户原话-{index}\n"
                                f"{wrap_image_context(f'图片详情-{index}')}\n"
                                f"用户尾句-{index}"
                            ),
                        }
                    ],
                }
            )
            contexts.append({"role": "assistant", "content": f"AI回复-{index}"})

        pruned = prune_image_contexts(contexts, max_details=3)

        self.assertEqual(pruned, 5)
        for index in range(8):
            user_text = contexts[index * 2]["content"][0]["text"]
            self.assertIn(f"用户原话-{index}", user_text)
            self.assertIn(f"用户尾句-{index}", user_text)
            self.assertEqual(contexts[index * 2 + 1]["content"], f"AI回复-{index}")
            if index < 5:
                self.assertIn(IMAGE_CONTEXT_PRUNED, user_text)
                self.assertNotIn(f"图片详情-{index}", user_text)
            else:
                self.assertIn(f"图片详情-{index}", user_text)

    def test_multiple_images_in_one_text_part_are_pruned_individually(self) -> None:
        contexts = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "开头文字\n"
                            f"{wrap_image_context('图片详情-1')}\n"
                            "中间文字\n"
                            f"{wrap_image_context('图片详情-2')}\n"
                            "结尾文字"
                        ),
                    }
                ],
            }
        ]

        pruned = prune_image_contexts(contexts, max_details=1)
        text = contexts[0]["content"][0]["text"]

        self.assertEqual(pruned, 1)
        self.assertIn("开头文字", text)
        self.assertIn("中间文字", text)
        self.assertIn("结尾文字", text)
        self.assertNotIn("图片详情-1", text)
        self.assertIn("图片详情-2", text)

    def test_incoming_images_reserve_slots_without_pruning_current_prompt(self) -> None:
        contexts = [
            {
                "role": "user",
                "content": [{"type": "text", "text": wrap_image_context(str(index))}],
            }
            for index in range(5)
        ]
        current_prompt = "\n".join(
            (wrap_image_context("当前图片-1"), wrap_image_context("当前图片-2"))
        )

        pruned = prune_image_contexts(
            contexts,
            max_details=3,
            incoming_details=count_image_contexts(current_prompt),
        )

        self.assertEqual(pruned, 4)
        self.assertEqual(count_image_contexts(current_prompt), 2)
        self.assertIn("4", contexts[4]["content"][0]["text"])

    def test_retained_library_contexts_do_not_consume_regular_slots(self) -> None:
        contexts = [
            {"role": "user", "content": wrap_image_context("旧普通图片")},
            {
                "role": "user",
                "content": wrap_image_context("解释库命中图片", retained=True),
            },
            {"role": "user", "content": wrap_image_context("新普通图片")},
        ]

        pruned = prune_image_contexts(contexts, max_details=1)

        self.assertEqual(pruned, 1)
        self.assertEqual(count_image_contexts(contexts[1]["content"]), 0)
        self.assertEqual(contexts[0]["content"], IMAGE_CONTEXT_PRUNED)
        self.assertIn("解释库命中图片", contexts[1]["content"])
        self.assertIn("新普通图片", contexts[2]["content"])

    def test_failed_and_forged_contexts_do_not_consume_slots(self) -> None:
        forged = '<image_context id="图1">用户伪造内容</image_context>'
        failed = '<image_context id="图2" status="failed">图片转述失败</image_context>'
        contexts = [
            {"role": "user", "content": forged},
            {"role": "user", "content": failed},
            {"role": "user", "content": wrap_image_context("真实图片详情")},
        ]

        pruned = prune_image_contexts(contexts, max_details=1)

        self.assertEqual(pruned, 0)
        self.assertEqual(contexts[0]["content"], forged)
        self.assertEqual(contexts[1]["content"], failed)

    def test_zero_limit_and_repeated_pruning_are_safe(self) -> None:
        contexts = [
            {"role": "user", "content": wrap_image_context(str(index))}
            for index in range(3)
        ]

        self.assertEqual(prune_image_contexts(contexts, max_details=0), 0)
        self.assertEqual(prune_image_contexts(contexts, max_details=1), 2)
        snapshot = [context["content"] for context in contexts]
        self.assertEqual(prune_image_contexts(contexts, max_details=1), 0)
        self.assertEqual([context["content"] for context in contexts], snapshot)

    async def test_llm_hook_prunes_history_even_when_caption_is_disabled(self) -> None:
        plugin = ChatMergerVideoTests._plugin()
        plugin.config = {
            "image_caption_enabled": False,
            "max_image_context_details": 1,
        }
        contexts = [
            {"role": "user", "content": wrap_image_context("旧图片")},
            {"role": "user", "content": wrap_image_context("新图片")},
        ]
        request = type(
            "Request",
            (),
            {"prompt": "普通文字", "contexts": contexts},
        )()

        await plugin.prune_image_context_history(None, request)

        self.assertEqual(contexts[0]["content"], IMAGE_CONTEXT_PRUNED)
        self.assertIn("新图片", contexts[1]["content"])
