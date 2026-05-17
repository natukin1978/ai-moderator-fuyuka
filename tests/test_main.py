import unittest
from unittest.mock import AsyncMock, Mock

import main  # main.pyをインポート


class TestMainLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.genai_chat = AsyncMock()
        main.genai_chat = self.genai_chat
        self.genai_chat.send_message_by_json.side_effect = [
            "初コメ",
            "ありがとう",
        ]

        self.read_ng_words = Mock()
        main.read_ng_words = self.read_ng_words
        self.read_ng_words.return_value = ["初コメ"]

    async def test_send_message_genai_chat(self):
        json_data = {
            "dateTime": "",
            "id": "id",
        }
        response_text = await main.send_message_genai_chat(json_data)
        self.assertEqual("ありがとう", response_text)

    async def test_chat_endpoint(self):
        json_data = main.ChatModel()
        json_data.noisy=True
        json_data.content="a"
        await main.chat_endpoint("", json_data)
        json_data.content="b"
        await main.chat_endpoint("", json_data)
        json_data.content="c"
        await main.chat_endpoint("", json_data)
