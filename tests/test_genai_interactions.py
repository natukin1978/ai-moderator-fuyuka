import os
import pickle
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import global_value as g

# 初期設定
g.app_name = "test_app"
g.BASE_PROMPT = "You are a test assistant."
g.ERROR_MESSAGE = "ERROR_MESSAGE"
g.STOP_CANDIDATE_MESSAGE = "STOP_CANDIDATE_MESSAGE"
g.RESOURCE_EXHAUSTED_MESSAGE = "RESOURCE_EXHAUSTED_MESSAGE"
g.config = {
    "google": {
        "geminiApiKey": ["key_0", "key_1", "key_2"],
        "modelName": "gemini-test-model",
        "maxHistoryLength": 4,  # 2往復分（user+model × 2）
    }
}

from google.genai import errors
from genai_interactions import GenAIInteractions


def make_api_error(code: int) -> errors.APIError:
    """テスト用の APIError を生成する。"""
    return errors.APIError(code=code, response_json={"error": {"message": "test", "code": code}})


def make_interaction_mock(output_text: str, interaction_id: str = "test_id_001") -> MagicMock:
    """API レスポンスに相当する interaction オブジェクトのモックを生成する。"""
    mock = MagicMock()
    mock.id = interaction_id
    mock.output_text = output_text
    return mock


class TestBuildContextInput(unittest.TestCase):
    """build_context_input メソッドのテスト。"""

    def setUp(self):
        self.gi = GenAIInteractions()

    def test_returns_message_as_is_when_history_empty(self):
        """履歴が空の場合、メッセージをそのまま返すこと。"""
        result = self.gi.build_context_input("Hello")
        self.assertEqual("Hello", result)

    def test_injects_history_as_context(self):
        """履歴がある場合、コンテキストが先頭に埋め込まれること。"""
        self.gi.history = [
            ("user", "My name is Fuyuka."),
            ("model", "Nice to meet you, Fuyuka!"),
        ]
        result = self.gi.build_context_input("What is my name?")
        self.assertIn("[直前の会話の文脈]", result)
        self.assertIn("ユーザー: My name is Fuyuka.", result)
        self.assertIn("アシスタント: Nice to meet you, Fuyuka!", result)
        self.assertIn("What is my name?", result)


class TestRemoveOldHistory(unittest.TestCase):
    """remove_old_history メソッドのテスト。"""

    def setUp(self):
        self.gi = GenAIInteractions()
        # テスト実行直前に確実に設定を固定する
        if "google" not in g.config:
            g.config["google"] = {}
        g.config["google"]["maxHistoryLength"] = 4

    def test_does_not_remove_within_limit(self):
        """maxHistoryLength 以内なら履歴が削除されないこと。"""
        # maxHistoryLength=4, 4エントリなら削除しない
        self.gi.history = [
            ("user", "1"), ("model", "a"),
            ("user", "2"), ("model", "b"),
        ]
        self.gi.remove_old_history()
        self.assertEqual(4, len(self.gi.history))

    def test_removes_oldest_pair_when_over_limit(self):
        """maxHistoryLength を超えたとき、最古の1往復（2エントリ）が削除されること。"""
        self.gi.history = [
            ("user", "1"), ("model", "a"),
            ("user", "2"), ("model", "b"),
            ("user", "3"), ("model", "c"),
        ]
        self.gi.remove_old_history()
        self.assertEqual(4, len(self.gi.history))
        # 最古の ("user","1"), ("model","a") が消えていること
        self.assertEqual("2", self.gi.history[0][1])

    def test_removes_one_pair_at_a_time(self):
        """1回の呼び出しで削除されるのは1往復（2エントリ）だけであること。"""
        self.gi.history = [
            ("user", "1"), ("model", "a"),
            ("user", "2"), ("model", "b"),
            ("user", "3"), ("model", "c"),
            ("user", "4"), ("model", "d"),
        ]
        self.gi.remove_old_history()
        self.assertEqual(6, len(self.gi.history))


class TestResetChatHistory(unittest.TestCase):
    """reset_chat_history メソッドのテスト。"""

    def setUp(self):
        self.gi = GenAIInteractions()

    def test_clears_interaction_id_and_history(self):
        """リセット後に interaction_id と history がクリアされること。"""
        self.gi.interaction_id = "some_id"
        self.gi.history = [("user", "hello"), ("model", "hi")]
        self.gi.reset_chat_history()
        self.assertIsNone(self.gi.interaction_id)
        self.assertEqual([], self.gi.history)


class TestLoadChatHistory(unittest.TestCase):
    """load_chat_history メソッドのテスト。"""

    def setUp(self):
        self.gi = GenAIInteractions()
        self._original_id_file = GenAIInteractions.FILENAME_INTERACTION_ID
        self._original_hist_file = GenAIInteractions.FILENAME_CHAT_HISTORY

    def tearDown(self):
        GenAIInteractions.FILENAME_INTERACTION_ID = self._original_id_file
        GenAIInteractions.FILENAME_CHAT_HISTORY = self._original_hist_file

    def _use_temp_files(self):
        tmp_dir = tempfile.mkdtemp()
        id_path = os.path.join(tmp_dir, "interaction_id.txt")
        hist_path = os.path.join(tmp_dir, "history.pkl")
        GenAIInteractions.FILENAME_INTERACTION_ID = id_path
        GenAIInteractions.FILENAME_CHAT_HISTORY = hist_path
        return id_path, hist_path

    def test_returns_false_when_no_files(self):
        """ファイルが存在しない場合、False を返すこと。"""
        id_path, _ = self._use_temp_files()
        # ファイルを作らずに呼び出す
        result = self.gi.load_chat_history()
        self.assertFalse(result)

    def test_loads_interaction_id(self):
        """interaction_id ファイルが存在する場合、ID が読み込まれて True を返すこと。"""
        id_path, _ = self._use_temp_files()
        with open(id_path, "w") as f:
            f.write("test_interaction_id\n")
        result = self.gi.load_chat_history()
        self.assertTrue(result)
        self.assertEqual("test_interaction_id", self.gi.interaction_id)


class TestGenerateText(unittest.IsolatedAsyncioTestCase):
    """generate_text メソッドのテスト（API 呼び出しをモック）。"""

    def setUp(self):
        self.gi = GenAIInteractions()
        # クライアントをモックに差し替えて実際のAPIを呼ばないようにする
        self.mock_client = MagicMock()

        # get_client メソッド自体をモック化して、
        # 内部で self.client = None されても常に mock_client を返すようにする
        self.gi.get_client = MagicMock(return_value=self.mock_client)
        self.gi.api_key_index = 0

        if "google" not in g.config:
            g.config["google"] = {}
        g.config["google"]["geminiApiKey"] = ["key_0", "key_1", "key_2"]
        g.config["google"]["modelName"] = "gemini-test-model"

    def _set_create_response(self, response):
        """client.aio.interactions.create の返り値を設定する。"""
        self.mock_client.aio.interactions.create = AsyncMock(return_value=response)

    def _set_create_side_effect(self, effects):
        """client.aio.interactions.create の side_effect を設定する。"""
        self.mock_client.aio.interactions.create = AsyncMock(side_effect=effects)

    async def test_success_returns_response_text(self):
        """正常時にレスポンステキストが返ること。"""
        self._set_create_response(make_interaction_mock("Hello!"))
        result = await self.gi.generate_text("Hi")
        self.assertEqual("Hello!", result)

    async def test_success_appends_to_history(self):
        """正常応答後に history にユーザー/モデルのペアが追加されること。"""
        self._set_create_response(make_interaction_mock("Hi there!"))
        await self.gi.generate_text("Hello")
        self.assertEqual([("user", "Hello"), ("model", "Hi there!")], self.gi.history)

    async def test_uses_interaction_id_when_present(self):
        """interaction_id がある場合、params に previous_interaction_id が設定されること。"""
        self.gi.interaction_id = "existing_id"
        self._set_create_response(make_interaction_mock("response"))
        await self.gi.generate_text("message")
        call_kwargs = self.mock_client.aio.interactions.create.call_args.kwargs
        self.assertEqual("existing_id", call_kwargs.get("previous_interaction_id"))

    async def test_uses_context_injection_when_no_interaction_id(self):
        """interaction_id がなく history がある場合、コンテキスト注入が行われること。"""
        self.gi.interaction_id = None
        self.gi.history = [("user", "I am Fuyuka."), ("model", "Hello Fuyuka!")]
        self._set_create_response(make_interaction_mock("response"))
        await self.gi.generate_text("next message")
        call_kwargs = self.mock_client.aio.interactions.create.call_args.kwargs
        self.assertNotIn("previous_interaction_id", call_kwargs)
        self.assertIn("[直前の会話の文脈]", call_kwargs.get("input", ""))

    async def test_429_switches_api_key_and_retries(self):
        """429 エラー時にAPIキーが切り替わり、次のキーでリトライして成功し、新しいIDが保存されること。"""
        self.gi.interaction_id = "old_id"
        self.gi.history = [("user", "prev"), ("model", "resp")]
        self._set_create_side_effect([
            make_api_error(429),
            make_interaction_mock("success after key switch", interaction_id="new_key_id"),
        ])
        result = await self.gi.generate_text("message")
        self.assertEqual("success after key switch", result)
        # 最終的にはリトライ成功時の新しいIDが入る
        self.assertEqual("new_key_id", self.gi.interaction_id)
        self.assertIn(("user", "prev"), self.gi.history)

    async def test_429_clears_interaction_id_but_keeps_history(self):
        """429 エラー後に interaction_id が引き継がれず（クリア扱いでリトライされ）、history は保持されること。"""
        self.gi.interaction_id = "some_id"
        self.gi.history = [("user", "A"), ("model", "B")]
        self._set_create_side_effect([
            make_api_error(429),
            make_interaction_mock("ok", interaction_id="new_id_after_429"),
        ])
        await self.gi.generate_text("hi")

        # 2回目の呼び出し（リトライ）の引数に、古い previous_interaction_id が渡されていないことを検証
        second_call_kwargs = self.mock_client.aio.interactions.create.call_args_list[1].kwargs
        self.assertNotIn("previous_interaction_id", second_call_kwargs)

        # 最終的には2回目の成功時のIDがセットされる
        self.assertEqual("new_id_after_429", self.gi.interaction_id)
        self.assertGreaterEqual(len(self.gi.history), 2)

    async def test_503_retries_and_eventually_succeeds(self):
        """503 エラーが続いた後に成功した場合、正しいレスポンスが返ること。"""
        self._set_create_side_effect([
            make_api_error(503),
            make_api_error(503),
            make_interaction_mock("recovered"),
        ])
        with patch("genai_interactions.asyncio.sleep", new_callable=AsyncMock):
            result = await self.gi.generate_text("message")
        self.assertEqual("recovered", result)

    async def test_503_returns_error_after_max_retries(self):
        """503 エラーが max_retries を超えた場合、エラーメッセージが返ること。"""
        # 最大5回のリトライ後は6回目でも 503 → エラー終了
        self._set_create_side_effect([make_api_error(503)] * 6)
        with patch("genai_interactions.asyncio.sleep", new_callable=AsyncMock):
            result = await self.gi.generate_text("message")
        self.assertEqual(g.STOP_CANDIDATE_MESSAGE, result)
