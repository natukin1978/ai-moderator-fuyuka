import asyncio
import json
import logging
import os
import pickle
import random

from google import genai

import global_value as g
from cache_helper import get_cache_filepath

logger = logging.getLogger(__name__)


class GenAIInteractions:
    FILENAME_INTERACTION_ID = get_cache_filepath(f"{g.app_name}_interaction_id.txt")
    FILENAME_API_KEY_INDEX = get_cache_filepath(f"{g.app_name}_api_key_index.pkl")
    FILENAME_CHAT_HISTORY = get_cache_filepath(f"{g.app_name}_gen_ai_interactions_history.pkl")

    GOOGLE_SEARCH_TOOL = [{"type": "google_search"}]

    def __init__(self):
        self.last_error_code = None
        self.api_key_index = None
        self.client = None
        self.interaction_id = None
        self.history: list[tuple[str, str]] = []  # (role, text) のリスト

    @staticmethod
    def get_error_message(error_code: int) -> str:
        match error_code:
            case 429:
                # トークン枯渇
                return g.RESOURCE_EXHAUSTED_MESSAGE
            case _:
                return g.STOP_CANDIDATE_MESSAGE

    @classmethod
    def load_api_key_index(cls) -> int:
        i = 0
        if os.path.isfile(cls.FILENAME_API_KEY_INDEX):
            with open(cls.FILENAME_API_KEY_INDEX, "r") as f:
                i = json.load(f)
        return i

    @classmethod
    def save_api_key_index(cls, index: int) -> int:
        with open(cls.FILENAME_API_KEY_INDEX, "w") as f:
            json.dump(index, f)

    def get_api_key_index(self, inc_value: int = 0) -> int:
        if self.api_key_index is None:
            self.api_key_index = self.load_api_key_index()

        i = self.api_key_index
        i += inc_value
        conf_g = g.config["google"]
        # if 0 <= i and i < len(conf_g["geminiApiKey"]):
        if 0 > i or i >= len(conf_g["geminiApiKey"]):
            i = 0

        if i != self.api_key_index:
            self.save_api_key_index(i)

        self.api_key_index = i
        return self.api_key_index

    def get_api_key(self) -> str:
        i = self.get_api_key_index()
        conf_g = g.config["google"]
        return conf_g["geminiApiKey"][i]

    def get_client(self) -> genai.Client:
        if self.client is None:
            self.client = genai.Client(api_key=self.get_api_key())

        return self.client

    def reset_chat_history(self) -> None:
        self.last_error_code = None
        self.interaction_id = None
        self.history = []
        for filepath in [self.FILENAME_INTERACTION_ID, self.FILENAME_CHAT_HISTORY]:
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.error(f"Failed to delete file {filepath}: {e}")

    def load_chat_history(self) -> bool:
        loaded = False
        if os.path.isfile(self.FILENAME_INTERACTION_ID):
            with open(self.FILENAME_INTERACTION_ID, "r") as f:
                self.interaction_id = f.read().strip()
                loaded = True
        if os.path.isfile(self.FILENAME_CHAT_HISTORY):
            with open(self.FILENAME_CHAT_HISTORY, "rb") as f:
                self.history = pickle.load(f)
        return loaded

    def save_chat_history(self, interaction_id: str) -> None:
        self.interaction_id = interaction_id
        with open(self.FILENAME_INTERACTION_ID, "w") as f:
            f.write(interaction_id)
        with open(self.FILENAME_CHAT_HISTORY, "wb") as f:
            pickle.dump(self.history, f)

    def remove_old_history(self) -> None:
        """maxHistoryLength を超えた古い履歴エントリを削除する。"""
        conf_g = g.config["google"]
        max_len = conf_g["maxHistoryLength"]
        if len(self.history) > max_len:
            # 古い履歴から削除（1往復 = user+model の2エントリ）
            del self.history[0:2]

    def build_context_input(self, message: str) -> str:
        """interaction_id がない場合にローカル履歴をコンテキストとして埋め込んだ入力を生成する。"""
        if not self.history:
            return message
        lines = ["[直前の会話の文脈]"]
        for role, text in self.history:
            label = "ユーザー" if role == "user" else "アシスタント"
            lines.append(f"{label}: {text}")
        lines.append("")
        lines.append("上記のやり取りを踏まえて、以下の新しいメッセージに応答してください。")
        lines.append(message)
        return "\n".join(lines)

    def _extract_status_code(self, e: Exception) -> int | None:
        """例外オブジェクトから HTTP ステータスコードを抽出するヘルパーメソッド"""
        # オブジェクトの属性（code / status_code）を最優先でチェック
        if getattr(e, "code", None) is not None:
            return e.code
        if getattr(e, "status_code", None) is not None:
            return e.status_code

        # 例外クラスの「名前（文字列）」から判定
        # クラスを直接参照しないため、AttributeError を完全に回避できます
        type_name = type(e).__name__
        if "RateLimitError" in type_name:
            return 429

        # エラーメッセージの文字列から判定（最後のセーフティネット）
        err_str = str(e).lower()
        if any(x in err_str for x in ["429", "too_many_requests", "quota"]):
            return 429
        if any(x in err_str for x in ["503", "service unavailable"]):
            return 503

        return None

    async def generate_text(self, message: str) -> str:
        retry_count = 0  # 503用のリトライカウンタ
        max_retries = 5  # 最大リトライ回数

        conf_g = g.config["google"]
        key_switch_count = 0
        max_key_switches = len(conf_g["geminiApiKey"]) # キーの総数

        while True:
            try:
                client = self.get_client()

                params = {
                    "model": conf_g["modelName"],
                    "system_instruction": g.BASE_PROMPT,
                    "tools": self.GOOGLE_SEARCH_TOOL,
                    "generation_config": {
                        "thinking_summaries": "none",
                    },
                }
                if self.interaction_id:
                    # 通常フロー: interaction_id で過去の会話を引き継ぐ
                    params["input"] = message
                    params["previous_interaction_id"] = self.interaction_id
                else:
                    # APIキー切り替え後の初回など: ローカル履歴をコンテキストとして埋め込む
                    params["input"] = self.build_context_input(message)

                interaction = await client.aio.interactions.create(**params)

                if interaction.id:
                    self.save_chat_history(interaction.id)

                # レスポンスからテキストを抽出
                if interaction.output_text:
                    response_text = interaction.output_text.rstrip()
                    logger.debug(f"Response: {response_text}")
                    # ローカル履歴に追記
                    self.history.append(("user", message))
                    self.history.append(("model", response_text))
                    self.remove_old_history()
                    return response_text

                return ""

            except Exception as e:
                # エラーオブジェクトやメッセージからステータスコードを確実に特定する
                status_code = self._extract_status_code(e)

                # ------------------------------------------------------------------
                # ステータスコードに応じた分岐処理
                # ------------------------------------------------------------------
                if status_code == 404:
                    # 【404: セッション消失（IDをクリアして同じキーで即時リトライ）】
                    logger.warning("Session (interaction_id) not found on server. Clearing ID and retrying with local history...")
                    self.interaction_id = None  # IDを初期化して、次回ループで build_context_input を通す

                    if os.path.isfile(self.FILENAME_INTERACTION_ID):
                        try:
                            os.remove(self.FILENAME_INTERACTION_ID)
                        except Exception as ex:
                            logger.error(f"Failed to delete file: {ex}")
                    continue  # 同じキーのままループの先頭に戻って再試行

                elif status_code == 429:
                    # 【429: トークン・クォータ枯渇（キー切り替え）】
                    key_switch_count += 1
                    if key_switch_count >= max_key_switches:
                        logger.error("All API keys are exhausted.")
                        self.last_error_code = 429
                        return self.get_error_message(429)

                    logger.warning("Token/Quota exhausted, switching API key...")
                    self.last_error_code = None
                    self.get_api_key_index(1)
                    self.client = None
                    self.interaction_id = None
                    retry_count = 0

                    if os.path.isfile(self.FILENAME_INTERACTION_ID):
                        try:
                            os.remove(self.FILENAME_INTERACTION_ID)
                        except Exception as ex:
                            logger.error(f"Failed to delete file: {ex}")
                    continue

                elif status_code == 503:
                    # 【503: 高需要・サーバー負荷（指数バックオフリトライ）】
                    if retry_count < max_retries:
                        delay = (2 ** retry_count) + random.uniform(0, 1)
                        logger.warning(f"503 Service Unavailable. Retrying in {delay:.2f}s...")
                        await asyncio.sleep(delay)
                        retry_count += 1
                        continue
                    else:
                        logger.error("Max retries reached for 503.")
                        self.last_error_code = 503
                        return self.get_error_message(503)

                else:
                    # 【その他のエラー】
                    if status_code is not None:
                        self.last_error_code = status_code
                        logger.error(f"API Error ({status_code}): {e}")
                        return self.get_error_message(self.last_error_code)
                    else:
                        logger.exception(f"Unexpected Error: {e}")
                        return g.ERROR_MESSAGE

    async def send_message(self, message: str) -> str:
        return await self.generate_text(message)

    async def send_message_by_json(self, json_data: dict[str, any]) -> str:
        json_str = json.dumps(json_data, ensure_ascii=False, separators=(",", ":"))
        return await self.send_message(json_str)
