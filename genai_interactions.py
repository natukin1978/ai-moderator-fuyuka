import json
import logging
import os

from google import genai
from google.genai import errors, types
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
    Tool,
)

import global_value as g
from cache_helper import get_cache_filepath

logger = logging.getLogger(__name__)


class GenAIInteractions:
    # 履歴は ID (文字列) だけ保存すれば良くなる
    FILENAME_INTERACTION_ID = get_cache_filepath(f"{g.app_name}_interaction_id.txt")
    FILENAME_API_KEY_INDEX = get_cache_filepath(f"{g.app_name}_api_key_index.pkl")

    GENAI_SAFETY_SETTINGS = [
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    ]

    GOOGLE_SEARCH_TOOL = Tool(google_search=GoogleSearch())

    def __init__(self):
        self.last_error_code = None
        self.api_key_index = None
        self.client = None
        self.interaction_id = None # 保存・復元するID

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
            # aio (Async) クライアントを使用
            self.client = genai.Client(api_key=self.get_api_key(), http_options={'api_version': 'v1alpha'})
        return self.client

    # --- 履歴の保存・復元 ---
    def load_chat_history(self) -> bool:
        if not os.path.isfile(self.FILENAME_INTERACTION_ID):
            return False
        with open(self.FILENAME_INTERACTION_ID, "r") as f:
            self.interaction_id = f.read().strip()
            return True

    def save_chat_history(self) -> None:
        if not self.interaction_id:
            return
        with open(self.FILENAME_INTERACTION_ID, "w") as f:
            f.write(self.interaction_id)

    def reset_chat_history(self) -> None:
        self.interaction_id = None
        if os.path.isfile(self.FILENAME_INTERACTION_ID):
            os.remove(self.FILENAME_INTERACTION_ID)

    # --- メインロジック ---
    async def generate_text(self, message: str) -> str:
        while True:
            try:
                conf_g = g.config["google"]
                client = self.get_client()

                # Interactions API を使用した生成
                # previous_interaction_id を指定することで文脈を維持
                response = await client.aio.models.generate_content(
                    model=conf_g["modelName"],
                    contents=message,
                    config=GenerateContentConfig(
                        system_instruction=g.BASE_PROMPT,
                        safety_settings=self.GENAI_SAFETY_SETTINGS,
                        tools=[self.GOOGLE_SEARCH_TOOL],
                        # ここがキモ：前回のIDがあれば渡す
                        previous_interaction_id=self.interaction_id 
                    )
                )

                # 新しい interaction_id を保存（これが次のターンの previous になる）
                if response.interaction_id:
                    self.interaction_id = response.interaction_id
                    self.save_chat_history()

                response_text = response.text.rstrip() if response.text else ""
                logger.debug(f"Gemini Response: {response_text}")
                return response_text

            except errors.APIError as e:
                logger.error(f"API Error: {e}")
                if e.code == 429: # Quota Exhausted
                    # キーを切り替える
                    self.api_key_index = (self.get_api_key_index() + 1) % len(g.config["google"]["geminiApiKey"])
                    self.save_api_key_index(self.api_key_index)
                    self.client = None
                    # 注意: キーが変わると interaction_id は無効になるためリセット
                    self.interaction_id = None 
                    continue
                
                self.last_error_code = e.code
                return self.get_error_message(e.code)
            except Exception as e:
                logger.exception(f"Unexpected Error: {e}")
                return g.ERROR_MESSAGE

    async def send_message(self, message: str) -> str:
        return await self.generate_text(message)

    async def send_message_by_json(self, json_data: dict[str, any]) -> str:
        json_str = json.dumps(json_data, ensure_ascii=False, separators=(",", ":"))
        return await self.send_message(json_str)
