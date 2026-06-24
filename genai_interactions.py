import json
import logging
import os

from google import genai
from google.genai import errors, types

import global_value as g
from cache_helper import get_cache_filepath

logger = logging.getLogger(__name__)


class GenAIInteractions:
    FILENAME_INTERACTION_ID = get_cache_filepath(f"{g.app_name}_interaction_id.txt")
    FILENAME_API_KEY_INDEX = get_cache_filepath(f"{g.app_name}_api_key_index.pkl")

    # GENAI_SAFETY_SETTINGS = [
    #     # ハラスメントは中程度を許容する
    #     SafetySetting(
    #         category=HarmCategory.HARM_CATEGORY_HARASSMENT,
    #         threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    #     ),
    #     # ヘイトスピーチは厳しく制限する
    #     SafetySetting(
    #         category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    #         threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    #     ),
    #     # セクシャルな内容を多少は許容する
    #     SafetySetting(
    #         category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    #         threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    #     ),
    #     # ゲーム向けなので、危険に分類されるコンテンツを許容できる
    #     SafetySetting(
    #         category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    #         threshold=HarmBlockThreshold.BLOCK_NONE,
    #     ),
    # ]

    GOOGLE_SEARCH_TOOL = [{"type": "google_search"}]

    def __init__(self):
        self.last_error_code = None
        self.api_key_index = None
        self.client = None
        self.interaction_id = None

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
        if os.path.isfile(self.FILENAME_INTERACTION_ID):
            try:
                os.remove(self.FILENAME_INTERACTION_ID)
            except Exception as e:
                logger.error(f"Failed to delete interaction ID file: {e}")

    def load_chat_history(self) -> bool:
        if not os.path.isfile(self.FILENAME_INTERACTION_ID):
            return False
        with open(self.FILENAME_INTERACTION_ID, "r") as f:
            self.interaction_id = f.read().strip()
            return True

    def save_chat_history(self, interaction_id: str) -> None:
        self.interaction_id = interaction_id
        with open(self.FILENAME_INTERACTION_ID, "w") as f:
            f.write(interaction_id)

    async def generate_text(self, message: str) -> str:
        while True:
            try:
                conf_g = g.config["google"]
                client = self.get_client()

                params = {
                    "model": conf_g["modelName"],
                    "system_instruction": g.BASE_PROMPT,
                    "input": message,
                    "tools": self.GOOGLE_SEARCH_TOOL,
                    "generation_config": {
                        "thinking_summaries": "none",
                    },
                }
                if self.interaction_id:
                    params["previous_interaction_id"] = self.interaction_id

                interaction = await client.aio.interactions.create(**params)

                if interaction.id:
                    self.save_chat_history(interaction.id)

                # レスポンスからテキストを抽出
                if interaction.output_text:
                    response_text = interaction.output_text.rstrip()
                    logger.debug(f"Response: {response_text}")
                    return response_text

                return ""

            except errors.APIError as e:
                logger.error(e)
                match e.code:
                    case 429:
                        # トークン枯渇
                        logger.warning("Token exhausted, switching API key...")
                        self.last_error_code = None
                        self.get_api_key_index(1)
                        self.client = None
                        self.interaction_id = None  # キーが変わるとIDも無効になる
                        if os.path.isfile(self.FILENAME_INTERACTION_ID):
                            try:
                                os.remove(self.FILENAME_INTERACTION_ID)
                            except Exception as e:
                                logger.error(f"Failed to delete interaction ID file on API key switch: {e}")
                        continue
                    case _:
                        self.last_error_code = e.code
                        pass
                return self.get_error_message(e.code)
            except Exception as e:
                logger.exception(f"Unexpected Error: {e}")
                return g.ERROR_MESSAGE

    async def send_message(self, message: str) -> str:
        return await self.generate_text(message)

    async def send_message_by_json(self, json_data: dict[str, any]) -> str:
        json_str = json.dumps(json_data, ensure_ascii=False, separators=(",", ":"))
        return await self.send_message(json_str)
