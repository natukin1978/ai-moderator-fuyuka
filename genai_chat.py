import json
import logging
import os
import pickle

from google import genai
from google.genai import chats, errors, types
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


class GenAIChat:
    FILENAME_CHAT_HISTORY = get_cache_filepath(f"{g.app_name}_gen_ai_chat_history.pkl")
    FILENAME_API_KEY_INDEX = get_cache_filepath(f"{g.app_name}_api_key_index.pkl")

    GENAI_SAFETY_SETTINGS = [
        # ハラスメントは中程度を許容する
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        ),
        # ヘイトスピーチは厳しく制限する
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        # セクシャルな内容を多少は許容する
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        # ゲーム向けなので、危険に分類されるコンテンツを許容できる
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    GOOGLE_SEARCH_TOOL = Tool(google_search=GoogleSearch())

    def __init__(self):
        self.is_abort = False
        self.last_error_code = None
        self.client = genai.Client(api_key=self.get_api_key())
        self.chat_history = None
        self.genaiChat = None

    @staticmethod
    def get_error_message(error_code: int) -> str:
        match error_code:
            case 429:
                # トークン枯渇
                return g.RESOURCE_EXHAUSTED_MESSAGE
            case _:
                return g.STOP_CANDIDATE_MESSAGE

    @classmethod
    def get_api_key_index(cls, inc_value: int = 0) -> int:
        i = 0
        if os.path.isfile(cls.FILENAME_API_KEY_INDEX):
            with open(cls.FILENAME_API_KEY_INDEX, "r") as f:
                i = json.load(f)

        i += inc_value
        conf_g = g.config["google"]
        # if 0 <= i and i < len(conf_g["geminiApiKey"]):
        if 0 > i or i >= len(conf_g["geminiApiKey"]):
            i = 0

        return i

    @staticmethod
    def get_api_key() -> str:
        i = GenAIChat.get_api_key_index()
        conf_g = g.config["google"]
        return conf_g["geminiApiKey"][i]

    @classmethod
    def rotate_api_key(cls) -> None:
        i = cls.get_api_key_index(1)
        with open(cls.FILENAME_API_KEY_INDEX, "w") as f:
            json.dump(i, f)

    def get_chat(self) -> chats.AsyncChat:
        if not self.genaiChat:
            conf_g = g.config["google"]
            self.genaiChat = self.client.aio.chats.create(
                model=conf_g["modelName"],
                config=GenerateContentConfig(
                    system_instruction=g.BASE_PROMPT,
                    safety_settings=self.GENAI_SAFETY_SETTINGS,
                    tools=[self.GOOGLE_SEARCH_TOOL],
                ),
                history=self.chat_history,
            )
        return self.genaiChat

    def reset_chat_history(self) -> None:
        self.is_abort = False
        self.last_error_code = None
        self.chat_history = None
        self.genaiChat = None

    def load_chat_history(self) -> bool:
        if not os.path.isfile(self.FILENAME_CHAT_HISTORY):
            return False
        with open(self.FILENAME_CHAT_HISTORY, "rb") as f:
            self.chat_history = pickle.load(f)
            self.genaiChat = None
            return True

    def save_chat_history(self) -> None:
        with open(self.FILENAME_CHAT_HISTORY, "wb") as f:
            pickle.dump(self.get_chat()._curated_history, f)

    def remove_old_history(self) -> None:
        curated_history = self.get_chat()._curated_history
        conf_g = g.config["google"]
        if len(curated_history) > conf_g["maxHistoryLength"]:
            del curated_history[0:2] # del index 0,1

    async def send_message(self, message: str) -> str:
        if self.is_abort and self.last_error_code:
            return GenAIChat.get_error_message(self.last_error_code)
        try:
            logger.debug(message)
            chat_session = self.get_chat()
            response = await chat_session.send_message(message)
            response_text = response.text
            if response_text:
                response_text = response_text.rstrip()
            else:
                response_text = ""
            logger.debug(response_text)

            self.remove_old_history()
            self.save_chat_history()
            return response_text
        except errors.APIError as e:
            logger.error(e)
            self.last_error_code = e.code
            match e.code:
                case 429:
                    # トークン枯渇
                    self.is_abort = True
                case _:
                    pass
            return GenAIChat.get_error_message(self.last_error_code)
        except IndexError as e:
            logger.error(e)
            return ""
        except Exception as e:
            logger.error(e)
            return g.ERROR_MESSAGE

    async def send_message_by_json(self, json_data: dict[str, any]) -> str:
        json_str = json.dumps(json_data, ensure_ascii=False, separators=(",", ":"))
        return await self.send_message(json_str)
