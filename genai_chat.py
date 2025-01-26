import json
import logging
import os
import pickle

import google.generativeai as genai
from google.generativeai.types import (
    HarmBlockThreshold,
    HarmCategory,
    StopCandidateException,
)

import global_value as g
from cache_helper import get_cache_filepath

logger = logging.getLogger(__name__)


class GenAIChat:
    FILENAME_CHAT_HISTORY = get_cache_filepath(f"{g.app_name}_gen_ai_chat_history.pkl")

    GENAI_SAFETY_SETTINGS = {
        # ハラスメントは中程度を許容する
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        # ヘイトスピーチは厳しく制限する
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        # セクシャルな内容を多少は許容する
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        # ゲーム向けなので、危険に分類されるコンテンツを許容できる
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    def __init__(self):
        conf_g = g.config["google"]
        genai.configure(api_key=conf_g["geminiApiKey"])
        self.genaiModel = genai.GenerativeModel(
            model_name=conf_g["modelName"],
            safety_settings=self.GENAI_SAFETY_SETTINGS,
            system_instruction=g.BASE_PROMPT,
        )
        self.chat_history = []
        self.genaiChat = None

    def get_chat(self) -> genai.ChatSession:
        if not self.genaiChat:
            self.genaiChat = self.genaiModel.start_chat(history=self.chat_history)
        return self.genaiChat

    def reset_chat_history(self) -> None:
        self.chat_history = []
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
            pickle.dump(self.get_chat().history, f)

    def send_message(self, message: str) -> str:
        try:
            logger.debug(message)
            response = self.get_chat().send_message(message)
            response_text = response.text.rstrip()
            logger.debug(response_text)
            self.save_chat_history()
            return response_text
        except StopCandidateException as e:
            logger.error(e)
            return g.STOP_CANDIDATE_MESSAGE
        except IndexError as e:
            logger.error(e)
            return ""
        except Exception as e:
            logger.error(e)
            return g.ERROR_MESSAGE

    def send_message_by_json(self, json_data: dict[str, any]) -> str:
        json_str = json.dumps(json_data, ensure_ascii=False)
        return self.send_message(json_str)
