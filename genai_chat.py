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
from genai_history import GenAIHistory

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
        self.reset_chat_history()

    def get_chat_history(self) -> GenAIHistory:
        if not self.chat_history:
            conf_g = g.config["google"]
            self.chat_history = GenAIHistory(
                conf_g["geminiApiKey"], conf_g["modelName"]
            )
        return self.chat_history

    def get_genaiModel(self):
        if not self.genaiModel:
            chat_history = self.get_chat_history()
            genai.configure(api_key=chat_history.api_key)
            self.genaiModel = genai.GenerativeModel(
                model_name=chat_history.model_name,
                safety_settings=self.GENAI_SAFETY_SETTINGS,
                system_instruction=g.BASE_PROMPT,
            )
        return self.genaiModel

    def get_chat(self) -> genai.ChatSession:
        if not self.genaiChat:
            self.genaiChat = self.get_genaiModel().start_chat(
                history=self.get_chat_history().data
            )
        return self.genaiChat

    def reset_chat_history(self) -> None:
        self.genaiModel = None
        self.chat_history = None
        self.genaiChat = None

    def load_chat_history(self) -> bool:
        if not os.path.isfile(self.FILENAME_CHAT_HISTORY):
            return False
        with open(self.FILENAME_CHAT_HISTORY, "rb") as f:
            self.reset_chat_history()
            self.chat_history = pickle.load(f)
            return True

    def save_chat_history(self) -> None:
        with open(self.FILENAME_CHAT_HISTORY, "wb") as f:
            self.chat_history.data = self.get_chat().history
            pickle.dump(self.chat_history, f)

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
