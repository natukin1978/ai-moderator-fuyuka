import asyncio
import copy
import datetime
import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import global_value as g
from config_helper import read_config
from input_helper import input_with_timeout
from logging_setup import setup_app_logging

is_testing = os.environ.get("APP_TESTING") == "True"

g.app_name = "ai_moderator_fuyuka"
g.base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

if not is_testing:
    res = input_with_timeout("前回の続きですか？(y/n) [10秒以内に未入力なら 'n']: ", timeout=10)
    is_continue = (res == "y")
else:
    is_continue = False

g.config = read_config()

# ロガーの設定
setup_app_logging(g.config["logLevel"], log_file_path=f"{g.app_name}.log")
logger = logging.getLogger(__name__)

from dict_helper import remove_keys_by_value

# from genai_chat import GenAIChat
from genai_interactions import GenAIInteractions
from ng_words_helper import read_ng_words
from text_cleaner import clean_and_extract_alt
from text_helper import read_text

g.BASE_PROMPT = read_text("prompts/base_prompt.txt")
g.ADDITIONAL_REQUESTS_PROMPT = read_text("prompts/additional_requests_prompt.txt")
g.ERROR_MESSAGE = read_text("messages/error_message.txt")
g.STOP_CANDIDATE_MESSAGE = read_text("messages/stop_candidate_message.txt")
g.RESOURCE_EXHAUSTED_MESSAGE = read_text("messages/resource_exhausted_message.txt")

g.storyteller = ""
g.story_buffer = ""

fuyuka_port = g.config["fuyukaApi"]["port"]

# genai_chat = GenAIChat()
genai_chat = GenAIInteractions()
if is_continue and genai_chat.load_chat_history():
    print("会話履歴を復元しました。")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        # すでに削除されている場合の ValueError を防ぐ
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        # 送信失敗した接続を特定するためにコピーを作成してループ
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except Exception:
                # 送信失敗した接続はここで除外
                self.disconnect(connection)

    async def send_personal_json(self, json_data: dict[str, any], websocket: WebSocket):
        await websocket.send_json(json_data)

    async def broadcast_json(self, json_data: dict[str, any]):
        # 送信失敗した接続を特定するためにコピーを作成してループ
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(json_data)
            except Exception:
                # 送信失敗した接続はここで除外
                self.disconnect(connection)


manager = ConnectionManager()

localtime = datetime.datetime.now()
localtime_iso_8601 = localtime.isoformat()
answerLength = 30


class ChatModel(BaseModel):
    dateTime: str = localtime_iso_8601
    id: str = "master"
    displayName: str = "マスター"
    nickname: str = "ご主人様"
    content: str = "おはようございます。今日もよろしくお願いします。"
    needsResponse: bool = False
    noisy: bool = False
    additionalRequests: list[str] = [f"あなたの回答は{answerLength}文字以内にまとめてください"]


class ChatResult(BaseModel):
    id: str
    request: ChatModel
    response: str
    errorCode: int


class Result(BaseModel):
    result: bool = True


chat_template = json.dumps(jsonable_encoder(ChatModel()), indent=2, ensure_ascii=False)

html = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>Fuyuka Chat Test</title>
    </head>
    <body>
        <h1>Fuyuka Chat Test</h1>
        <h2>Your ID: <span id="ws-id"></span></h2>
        <form action="" onsubmit="sendMessage(event)">
            <textarea id="messageText" rows="16" cols="96">
{chat_template}
            </textarea>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            const fuyuka_port = {fuyuka_port}
            const client_id = Date.now()
            document.querySelector("#ws-id").textContent = client_id;
            const chat_endpoint = `ws://localhost:${{fuyuka_port}}/chat/${{client_id}}`
            const ws = new WebSocket(chat_endpoint);
            ws.onmessage = function(event) {{
                const messages = document.getElementById("messages")
                const message = document.createElement("li")
                const json = JSON.parse(event.data)
                if (!json.response) return
                const content = document.createTextNode(`${{json.id}}: ${{json.response}}`)
                message.appendChild(content)
                messages.prepend(message)
            }};
            function sendMessage(event) {{
                const input = document.getElementById("messageText")
                ws.send(input.value)
                event.preventDefault()
            }}
        </script>
    </body>
</html>
"""


def remove_newlines(value: str) -> str:
    return re.sub(r"[\r\n]", " ", value)

def update_viewerStatus(json_data: dict[str, any]):
    # popで値を取り出しつつ、辞書から安全に削除する（キーがなければFalseになる）
    is_first = json_data.pop("isFirst", False)
    is_first_on_stream = json_data.pop("isFirstOnStream", False)

    if is_first:
        viewerStatus = "newViewer"
    elif is_first_on_stream:
        viewerStatus = "streamFirst"
    else:
        viewerStatus = "regular"

    json_data["viewerStatus"] = viewerStatus

def append_additional_request(
    json_data: dict[str, any], value: str
) -> None:
    ars = json_data.get("additionalRequests", [])
    ars.append(value)
    json_data["additionalRequests"] = ars

def clean_and_extract_alt_by_json(json_data: dict[str, any]) -> None:
    json_data["content"] = clean_and_extract_alt(json_data["content"])

async def flow_story_genai_chat() -> str:
    if not g.story_buffer:
        return

    localtime = datetime.datetime.now()
    localtime_iso_8601 = localtime.isoformat()
    json_data = {
        "dateTime": localtime_iso_8601,
        "id": None,
        "displayName": g.storyteller,
        "content": g.story_buffer.rstrip(),
        "needsResponse": False,
        "noisy": True,
        "additionalRequests": ["Get a general idea of the flow of the conversation."],
    }
    response_text = await send_message_genai_chat(json_data)
    g.story_buffer = ""
    return remove_newlines(response_text)


async def _flow_story(json_data: dict[str, any]) -> str:
    g.storyteller = json_data["displayName"]
    g.story_buffer += json_data["content"] + " "
    if len(g.story_buffer) <= 1000:
        return ""
    response_text = await flow_story_genai_chat()
    return remove_newlines(response_text)


async def send_message_genai_chat(json_data: dict[str, any]) -> str:
    ng_words = read_ng_words()
    pattern = "|".join(ng_words)
    json_data_send = copy.deepcopy(json_data)
    update_viewerStatus(json_data_send)
    remove_keys_by_value(json_data_send, ["noisy"], False)
    while True:
        response_text = await genai_chat.send_message_by_json(json_data_send)
        if not response_text:
            return response_text

        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            matched_word = match.group()
            logger.warning(response_text)
            # 指摘文に具体的なキーワードを埋め込む
            content = (
                f"{json_data['dateTime']}の出力ですが`{matched_word}`という文章を含めずやり直してください。"
            )
            logger.warning(content)
            json_data_send["content"] = content
        else:
            return remove_newlines(response_text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    caption = "電脳娘フユカ(AIモデレーター Fuyuka API)"
    # startup
    logger.info(caption + "スタートしました。", extra={'force': True})
    yield
    # shutdown
    logger.info(caption + "終了しました。", extra={'force': True})


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def chat_test() -> str:
    return HTMLResponse(html)


@app.post("/chat/{id}")
async def chat_endpoint(id: str, chat: ChatModel) -> ChatResult:
    json_data = jsonable_encoder(chat)
    clean_and_extract_alt_by_json(json_data)

    if json_data.get("noisy", False):
        # 例外: noisyの場合、flow_storyとしてバッファにためておく
        await _flow_story(json_data)
        return None

    response_json = {
        "id": id,
        "request": json_data,
    }
    await manager.broadcast_json(response_json)

    await flow_story_genai_chat()
    append_additional_request(json_data, g.ADDITIONAL_REQUESTS_PROMPT)
    response_text = await send_message_genai_chat(json_data)

    response_json["response"] = response_text
    response_json["errorCode"] = genai_chat.last_error_code
    await manager.broadcast_json(response_json)
    return JSONResponse(response_json)


@app.websocket("/chat/{id}")
async def chat_ws(websocket: WebSocket, id: str) -> None:
    await manager.connect(websocket)
    try:
        while True:
            json_data = await websocket.receive_json()
            clean_and_extract_alt_by_json(json_data)
            if json_data.get("noisy", False):
                # 例外: noisyの場合、flow_storyとしてバッファにためておく
                asyncio.create_task(_flow_story(json_data))
                continue

            response_json = {
                "id": id,
                "request": json_data,
            }
            await manager.broadcast_json(response_json)

            await flow_story_genai_chat()
            append_additional_request(json_data, g.ADDITIONAL_REQUESTS_PROMPT)
            response_text = await send_message_genai_chat(json_data)
            if not response_text:
                continue

            response_json["response"] = response_text
            response_json["errorCode"] = genai_chat.last_error_code
            await manager.broadcast_json(response_json)
    except WebSocketDisconnect:
        logger.info(f"Client #{id} disconnected normally")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # 正常終了でも異常終了でも必ずリストから削除
        manager.disconnect(websocket)
        logger.info(f"Cleanup for Client #{id} completed")


@app.get("/reset_chat")
async def reset_chat() -> Result:
    g.story_buffer = ""
    genai_chat.reset_chat_history()
    return JSONResponse({"result": True})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=fuyuka_port)
