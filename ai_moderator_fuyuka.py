import asyncio
import datetime
import json
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import global_value as g

g.app_name = "ai_moderator_fuyuka"
g.base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

from config_helper import readConfig
from genai_chat import GenAIChat
from text_helper import readText

print("前回の続きですか？(y/n) ", end="")
is_continue = input() == "y"

g.BASE_PROMPT = readText("prompts/base_prompt.txt")
g.ERROR_MESSAGE = readText("messages/error_message.txt")
g.STOP_CANDIDATE_MESSAGE = readText("messages/stop_candidate_message.txt")

g.config = readConfig()

g.storyteller = ""
g.story_buffer = ""

fuyuka_port = g.config["fuyukaApi"]["port"]

# ロガーの設定
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

genai_chat = GenAIChat()
if is_continue and genai_chat.load_chat_history():
    print("会話履歴を復元しました。")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def send_personal_json(self, json_data: dict[str, any], websocket: WebSocket):
        await websocket.send_json(json_data)

    async def broadcast_json(self, json_data: dict[str, any]):
        for connection in self.active_connections:
            await connection.send_json(json_data)


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
    isFirst: bool = False
    isFirstOnStream: bool = False
    noisy: bool = False
    additionalRequests: str = f"あなたの回答は{answerLength}文字以内にまとめてください"


class ChatResult(BaseModel):
    id: str
    request: ChatModel
    response: str


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
        "isFirst": False,
        "isFirstOnStream": False,
        "noisy": True,
        "additionalRequests": "You understand the flow of the story. reply OK.",
    }
    response_text = await genai_chat.send_message_by_json(json_data)
    g.story_buffer = ""
    return response_text


async def _flow_story(json_data: dict[str, any]) -> str:
    g.storyteller = json_data["displayName"]
    g.story_buffer += json_data["content"] + " "
    if len(g.story_buffer) <= 1000:
        return ""
    response_text = await flow_story_genai_chat()
    return response_text


app = FastAPI()


@app.get("/")
async def chat_test() -> str:
    return HTMLResponse(html)


@app.post("/chat/{id}")
async def chat_endpoint(id: str, chat: ChatModel) -> ChatResult:
    json_data = jsonable_encoder(chat)
    response_text = ""
    if "noisy" in json_data and json_data["noisy"]:
        response_text = await _flow_story(json_data)
    else:
        await flow_story_genai_chat()
        response_text = await genai_chat.send_message_by_json(json_data)

    response_json = {
        "id": id,
        "request": json_data,
        "response": response_text,
    }
    await manager.broadcast_json(response_json)
    return JSONResponse(response_json)


@app.websocket("/chat/{id}")
async def chat_ws(websocket: WebSocket, id: str) -> None:
    await manager.connect(websocket)
    try:
        while True:
            json_data = await websocket.receive_json()
            if "noisy" in json_data and json_data["noisy"]:
                # 例外: noisyの場合、flow_storyとしてバッファにためておく
                asyncio.create_task(_flow_story(json_data))
                continue

            await flow_story_genai_chat()
            response_text = await genai_chat.send_message_by_json(json_data)
            if not response_text:
                continue
            response_json = {
                "id": id,
                "request": json_data,
                "response": response_text,
            }
            await manager.broadcast_json(response_json)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{id} left the chat")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.error("Client disconnected")


@app.get("/reset_chat")
async def reset_chat() -> Result:
    g.story_buffer = ""
    genai_chat.reset_chat_history()
    return JSONResponse({"result": True})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=fuyuka_port)
