import datetime
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import global_value as g
from config_helper import readConfig
from genai_chat import GenAIChat
from text_helper import readText

g.BASE_PROMPT = readText("prompts/base_prompt.txt")
g.ERROR_MESSAGE = readText("messages/error_message.txt")
g.STOP_CANDIDATE_MESSAGE = readText("messages/stop_candidate_message.txt")

g.config = readConfig()


genai_chat = GenAIChat()
genai_chat.load_chat_history()

app = FastAPI()


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
    additionalRequests: str = f"あなたの回答は{answerLength}文字以内にまとめてください"


class ChatResult(BaseModel):
    id: str
    request: ChatModel
    response: str


class Result(BaseModel):
    result: bool = True


chat_template = json.dumps(jsonable_encoder(ChatModel()), indent=2, ensure_ascii=False)

html = """
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
"""
html += chat_template
html += """
            </textarea>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            const client_id = Date.now()
            document.querySelector("#ws-id").textContent = client_id;
            const ws = new WebSocket(`ws://localhost:8000/chat/${client_id}`);
            ws.onmessage = function(event) {
                const messages = document.getElementById('messages')
                const message = document.createElement('li')
                const json = JSON.parse(event.data)
                const content = document.createTextNode(json.id + ": " + json.response)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                const input = document.getElementById("messageText")
                ws.send(input.value)
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def chat_test() -> str:
    return HTMLResponse(html)


@app.get("/chat/{id}")
async def chat_endpoint(id: str, chat: ChatModel) -> ChatResult:
    json_data = jsonable_encoder(chat)
    response_text = genai_chat.send_message_by_json(json_data)
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
            response_text = genai_chat.send_message_by_json(json_data)
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
        print(f"Error: {e}")
    finally:
        print("Client disconnected")


@app.get("/reset_chat")
async def reset_chat() -> Result:
    genai_chat.reset_chat_history()
    return {"result": True}
