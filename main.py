from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

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

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <h2>Your ID: <span id="ws-id"></span></h2>
        <form action="" onsubmit="sendMessage(event)">
            <textarea id="messageText">
{
  "dateTime": null,
  "id": "tester",
  "displayName": "テスター",
  "nickname": null,
  "content": "",
  "isFirst": false,
  "isFirstOnStream": false,
  "additionalRequests": null
}
            </textarea>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var client_id = Date.now()
            document.querySelector("#ws-id").textContent = client_id;
            var ws = new WebSocket(`ws://localhost:8000/chat/${client_id}`);
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
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


@app.get("/reset_chat")
async def reset_chat():
    genai_chat.reset_chat_history()
    return {"result": True}


@app.websocket("/chat/{id}")
async def chat_endpoint(websocket: WebSocket, id: str):
    await manager.connect(websocket)
    try:
        while True:
            json_data = await websocket.receive_json()
            response_text = genai_chat.send_message_by_json(json_data)
            if not response_text:
                continue
            response_json = {
                "id": id,
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
