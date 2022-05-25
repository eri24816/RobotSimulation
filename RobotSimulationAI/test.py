import asyncio
import websockets
import json

async def hello():
    async with websockets.connect("ws://localhost:8765") as websocket:
        await websocket.send("Gwr")
        await websocket.recv()

asyncio.run(hello())