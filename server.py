import numpy as np
import asyncio
import json
import numpy as np
from ollama_obj import ollamas

om = ollamas()


async def ollama_machine(socket):
    global om
    try:
        async for message in socket:
            msg = json.loads(message)
            if msg["TYPE"] == "new_char":
                om.char_prompt = msg["DATA"]
                # print(msg["DATA"])
            if msg["TYPE"] == "reload_rag":
                om.remake_VDB(msg["DATA"])
                print("RAG COMPLETE")
                await socket.send('{"TYPE" : "reload_end"}')
            if msg["TYPE"] == "msg":
                await socket.send(om.text_chat(msg["DATA"]))
            if msg["TYPE"] == "reset_history":
                await om.memory.aclear()

    except json.JSONDecodeError:
        pass
    except ws.exceptions.ConnectionClosed:
        pass
        # exit(0)
    except ws.exceptions.ConnectionClosedError:
        exit(1)


async def main():
    global om
    om.load_VDB()
    async with ws.serve(ollama_machine, "localhost", 55177):
        await asyncio.Future()


asyncio.run(main())
