import os
import discord
from discord import app_commands
from ollama_obj import ollamas

llm = ollamas()
TOKEN = str()
with open("../tokenfile.txt", "r") as tokenfile:
    TOKEN = tokenfile.read()


class MyClient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        
    async def setup_hook(self):
        await self.tree.sync()



client = MyClient()
def chunk_text(s, size=2000):
    return [s[i:i+size] for i in range(0, len(s), size)]


@client.tree.command(name="set_prompt", description="change system prompt")
@app_commands.describe(prompt="추가할 프롬프트", reset_and_rewrite="프롬프트 리셋 후 새로 입력")
async def set_prompt(interaction: discord.Interaction,
               prompt:str, reset_and_rewrite:bool):
    if reset_and_rewrite:
        llm.char_prompt = prompt
    else:
        llm.char_prompt = llm.char_prompt + prompt
    
    await interaction.response.send_message("Set Prompt!")
    for i in chunk_text(llm.char_prompt):
        await interaction.followup.send(f"{i}")
    


@client.tree.command(name="ping", description="Replies pong")
@app_commands.describe(msg="아무거나")
async def ping(interaction: discord.Interaction,
               msg:str):
    print(f"got msg: {msg}")
    # llm.char_prompt = msg
    await interaction.response.send_message(f"pong! GOT MSG: {msg}")
    res = llm.text_chat(msg)["DATA"]
    print(res)
    for i in chunk_text(res):
        await interaction.followup.send(f"{i}")
    # for i in range(count - 1):

@client.event
async def on_ready():
    await client.tree.sync()
    print("synced")

if __name__ == "__main__":
    llm.init_chain()
    client.run(TOKEN)
