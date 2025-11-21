import asyncio
import os
import discord
from discord import app_commands

from Query import hybrid_query

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
Qengine = hybrid_query()

def chunk_text(s, size=2000):
    return [s[i : i + size] for i in range(0, len(s), size)]


@client.tree.command(name="ask_pdje_codebase", description="Ask to PDJE Codebase")
@app_commands.describe(
    question="Some question About PDJE or PDJE Wrapper"
)
async def ask_pdje_codebase(
    interaction: discord.Interaction, question: str
):
    await interaction.response.send_message(f"Got question about '{question}'. LLM is scouring the code base to generate an answer. Please wait a few minutes!")
    try:
        response = await asyncio.to_thread(Qengine.query, question)
    except Exception as e:
        await interaction.followup.send(
            "Ollama runner crashed. Please retry later. If this repeats, mention an admin."
        )
        print(e)
    for i in chunk_text(str(response)):
        await interaction.followup.send(i)


@client.event
async def on_ready():
    await client.tree.sync()
    print("synced")


if __name__ == "__main__":
    client.run(TOKEN)
