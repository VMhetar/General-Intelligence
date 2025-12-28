import os
import logging
import httpx
import asyncio
from mcp.server.fastmcp import FastMCP
from world_model import WorldModel

mcp = FastMCP('General-Intelligence')

url = "https://openrouter.ai/api/v1/chat/completions"

api_key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

prompt = f"""
You are an intlligent languge interpreter. 
Your task is to:
- See the world model.
- Understand the undergoing physics.
- Interpret the situations to generalize the understand and creating rules for transfer learning.
"""
@mcp.tool()
async def llm_call(prompt: str):
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        result = response.json()
        return result['choices'][0]['message']['content']

def summarize_dynamics(z, z_next, logvar, r, d):
    delta = (z_next - z).detach()

    summary = {
        "state_change": {
            "magnitude": float(delta.norm()),
            "dominant_dims": delta.abs().topk(3).indices.tolist()
        },
        "uncertainty": {
            "mean": float(logvar.mean()),
            "max": float(logvar.max())
        },
        "reward_trend": "positive" if r.item() > 0 else "negative",
        "termination_risk": float(d.item())
    }

    return summary

async def interpret_world_model(world_model: WorldModel, z, a):
    z_next, logvar, r, d = world_model(z, a)

    summary = summarize_dynamics(z, z_next, logvar, r, d)

    prompt = f"""
You are a language interpreter for a physical world model.
Here is a summarized transition:
{summary}

Your task:
1. Describe what kind of physical behavior this suggests.
2. Identify whether the dynamics appear stable, unstable, or uncertain.
3. Propose high-level rules that could transfer to similar environments.

Do NOT speculate beyond the given information.
"""

    interpretation = await llm_call(prompt)
    return interpretation
