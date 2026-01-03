import os
import httpx
from typing import Dict, Any
from world_model import WorldModel

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY not set")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


async def llm_call(system_prompt: str, user_prompt: str) -> str:
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers=HEADERS,
            json=data
        )
        response.raise_for_status()
        result = response.json()

    return result["choices"][0]["message"]["content"]


def summarize_dynamics(z, z_next, logvar, r, d) -> Dict[str, Any]:
    delta = (z_next - z).detach()
    k = min(3, delta.numel())

    return {
        "state_change": {
            "magnitude": float(delta.norm()),
            "dominant_dimensions": delta.abs().topk(k).indices.tolist()
        },
        "uncertainty": {
            "mean": float(logvar.mean()),
            "max": float(logvar.max())
        },
        "reward_trend": "positive" if r.item() > 0 else "negative",
        "termination_risk": float(d.item())
    }


async def interpret_world_model(world_model: WorldModel, z, action) -> str:
    z_next, logvar, reward, done = world_model(z, action)
    summary = summarize_dynamics(z, z_next, logvar, reward, done)

    system_prompt = """You are a language interpreter for a physical world model.

Rules:
- You do NOT predict physics
- You do NOT invent hidden variables
- You ONLY interpret the given summary
- If information is insufficient, say so"""

    user_prompt = f"""Here is a summarized state transition:

{summary}

Tasks:
1. Describe what kind of physical behavior this suggests
2. Judge whether the system appears stable, unstable, or uncertain
3. Propose high-level rules that could transfer to similar environments

Do NOT speculate beyond the information given."""

    return await llm_call(system_prompt, user_prompt)
