"""LiveKit + MCP integration for UAP Podcast.

This module wires a voice agent that runs fully locally using the LiveKit CLI simulator
and integrates Azure STT/TTS, Azure OpenAI LLM, and MCP tool calling. It is designed to
co-exist with the existing LangGraph-based workflow without altering the agents.

Usage (CLI simulate):
    python -m livekit.agents.cli simulate  # then this module provides the worker entrypoint

Programmatic:
    from src.uap_podcast.livekit_agent import run_cli, entrypoint
    run_cli()  # starts the worker; use LiveKit CLI simulate for the room
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

# LiveKit Agents & plugins
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, mcp
from livekit.plugins import openai, silero, azure
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import langchain as lk_langchain  # noqa: F401 (available for extensions)

# LangChain MCP client (tools exposure)
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401 (available for extensions)

# Optional: LangGraph prebuilt (kept for parity and future routing)
from langgraph.prebuilt import create_react_agent  # noqa: F401

# Optional Azure LLM wrapper if needed alongside livekit.plugins.openai
from langchain_openai import AzureChatOpenAI  # noqa: F401

logger = logging.getLogger("uap_podcast.livekit")

# Load env from current working directory (expecting .env next to this file when run from uap_podcast/)
load_dotenv()

# Orchestrator (LangGraph) import for optional live bridging
from workflow import AgentBasedOrchestrator


class MyAgent(Agent):
    """Voice-first agent that can interact with MCP tools via the LiveKit session."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are MCP Sentinel, a voice-based agent that interacts with the MCP server. "
                "You can retrieve data via the MCP server. The interface is voice-based: "
                "accept spoken user queries and respond with synthesized speech."
            ),
        )

    async def on_enter(self):
        # Kick off an initial reply (e.g., greeting)
        await self.session.generate_reply()


class PodcastVoiceAgent(Agent):
    """Bridges LiveKit session with the existing LangGraph orchestrator and agents.

    Note: For low-latency demo, we trigger a short intro sequence and stream lines
    back using session.generate_reply (as a TTS stand-in). This preserves existing
    architecture and avoids breaking changes.
    """

    def __init__(self, topic: str | None = None, max_turns: int = 1):
        super().__init__(
            instructions=(
                "You are the Podcast Voice Agent. Coordinate Nexus, Reco, and Stat to narrate a short, "
                "real-time sequence based on LangGraph. Keep the flow natural and concise."
            ),
        )
        self.topic = topic
        self.max_turns = max_turns
        self._orch: AgentBasedOrchestrator | None = None

    async def on_enter(self):
        # Build a minimal orchestrator run and speak resulting lines
        try:
            self._orch = AgentBasedOrchestrator()
            result = await self._orch.generate_podcast(
                topic=self.topic,
                max_turns=self.max_turns,
                file_choice="both",
                recursion_limit=40,
            )
            # Prefer script_file contents if available, otherwise script_content
            # For simplicity, speak the summary topic and turns info.
            summary = f"Topic: {result.get('topic','Unknown')}. Turns: {result.get('turns',0)}. Beginning the session."
            await self.session.generate_reply(instructions=summary)
        except Exception as e:
            logger.warning(f"PodcastVoiceAgent on_enter failed: {e}")
            await self.session.generate_reply(instructions="Starting live session. Let's begin with the introduction.")

async def entrypoint(ctx: JobContext):
    """LiveKit worker entrypoint used by the CLI simulator or an external worker process."""
    # Pull env vars (may be absent in mock mode)
    speech_region = os.getenv("AZURE_SPEECH_REGION") or os.getenv("SPEECH_REGION")
    speech_auth_token = os.getenv("AZURE_SPEECH_AUTH_TOKEN")
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")

    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
    mcp_auth = os.getenv("MCP_AUTH_TOKEN")

    is_mock = os.getenv("LIVEKIT_MOCK") == "1"

    if not (speech_region and (speech_auth_token or speech_key or speech_endpoint)):
        if not is_mock:
            logger.warning("Missing Azure Speech env vars; STT/TTS will be disabled.")
    if not (azure_endpoint and api_key):
        if not is_mock:
            logger.warning("Missing Azure OpenAI env vars; LLM will be disabled.")

    # Build plugins conditionally
    stt_plugin = None
    tts_plugin = None
    llm_plugin = None

    if not is_mock:
        # STT/TTS: allow any supported auth combination
        try:
            if speech_region and (speech_auth_token or speech_key or speech_endpoint):
                if speech_auth_token:
                    stt_plugin = azure.STT(speech_region=speech_region, speech_auth_token=speech_auth_token)
                    tts_plugin = azure.TTS(speech_region=speech_region, speech_auth_token=speech_auth_token)
                elif speech_key:
                    stt_plugin = azure.STT(speech_region=speech_region, speech_key=speech_key)
                    tts_plugin = azure.TTS(speech_region=speech_region, speech_key=speech_key)
                elif speech_endpoint:
                    stt_plugin = azure.STT(speech_endpoint=speech_endpoint)
                    tts_plugin = azure.TTS(speech_endpoint=speech_endpoint)
        except Exception as e:
            logger.warning(f"Azure Speech plugins disabled due to error: {e}")

        # LLM
        try:
            if azure_endpoint and api_key:
                llm_plugin = openai.LLM.with_azure(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                )
        except Exception as e:
            logger.warning(f"Azure OpenAI LLM disabled due to error: {e}")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        turn_detection=MultilingualModel(),
        mcp_servers=[
            mcp.MCPServerHTTP(
                url=mcp_url,
                headers={"Authorization": f"Bearer {mcp_auth}"} if mcp_auth else None,
                timeout=60,
                client_session_timeout_seconds=60,
            )
        ],
    )

    # Start the voice session in the provided room context (no extra room options)
    await session.start(
        agent=PodcastVoiceAgent(topic=None, max_turns=1),
        room=ctx.room,
    )

    # Send an explicit greeting like the sample
    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


def run_cli():
    """Start a worker process suitable for use with the LiveKit CLI simulate command."""
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


# Convenience alias for external runners
run_app = run_cli

if __name__ == "__main__":
    run_cli()
