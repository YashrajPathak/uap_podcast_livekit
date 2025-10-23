"""Mock Room Test Harness for LiveKit Agent

This module lets you run a lightweight test of `MyAgent` and `entrypoint` without
starting the LiveKit simulator. It monkeypatches `AgentSession` with a fake
implementation and provides a fake `JobContext` where `ctx.room` is a dummy object.

Run:
    python -m src.uap_podcast.livekit_mock_room

Expected output:
    - Logs showing the fake session starting
    - A single call to `agent.session.generate_reply()`
    - Clean shutdown
"""

import asyncio
import logging
from types import SimpleNamespace
from contextlib import asynccontextmanager
from unittest.mock import patch

from livekit_agent import entrypoint, MyAgent

logger = logging.getLogger("uap_podcast.livekit.mock")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FakeRoom:
    """Minimal stub for a LiveKit room object used by the agent session."""
    async def close(self):
        logger.info("FakeRoom: closed")


class FakeAgentSession:
    """Minimal stand-in for livekit.agents.AgentSession used by tests."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False

    async def start(self, agent: MyAgent, room: FakeRoom):
        logger.info("FakeAgentSession: start called with FakeRoom")
        self.started = True

        class _AgentSessionBinding:
            async def generate_reply(self_inner):
                logger.info("FakeAgentSession: generate_reply invoked (mock)")

        # Bind a minimal session object to the agent, mimicking real behavior
        agent.session = _AgentSessionBinding()
        await agent.on_enter()
        # Simulate short-lived session, then close
        await asyncio.sleep(0.1)
        await room.close()
        logger.info("FakeAgentSession: session finished")


class FakeJobContext:
    """Carries a fake room to mimic LiveKit JobContext."""
    def __init__(self):
        self.room = FakeRoom()


async def _run_mock():
    logger.info("Starting mock room test harness…")
    ctx = FakeJobContext()

    # Patch the AgentSession class used inside livekit_agent.entrypoint
    with patch("livekit_agent.AgentSession", FakeAgentSession):
        await entrypoint(ctx)

    logger.info("Mock room test harness completed successfully.")


def main():
    asyncio.run(_run_mock())


if __name__ == "__main__":
    main()
