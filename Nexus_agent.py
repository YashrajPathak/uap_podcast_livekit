"""Nexus Agent - Host agent for podcast orchestration."""

import asyncio
from typing import Dict, Any, Optional

from uap_podcast.utils.config import Config
from uap_podcast.utils.logging import default_logger
from uap_podcast.models.podcast import PodcastEngine
from .utils.state import NexusState, PodcastState
from .utils.nodes import NexusNodes


class NexusAgent:
    """
    Nexus Agent - Host agent responsible for podcast orchestration.
    
    Handles introductions, topic setting, transitions, and conclusions
    in the podcast generation workflow.
    """
    
    def __init__(self, podcast_engine: PodcastEngine):
        """Initialize Nexus agent with podcast engine."""
        self.engine = podcast_engine
        self.nodes = NexusNodes()
        self.state: Optional[NexusState] = None
        
        default_logger.info("Nexus Agent initialized")
    
    def initialize_session(self, session_id: str, topic: str = "") -> NexusState:
        """Initialize a new podcast session."""
        self.state = NexusState(
            session_id=session_id,
            topic=topic,
            is_active=True
        )
        
        default_logger.info(f"Nexus session initialized: {session_id}")
        return self.state
    
    async def generate_introduction(self, state: PodcastState) -> Dict[str, Any]:
        """Generate complete podcast introduction including topic setup."""
        if not self.state:
            raise RuntimeError("Nexus agent not initialized")
        
        # Update state with topic if not set
        if not self.state.topic:
            # Use context to infer topic
            context_text = state["context"].get("summary", "")
            if context_text:
                self.state.update_topic("Data Metrics Discussion")
        
        # Generate both general intro and topic intro in one call
        result = await self.nodes.nexus_intro_node(state)
        self.state.mark_intro_complete()
        
        # Track generated content
        generated_line = result.get("conversation_history", [])[-1].get("text", "")
        self.state.add_generated_line(generated_line)
        
        default_logger.info("Nexus introduction with topic generated")
        return result
    
    async def generate_conclusion(self, state: PodcastState) -> Dict[str, Any]:
        """Generate podcast conclusion."""
        if not self.state:
            raise RuntimeError("Nexus agent not initialized")
        
        # Use the proper nexus_outro_node which handles state correctly
        result = await self.nodes.nexus_outro_node(state)
        
        # State is already updated in nexus_outro_node, so we don't need to duplicate it
        default_logger.info("Nexus conclusion generated")
        return result
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for Nexus agent."""
        return Config.SYSTEM_NEXUS
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the Nexus agent."""
        return {
            "name": "Agent Nexus",
            "role": "Host and Orchestrator",
            "voice": Config.VOICE_NEXUS,
            "description": "Warm, concise host responsible for introductions, transitions, and conclusions",
            "capabilities": [
                "Podcast introduction",
                "Topic setting and context introduction", 
                "Smooth transitions between agents",
                "Comprehensive conclusions and summaries"
            ]
        }
    
    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """Get current session status."""
        return self.state.get_status() if self.state else None
    
    async def cleanup(self):
        """Clean up agent resources."""
        if self.state:
            self.state.is_active = False
        
        default_logger.info("Nexus agent cleaned up")
