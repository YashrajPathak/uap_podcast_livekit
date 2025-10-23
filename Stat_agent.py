"""Stat Agent - Data integrity and statistical validation specialist."""

import asyncio
from typing import Dict, Any, Optional, List

from ...utils.config import Config
from ...utils.logging import default_logger
from ...models.podcast import PodcastEngine
from .utils.state import StatState
from .utils.nodes import StatNodes
from ..nexus_agent.utils.state import PodcastState


class StatAgent:
    """
    Stat Agent - Data integrity and statistical validation specialist.
    
    Responsible for validating assumptions, challenging leaps in logic,
    and grounding decisions in measurement quality and statistical integrity.
    """
    
    def __init__(self, podcast_engine: PodcastEngine):
        """Initialize Stat agent with podcast engine."""
        self.engine = podcast_engine
        self.nodes = StatNodes()
        self.state: Optional[StatState] = None
        
        default_logger.info("Stat Agent initialized")
    
    def initialize_session(self, session_id: str) -> StatState:
        """Initialize a new podcast session."""
        self.state = StatState(session_id=session_id)
        
        default_logger.info(f"Stat session initialized: {session_id}")
        return self.state
    
    async def generate_introduction(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Stat introduction."""
        if not self.state:
            raise RuntimeError("Stat agent not initialized")
        
        # Use the nodes implementation for consistency
        result = await self.nodes.stat_intro_node(state)
        
        # Update internal state (nodes already handles stat_state)
        self.state.add_conversation_context("STAT", Config.STAT_INTRO)
        
        default_logger.info("Stat introduction generated")
        return result
    
    async def generate_turn_response(self, state: PodcastState) -> Dict[str, Any]:
        """Generate conversation turn response."""
        if not self.state:
            raise RuntimeError("Stat agent not initialized")
        
        # Update turn counter
        self.state.increment_turn()
        
        # Use the nodes implementation for consistency
        result = await self.nodes.stat_turn_node(state)
        
        # Extract the generated response for internal tracking
        generated_text = result.get("conversation_history", [])[-1].get("text", "")
        if generated_text:
            self.state.add_conversation_context("STAT", generated_text)
        
        default_logger.info(f"Stat turn {self.state.current_turn} generated")
        return result
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for Stat agent."""
        return Config.SYSTEM_STAT
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the Stat agent."""
        return {
            "name": "Agent Stat", 
            "role": "Data Integrity & Statistical Validation Specialist",
            "voice": Config.VOICE_STAT,
            "description": "Senior statistical expert focused on data quality and measurement integrity",
            "capabilities": [
                "Statistical validation and testing",
                "Data quality assessment", 
                "Trend analysis and decomposition",
                "Risk assessment and mitigation",
                "Measurement methodology review"
            ],
            "focus_areas": [
                "Stationarity and seasonality checks",
                "Control charts and process monitoring", 
                "Cohort analysis and segmentation",
                "Anomaly detection and outlier handling",
                "Data validation and integrity audits"
            ]
        }
    
    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """Get current session status."""
        return self.state.get_status() if self.state else None
    
    async def cleanup(self):
        """Clean up agent resources."""
        if self.state:
            default_logger.info(f"Stat session {self.state.session_id} completed with {self.state.current_turn} turns")
        
        default_logger.info("Stat agent cleaned up")
