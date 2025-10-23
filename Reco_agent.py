"""Reco Agent - Metrics recommendation specialist."""

import asyncio
from typing import Dict, Any, Optional, List

from uap_podcast.utils.config import Config
from uap_podcast.utils.logging import default_logger
from uap_podcast.models.podcast import PodcastEngine
from .utils.state import RecoState
from .utils.nodes import RecoNodes
from ..nexus_agent.utils.state import PodcastState


class RecoAgent:
    """
    Reco Agent - Metrics recommendation specialist.
    
    Responsible for providing actionable recommendations on metrics,
    monitoring methods, and operational improvements.
    """
    
    def __init__(self, podcast_engine: PodcastEngine):
        """Initialize Reco agent with podcast engine."""
        self.engine = podcast_engine
        self.nodes = RecoNodes()
        self.state: Optional[RecoState] = None
        
        default_logger.info("Reco Agent initialized")
    
    def initialize_session(self, session_id: str) -> RecoState:
        """Initialize a new podcast session."""
        self.state = RecoState(session_id=session_id)
        
        default_logger.info(f"Reco session initialized: {session_id}")
        return self.state
    
    async def generate_introduction(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Reco introduction."""
        if not self.state:
            raise RuntimeError("Reco agent not initialized")
        
        result = await self.nodes.reco_intro_node(state)
        
        # Update internal state
        self.state.add_conversation_context("RECO", Config.RECO_INTRO)
        
        default_logger.info("Reco introduction generated")
        return result
    
    async def generate_turn_response(self, state: PodcastState) -> Dict[str, Any]:
        """Generate conversation turn response."""
        if not self.state:
            raise RuntimeError("Reco agent not initialized")
        
        # Update turn counter
        self.state.increment_turn()
        
        # Generate the response
        result = await self.nodes.reco_turn_node(state)
        
        # Extract and analyze the response
        generated_text = result.get("conversation_history", [])[-1].get("text", "")
        
        # Update internal state with generated content
        if generated_text:
            self.state.add_recommendation(generated_text)
        
        # Update conversation context
        self.state.add_conversation_context("RECO", generated_text)
        
        default_logger.info(f"Reco turn {self.state.current_turn} generated")
        return result
    
    def analyze_conversation_performance(self) -> Dict[str, Any]:
        """Analyze the agent's performance in the conversation."""
        if not self.state:
            return {"status": "not_initialized"}
        
        return {
            "turns_completed": self.state.current_turn,
            "total_recommendations": len(self.state.recommendations_made),
            "metrics_discussed": len(self.state.metrics_discussed),
            "recommendation_summary": self._format_recommendation_summary(),
            "conversation_quality": self._assess_conversation_quality()
        }
    
    def _format_recommendation_summary(self) -> str:
        """Format a summary of recommendations made."""
        if not self.state.recommendations_made:
            return "No recommendations made"
        
        return f"{len(self.state.recommendations_made)} recommendations provided"
    
    def _assess_conversation_quality(self) -> Dict[str, Any]:
        """Assess the quality of conversation contributions."""
        if not self.state.recommendations_made:
            return {"score": 0.0, "issues": ["No recommendations made"]}
        
        # Simple quality assessment based on available data
        return {
            "score": 0.8,  # Default quality score
            "valid_recommendations": len(self.state.recommendations_made),
            "total_recommendations": len(self.state.recommendations_made),
            "issues": []
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for Reco agent."""
        return Config.SYSTEM_RECO
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the Reco agent."""
        return {
            "name": "Agent Reco",
            "role": "Metrics Recommendation Specialist",
            "voice": Config.VOICE_RECO,
            "description": "Senior consultant specializing in actionable metrics recommendations",
            "capabilities": [
                "Metrics strategy and selection",
                "Monitoring method recommendations",
                "Operational improvement suggestions",
                "Risk mitigation strategies",
                "Performance target setting"
            ],
            "focus_areas": [
                "Smoothing and trend analysis",
                "Control charting",
                "Cohort analysis",
                "Operational levers",
                "Quality gates and validation"
            ]
        }
    
    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """Get current session status."""
        return self.state.get_status() if self.state else None
    
    async def cleanup(self):
        """Clean up agent resources."""
        if self.state:
            default_logger.info(f"Reco session {self.state.session_id} completed with {self.state.current_turn} turns")
        
        default_logger.info("Reco agent cleaned up")
