"""Node implementations for Reco Agent in LangGraph workflow."""

import datetime
import asyncio
from typing import Dict, Any, List
from langgraph.graph.message import add_messages

from utils.config import Config
from utils.logging import default_logger
from models.podcast import PodcastEngine, ConversationDynamics
from models.audio import AudioProcessor
from agents.nexus_agent.utils.Nexus_state import PodcastState


class RecoNodes:
    """Node implementations for Reco agent operations."""
    
    def __init__(self):
        """Initialize with required services."""
        self.engine = PodcastEngine()
        self.audio_processor = AudioProcessor()
        self.conversation_dynamics = ConversationDynamics()
        self.logger = default_logger
    
    async def _generate_tts(self, text: str, role: str) -> str:
        """Generate TTS audio for text with specified role voice."""
        ssml = self.audio_processor.text_to_ssml(text, role)
        audio_path = await asyncio.to_thread(
            self.audio_processor.synthesize_speech, 
            ssml
        )
        return audio_path
    
    def _ensure_complete_response(self, text: str) -> str:
        """Ensure response is complete and properly formatted."""
        import re
        t = re.sub(r'[`*_#>]+', ' ', (text or "")).strip()
        t = re.sub(r'\s{2,}', ' ', t)
        if t and t[-1] not in {'.', '!', '?'}:
            t += '.'
        return t
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract specific recommendations from text."""
        import re
        
        # Look for common recommendation patterns
        patterns = [
            r'recommend[ing]?\s+([^.]+)',
            r'should\s+([^.]+)',
            r'suggest[ing]?\s+([^.]+)',
            r'propose[d]?\s+([^.]+)',
            r'use\s+([^.]+\s+(?:average|chart|analysis|method))'
        ]
        
        recommendations = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            recommendations.extend(matches)
        
        return [rec.strip() for rec in recommendations if len(rec.strip()) > 5]
    
    # LangGraph Node Functions
    async def reco_intro_node(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Reco agent introduction."""
        line = Config.RECO_INTRO
        audio = await self._generate_tts(line, "RECO")
        self.logger.info("Reco intro generated.")
        
        # Update Reco agent state
        reco_state = state["reco_state"]
        if reco_state:
            reco_state.conversation_context.append({
                "speaker": "RECO",
                "text": line,
                "turn": 0
            })
        
        return {
            "messages": add_messages(state["messages"], [{"role": "system", "content": line}]),
            "audio_segments": state["audio_segments"] + [audio],
            "conversation_history": state["conversation_history"] + [{"speaker": "RECO", "text": line}],
            "script_lines": state["script_lines"] + [f"Agent Reco: {line}"],
            "current_speaker": "STAT",
            "node_history": state["node_history"] + [{"node": "reco_intro", "ts": datetime.datetime.now().isoformat()}],
            "current_node": "reco_intro",
            "reco_state": reco_state
        }
    
    async def reco_turn_node(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Reco agent conversation turn."""
        total_pairs = int(state["max_turns"])
        current_pair = int(state["current_turn"]) + 1
        self.logger.info(f"Generating turn {current_pair}/{total_pairs}… (Reco)")
        
        # Generate response using PodcastEngine
        line = await self.engine.generate_agent_response(
            role="RECO",
            context=state["context"]["summary"],
            conversation_history=state["conversation_history"],
            turn_count=int(state["current_turn"])
        )
        
        # Apply conversation dynamics
        line = self.conversation_dynamics.add_conversation_dynamics(
            line, "RECO", "STAT", state["context"]["summary"], 
            int(state["current_turn"]), state["conversation_history"]
        )
        line = self.conversation_dynamics.strip_forbidden_words(line, "RECO")
        line = self._ensure_complete_response(line)
        
        audio = await self._generate_tts(line, "RECO")
        
        # Update Reco agent state
        reco_state = state["reco_state"]
        if reco_state:
            reco_state.increment_turn()
            
            # Extract and track recommendations
            recommendations = self._extract_recommendations(line)
            for rec in recommendations:
                reco_state.add_recommendation(rec)
            
            # Add conversation context
            reco_state.add_conversation_context("RECO", line)
        
        return {
            "messages": add_messages(state["messages"], [{"role": "system", "content": line}]),
            "audio_segments": state["audio_segments"] + [audio],
            "conversation_history": state["conversation_history"] + [{"speaker": "RECO", "text": line}],
            "script_lines": state["script_lines"] + [f"Agent Reco: {line}"],
            "current_speaker": "STAT",
            "current_turn": state["current_turn"] + 0.5,  # FIXED: Ensure proper increment
            "node_history": state["node_history"] + [{"node": "reco_turn", "ts": datetime.datetime.now().isoformat()}],
            "current_node": "reco_turn",
            "reco_state": reco_state
        }
