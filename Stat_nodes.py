"""Node implementations for Stat Agent in LangGraph workflow."""

import datetime
import asyncio
from typing import Dict, Any, List
from langgraph.graph.message import add_messages

from utils.config import Config
from utils.logging import default_logger
from models.podcast import PodcastEngine, ConversationDynamics
from models.audio import AudioProcessor
from agents.nexus_agent.utils.state import PodcastState


class StatNodes:
    """Node implementations for Stat agent operations."""
    
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
    
    # LangGraph Node Functions
    async def stat_intro_node(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Stat agent introduction."""
        line = Config.STAT_INTRO
        audio = await self._generate_tts(line, "STAT")
        self.logger.info("Stat intro generated.")
        
        # Update Stat agent state
        stat_state = state["stat_state"]
        if stat_state:
            stat_state.conversation_context.append({
                "speaker": "STAT",
                "text": line,
                "turn": 0
            })
        
        return {
            "messages": add_messages(state["messages"], [{"role": "system", "content": line}]),
            "audio_segments": state["audio_segments"] + [audio],
            "conversation_history": state["conversation_history"] + [{"speaker": "STAT", "text": line}],
            "script_lines": state["script_lines"] + [f"Agent Stat: {line}"],
            "current_speaker": "NEXUS",
            "node_history": state["node_history"] + [{"node": "stat_intro", "ts": datetime.datetime.now().isoformat()}],
            "current_node": "stat_intro",
            "stat_state": stat_state
        }
    
    async def stat_turn_node(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Stat agent conversation turn."""
        total_pairs = int(state["max_turns"])
        # Calculate current turn pair: Stat follows Reco, so same turn number
        current_pair = int(state["current_turn"]) + 1
        self.logger.info(f"Generating turn {current_pair}/{total_pairs}… (Stat)")
        
        # Generate response using PodcastEngine
        line = await self.engine.generate_agent_response(
            role="STAT",
            context=state["context"]["summary"],
            conversation_history=state["conversation_history"],
            turn_count=int(state["current_turn"])
        )
        
        # Apply conversation dynamics
        line = self.conversation_dynamics.add_conversation_dynamics(
            line, "STAT", "RECO", state["context"]["summary"], 
            int(state["current_turn"]), state["conversation_history"]
        )
        line = self.conversation_dynamics.strip_forbidden_words(line, "STAT")
        line = self._ensure_complete_response(line)
        
        audio = await self._generate_tts(line, "STAT")
        
        next_speaker = "RECO" if state["current_turn"] + 0.5 < state["max_turns"] else "NEXUS"
        
        # Update Stat agent state
        stat_state = state["stat_state"]
        if stat_state:
            stat_state.increment_turn()
            
            # Extract and track data validations
            if "valid" in line.lower() or "check" in line.lower():
                stat_state.add_validation(line[:100])  # First 100 chars as validation summary
            
            # Track data concerns
            if "concern" in line.lower() or "issue" in line.lower():
                stat_state.add_data_concern(line[:100])
        
        return {
            "messages": add_messages(state["messages"], [{"role": "system", "content": line}]),
            "audio_segments": state["audio_segments"] + [audio],
            "conversation_history": state["conversation_history"] + [{"speaker": "STAT", "text": line}],
            "script_lines": state["script_lines"] + [f"Agent Stat: {line}"],
            "current_speaker": next_speaker,
            "current_turn": state["current_turn"] + 0.5,
            "node_history": state["node_history"] + [{"node": "stat_turn", "ts": datetime.datetime.now().isoformat()}],
            "current_node": "stat_turn",
            "stat_state": stat_state
        }
