"""Node implementations for Nexus Agent in LangGraph workflow."""

import datetime
import asyncio
from typing import Dict, Any, List
from langgraph.graph.message import add_messages

from utils.config import Config
from utils.logging import default_logger
from models.podcast import PodcastEngine
from models.audio import AudioProcessor
from .Nexus_state import PodcastState


class NexusNodes:
    """Node implementations for Nexus agent operations."""
    
    def __init__(self):
        """Initialize with required services."""
        self.engine = PodcastEngine()
        self.audio_processor = AudioProcessor()
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
    
    
    def generate_topic_intro(self, context: str, topic: str) -> str:
        """Generate topic introduction for Nexus."""
        return f"Welcome to our discussion on {topic}. Based on the current context: {context[:100]}..."
    
    # LangGraph Node Functions
    async def nexus_intro_node(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Nexus agent introduction."""
        line = Config.NEXUS_INTRO
        audio = await self._generate_tts(line, "NEXUS")
        self.logger.info("Nexus intro generated.")
        
        # Update Nexus agent state
        nexus_state = state["nexus_state"]
        if nexus_state:
            nexus_state.mark_intro_complete()
            nexus_state.add_generated_line(line)
        
        return {
            "messages": add_messages(state["messages"], [{"role": "system", "content": line}]),
            "audio_segments": state["audio_segments"] + [audio],
            "conversation_history": state["conversation_history"] + [{"speaker": "NEXUS", "text": line}],
            "script_lines": state["script_lines"] + [f"Agent Nexus: {line}"],
            "current_speaker": "RECO",
            "node_history": state["node_history"] + [{"node": "nexus_intro", "ts": datetime.datetime.now().isoformat()}],
            "current_node": "nexus_intro",
            "nexus_state": nexus_state
        }
    
    async def nexus_topic_intro_node(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Nexus topic introduction."""
        self.logger.info("Generating Nexus topic introduction...")
        
        topic_line = await self.engine.generate_nexus_topic_intro(
            state["context"]["summary"]
        )
        topic_line = self._ensure_complete_response(topic_line)
        audio = await self._generate_tts(topic_line, "NEXUS")
        
        # Update Nexus agent state
        nexus_state = state["nexus_state"]
        if nexus_state:
            nexus_state.add_generated_line(topic_line)
            nexus_state.update_topic(state["topic"])
        
        return {
            "messages": add_messages(state["messages"], [{"role": "system", "content": topic_line}]),
            "audio_segments": state["audio_segments"] + [audio],
            "conversation_history": state["conversation_history"] + [{"speaker": "NEXUS", "text": topic_line}],
            "script_lines": state["script_lines"] + [f"Agent Nexus: {topic_line}"],
            "current_speaker": "RECO",
            "current_turn": 0.0,
            "node_history": state["node_history"] + [{"node": "nexus_topic_intro", "ts": datetime.datetime.now().isoformat()}],
            "current_node": "nexus_topic_intro",
            "nexus_state": nexus_state
        }
    
    async def nexus_outro_node(self, state: PodcastState) -> Dict[str, Any]:
        """Generate Nexus agent outro."""
        line = Config.NEXUS_OUTRO
        audio = await self._generate_tts(line, "NEXUS")
        self.logger.info("Nexus outro generated.")
        
        # Update Nexus agent state
        nexus_state = state["nexus_state"]
        if nexus_state:
            nexus_state.mark_outro_complete()
            nexus_state.add_generated_line(line)
        
        return {
            "messages": add_messages(state["messages"], [{"role": "system", "content": line}]),
            "audio_segments": state["audio_segments"] + [audio],
            "conversation_history": state["conversation_history"] + [{"speaker": "NEXUS", "text": line}],
            "script_lines": state["script_lines"] + [f"Agent Nexus: {line}"],
            "current_speaker": "END",
            "node_history": state["node_history"] + [{"node": "nexus_outro", "ts": datetime.datetime.now().isoformat()}],
            "current_node": "nexus_outro",
            "nexus_state": nexus_state
        }
    
    def _ensure_complete_response(self, text: str) -> str:
        """Ensure response is a complete sentence without artificial truncation."""
        import re
        t = re.sub(r'[`*_#>]+', ' ', (text or "")).strip()
        t = re.sub(r'\s{2,}', ' ', t)
        if t and t[-1] not in {'.', '!', '?'}:
            t += '.'
        return t
