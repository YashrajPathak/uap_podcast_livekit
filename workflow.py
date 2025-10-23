"""Agent-based LangGraph workflow orchestrator."""

import uuid
import datetime
import json
import asyncio
from typing import Dict, Any, Optional, Literal, List
from pathlib import Path

from langgraph.graph import StateGraph, END

from utils.logging import setup_logger
from models.podcast import PodcastContext
from agents.nexus_agent.utils.state import PodcastState, NexusAgentState
from agents.reco_agent.utils.state import RecoAgentState
from agents.stat_agent.utils.state import StatAgentState
from agents.nexus_agent.utils.nodes import NexusNodes
from agents.reco_agent.utils.nodes import RecoNodes
from agents.stat_agent.utils.nodes import StatNodes


class AgentBasedOrchestrator:
    """LangGraph-based podcast orchestration using agent structure."""
    
    def __init__(self):
        """Initialize the Agent-based Orchestrator."""
        self.nexus_nodes = NexusNodes()
        self.reco_nodes = RecoNodes()
        self.stat_nodes = StatNodes()
        self.logger = setup_logger("uap_podcast")
        
        # Compiled graph cache
        self._compiled_graph = None
    
    def _build_compiled_graph(self):
        """Build and compile the LangGraph workflow."""
        builder = StateGraph(PodcastState)
        
        # Add all nodes
        builder.add_node("nexus_intro", self.nexus_nodes.nexus_intro_node)
        builder.add_node("reco_intro", self.reco_nodes.reco_intro_node)
        builder.add_node("stat_intro", self.stat_nodes.stat_intro_node)
        builder.add_node("nexus_topic_intro", self.nexus_nodes.nexus_topic_intro_node)
        builder.add_node("reco_turn", self.reco_nodes.reco_turn_node)
        builder.add_node("stat_turn", self.stat_nodes.stat_turn_node)
        builder.add_node("nexus_outro", self.nexus_nodes.nexus_outro_node)
        
        # Define the workflow
        builder.set_entry_point("nexus_intro")
        
        # Linear flow through introductions
        builder.add_edge("nexus_intro", "reco_intro")
        builder.add_edge("reco_intro", "stat_intro")
        builder.add_edge("stat_intro", "nexus_topic_intro")
        
        # Main conversation loop
        builder.add_edge("nexus_topic_intro", "reco_turn")
        
        # Conditional logic for conversation flow
        def should_continue(state: PodcastState) -> Literal["continue_conversation", "end_conversation"]:
            """Determine whether to continue or end the conversation - FIXED VERSION."""
            current_turn = float(state["current_turn"])
            max_turns = float(state["max_turns"])
            
            # Safety check: Force termination after reasonable limit
            if current_turn > max_turns + 2:  # Allow 2 extra turns as safety buffer
                self.logger.warning(f"⚠️ Force termination: {current_turn} > {max_turns} + safety buffer")
                return "end_conversation"
            
            # Normal termination condition with floating point tolerance
            should_end = current_turn >= max_turns - 0.1
            
            #self.logger.info(f"🔄 Turn check: {current_turn}/{max_turns}, should_end: {should_end}")
            
            return "end_conversation" if should_end else "continue_conversation"
        
        # Add conditional edges with proper routing dictionaries
        builder.add_conditional_edges("reco_turn", should_continue, {
            "continue_conversation": "stat_turn",
            "end_conversation": "nexus_outro"
        })
        builder.add_conditional_edges("stat_turn", should_continue, {
            "continue_conversation": "reco_turn", 
            "end_conversation": "nexus_outro"
        })
        
        # End workflow
        builder.add_edge("nexus_outro", END)
        
        return builder.compile()
    
    def get_compiled_graph(self):
        """Get or build the compiled graph."""
        if self._compiled_graph is None:
            self._compiled_graph = self._build_compiled_graph()
        return self._compiled_graph
    
    def _determine_conversation_flow(self, state: PodcastState) -> str:
        """Determine the next conversation node based on turns."""
        if not state.conversation_turns:
            return "nexus_intro"
        
        if state.conversation_turns < state.max_turns:
            # Alternate between reco and stat turns
            if state.conversation_turns % 1 == 0.5:  # X.5 turns (after reco)
                return "stat_turn"
            else:  # Whole number turns
                return "reco_turn"
        else:
            return "nexus_outro"
    
    def _validate_reco_analysis(self, analysis: str) -> str:
        """Validate and potentially enhance reco analysis."""
        if not analysis or len(analysis.strip()) < 50:
            return "Comprehensive recommendation analysis based on industry best practices and market insights."
        return analysis
    
    def _validate_stat_analysis(self, analysis: str) -> str:
        """Validate and potentially enhance stat analysis.""" 
        if not analysis or len(analysis.strip()) < 50:
            return "Operational metrics analysis"
        return analysis
    
    async def generate_podcast(
        self,
        topic: Optional[str] = None,
        max_turns: int = 6,
        file_choice: str = "both",
        session_id: Optional[str] = None,
        recursion_limit: int = 60
    ) -> Dict[str, Any]:
        """Generate a podcast using LangGraph workflow."""
        try:
            self.logger.info("Starting LangGraph-based podcast generation")
            
            # Initialize session ID
            if not session_id:
                session_id = f"session_{uuid.uuid4().hex[:8]}"
            
            # Setup context
            context = PodcastContext.load_from_files(file_choice)
            
            # Create initial state
            context_dict = {
                "content": context.content, 
                "metadata": context.metadata,
                "summary": context.content[:500] + "..." if len(context.content) > 500 else context.content
            }
            
            initial_state = PodcastState(
                messages=[],
                context=context_dict,
                topic=topic or "Healthcare Technology Innovation",
                max_turns=max_turns,
                conversation_turns=0,
                session_id=session_id,
                nexus_state=NexusAgentState(session_id=session_id),
                reco_state=RecoAgentState(session_id=session_id), 
                stat_state=StatAgentState(session_id=session_id),
                current_speaker="nexus",
                turn_number=0,
                episode_complete=False,
                audio_segments=[],
                script_content="",
                workflow_type="agent_based_langgraph",
                # Additional required fields for LangGraph
                conversation_history=[],
                current_turn=0.0,
                node_history=[],
                current_node="nexus_intro",
                script_lines=[],
                interrupted=False
            )
            
            # Get compiled graph
            graph = self.get_compiled_graph()
            
            # Execute the workflow
            config = {"recursion_limit": recursion_limit}
            
            self.logger.info(f"Executing LangGraph workflow with topic: {initial_state['topic']}")
            
            final_state = None
            async for state in graph.astream(initial_state, config=config):
                final_state = state
                # Log progress
                if 'current_speaker' in state:
                    self.logger.info(f"   📍 Current speaker: {state['current_speaker']}, Turn: {state['conversation_turns']}")
            
            if not final_state:
                raise Exception("Workflow execution failed - no final state received")
                
            # Extract final state (LangGraph returns dict with single key)
            if isinstance(final_state, dict):
                final_state = list(final_state.values())[0]
            
            self.logger.info("✅ LangGraph workflow execution completed")
            
            # Generate audio file
            audio_file = await self._finalize_audio(final_state)
            script_file = self._save_script(final_state)
            
            # Calculate duration
            duration = self._calculate_duration(final_state.get('audio_segments', []))
            
            result = {
                "session_id": session_id,
                "audio_file": audio_file,
                "script_file": script_file,
                "topic": final_state.get('topic', 'Unknown'),
                "turns": final_state.get('conversation_turns', 0),
                "duration_seconds": duration,
                "workflow_type": "agent_based_langgraph",
                "timestamp": datetime.datetime.now().isoformat(),
                "success": True
            }
            
            self.logger.info(f"🎉 Agent-based LangGraph generation complete! Audio: {audio_file}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Agent-based LangGraph generation failed: {str(e)}")
            raise e
    
    async def _finalize_audio(self, state: Dict[str, Any]) -> str:
        """Create final audio file from segments."""
        audio_segments = state.get('audio_segments', [])
        if not audio_segments:
            raise Exception("No audio segments generated")
        
        from .models.audio import AudioProcessor
        audio_processor = AudioProcessor()
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_audio_path = f"podcast_agent_langgraph_{timestamp}.wav"
        
        # Concatenate all audio segments
        await asyncio.to_thread(
            audio_processor.concatenate_audio_segments,
            audio_segments,
            final_audio_path
        )
        
        self.logger.info(f"🎵 Final audio created: {final_audio_path}")
        return final_audio_path
    
    def _save_script(self, state: Dict[str, Any]) -> str:
        """Save the script content to file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_file = f"podcast_script_agent_langgraph_{timestamp}.txt"
        
        script_content = state.get('script_content', '')
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        self.logger.info(f"📝 Script saved: {script_file}")
        return script_file
    
    def _calculate_duration(self, audio_segments: List[str]) -> float:
        """Calculate total duration from audio segments."""
        try:
            import wave
            total_duration = 0.0
            
            for segment_path in audio_segments:
                if Path(segment_path).exists():
                    with wave.open(segment_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        duration = frames / float(sample_rate)
                        total_duration += duration
            
            return total_duration
        except Exception as e:
            self.logger.warning(f"Could not calculate duration: {e}")
            return 0.0
