"""State management for Nexus Agent."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from typing_extensions import TypedDict


@dataclass
class NexusAgentState:
    """Nexus agent specific state."""
    session_id: str
    topic: str = ""
    is_active: bool = False
    intro_completed: bool = False
    outro_completed: bool = False
    context_summary: str = ""
    generated_lines: List[str] = field(default_factory=list)
    
    def update_topic(self, topic: str):
        """Update the podcast topic."""
        self.topic = topic
    
    def mark_intro_complete(self):
        """Mark the introduction as completed."""
        self.intro_completed = True
    
    def mark_outro_complete(self):
        """Mark the outro as completed."""
        self.outro_completed = True
        self.is_active = False
    
    def add_generated_line(self, line: str):
        """Add a generated line to the history."""
        self.generated_lines.append(line)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status summary."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "is_active": self.is_active,
            "intro_completed": self.intro_completed,
            "outro_completed": self.outro_completed,
            "total_lines": len(self.generated_lines)
        }


class PodcastState(TypedDict):
    """Enhanced state structure for podcast generation with agent-specific states."""
    messages: List[Dict[str, Any]]
    current_speaker: str
    topic: str
    context: Dict[str, Any]
    interrupted: bool
    audio_segments: List[str]
    conversation_history: List[Dict[str, str]]
    current_turn: float
    max_turns: int
    session_id: str
    node_history: List[Dict[str, Any]]
    current_node: str
    script_lines: List[str]
    
    # Agent-specific states
    nexus_state: Optional['NexusAgentState']
    reco_state: Optional[Any]  # Will be imported from reco agent
    stat_state: Optional[Any]  # Will be imported from stat agent


# Legacy alias for backward compatibility
NexusState = NexusAgentState
