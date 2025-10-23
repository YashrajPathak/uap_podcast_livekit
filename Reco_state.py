"""State management for Reco Agent."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class RecoAgentState:
    """Reco agent specific state."""
    session_id: str
    current_turn: int = 0
    last_recommendation: str = ""
    conversation_context: List[Dict[str, str]] = field(default_factory=list)
    metrics_discussed: List[str] = field(default_factory=list)
    recommendations_made: List[str] = field(default_factory=list)
    last_opener: Optional[str] = None
    
    def increment_turn(self):
        """Increment the turn counter."""
        self.current_turn += 1
    
    def add_recommendation(self, recommendation: str):
        """Add a recommendation to the history."""
        self.recommendations_made.append(recommendation)
        self.last_recommendation = recommendation
    
    def add_discussed_metric(self, metric: str):
        """Add a metric to the discussed list."""
        if metric not in self.metrics_discussed:
            self.metrics_discussed.append(metric)
    
    def update_opener(self, opener: str):
        """Update the last opener used."""
        self.last_opener = opener
    
    def add_conversation_context(self, speaker: str, text: str):
        """Add conversation context."""
        self.conversation_context.append({
            "speaker": speaker,
            "text": text,
            "turn": self.current_turn
        })
        
        # Keep only last 10 entries to manage memory
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    
    def get_last_stat_response(self) -> Optional[str]:
        """Get the last response from Stat agent."""
        for entry in reversed(self.conversation_context):
            if entry["speaker"] == "STAT":
                return entry["text"]
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the Reco agent."""
        return {
            "session_id": self.session_id,
            "current_turn": self.current_turn,
            "recommendations_count": len(self.recommendations_made),
            "metrics_discussed_count": len(self.metrics_discussed),
            "last_recommendation": self.last_recommendation,
            "context_entries": len(self.conversation_context)
        }


# Legacy alias for backward compatibility
RecoState = RecoAgentState
