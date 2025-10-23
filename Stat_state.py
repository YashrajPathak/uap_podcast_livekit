"""State management for Stat Agent."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class StatAgentState:
    """State container for Stat agent operations."""
    
    session_id: str
    current_turn: int = 0
    last_validation: str = ""
    conversation_context: List[Dict[str, str]] = field(default_factory=list)
    data_concerns_raised: List[str] = field(default_factory=list)
    validations_performed: List[str] = field(default_factory=list)
    statistical_checks_suggested: List[str] = field(default_factory=list)
    last_opener: Optional[str] = None
    
    def increment_turn(self):
        """Increment the turn counter."""
        self.current_turn += 1
    
    def add_validation(self, validation: str):
        """Add a validation to the history."""
        self.validations_performed.append(validation)
        self.last_validation = validation
    
    def add_data_concern(self, concern: str):
        """Add a data concern to the list."""
        if concern not in self.data_concerns_raised:
            self.data_concerns_raised.append(concern)
    
    def add_statistical_check(self, check: str):
        """Add a statistical check to the list."""
        self.statistical_checks_suggested.append(check)
    
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
    
    def get_last_reco_response(self) -> Optional[str]:
        """Get the last response from Reco agent."""
        for entry in reversed(self.conversation_context):
            if entry["speaker"] == "RECO":
                return entry["text"]
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the Stat agent."""
        return {
            "session_id": self.session_id,
            "current_turn": self.current_turn,
            "validations_count": len(self.validations_performed),
            "concerns_raised": len(self.data_concerns_raised),
            "statistical_checks": len(self.statistical_checks_suggested),
            "last_validation": self.last_validation,
            "context_entries": len(self.conversation_context)
        }


# Legacy alias for backward compatibility
StatState = StatAgentState
