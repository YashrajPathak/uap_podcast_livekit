"""
Real-time LangGraph State Monitor
Integrates with the existing workflow to provide live state visualization.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import threading
import asyncio


class StateMonitor:
    """Real-time state monitor for LangGraph workflows."""
    
    def __init__(self, output_file: str = "langgraph_state.json"):
        """Initialize the state monitor."""
        self.output_file = Path(output_file)
        self.state_history = []
        self.current_state = None
        self.execution_metadata = {
            "start_time": None,
            "end_time": None,
            "total_duration": 0,
            "node_count": 0,
            "error_count": 0
        }
        self.callbacks = []
        
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback function to be called on state updates."""
        self.callbacks.append(callback)
    
    def start_execution(self, session_id: str, initial_state: Dict[str, Any]):
        """Start monitoring a new execution."""
        self.execution_metadata["start_time"] = datetime.now().isoformat()
        self.execution_metadata["session_id"] = session_id
        self.state_history = []
        self.current_state = initial_state
        
        self._save_state()
        self._notify_callbacks("execution_started")
    
    def record_node_execution(self, node_name: str, input_state: Dict[str, Any], 
                            output_state: Dict[str, Any], duration: float = 0):
        """Record a node execution."""
        timestamp = datetime.now().isoformat()
        
        # Determine agent from node name
        agent = self._get_agent_from_node(node_name)
        
        state_record = {
            "timestamp": timestamp,
            "node_name": node_name,
            "agent": agent,
            "duration": duration,
            "input_state": self._sanitize_state(input_state),
            "output_state": self._sanitize_state(output_state),
            "turn": output_state.get("current_turn", 0),
            "speaker": output_state.get("current_speaker", "UNKNOWN"),
            "metadata": {
                "conversation_length": len(output_state.get("conversation_history", [])),
                "audio_segments": len(output_state.get("audio_segments", [])),
                "script_lines": len(output_state.get("script_lines", []))
            }
        }
        
        self.state_history.append(state_record)
        self.current_state = output_state
        self.execution_metadata["node_count"] += 1
        
        self._save_state()
        self._notify_callbacks("node_executed", state_record)
    
    def record_error(self, node_name: str, error: Exception, state: Dict[str, Any]):
        """Record an execution error."""
        timestamp = datetime.now().isoformat()
        
        error_record = {
            "timestamp": timestamp,
            "node_name": node_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "state": self._sanitize_state(state)
        }
        
        self.execution_metadata["error_count"] += 1
        self.state_history.append(error_record)
        
        self._save_state()
        self._notify_callbacks("error_occurred", error_record)
    
    def end_execution(self, final_state: Dict[str, Any]):
        """End the execution monitoring."""
        self.execution_metadata["end_time"] = datetime.now().isoformat()
        
        if self.execution_metadata["start_time"]:
            start = datetime.fromisoformat(self.execution_metadata["start_time"])
            end = datetime.fromisoformat(self.execution_metadata["end_time"])
            self.execution_metadata["total_duration"] = (end - start).total_seconds()
        
        self.current_state = final_state
        self._save_state()
        self._notify_callbacks("execution_completed")
    
    def _get_agent_from_node(self, node_name: str) -> str:
        """Extract agent name from node name."""
        node_lower = node_name.lower()
        if "nexus" in node_lower:
            return "NEXUS"
        elif "reco" in node_lower:
            return "RECO"
        elif "stat" in node_lower:
            return "STAT"
        elif "end" in node_lower:
            return "END"
        else:
            return "UNKNOWN"
    
    def _sanitize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize state for JSON serialization."""
        sanitized = {}
        
        for key, value in state.items():
            try:
                # Try to serialize to check if it's JSON-compatible
                json.dumps(value)
                sanitized[key] = value
            except (TypeError, ValueError):
                # If not serializable, convert to string representation
                sanitized[key] = str(value)
        
        return sanitized
    
    def _save_state(self):
        """Save current state to file."""
        data = {
            "execution_metadata": self.execution_metadata,
            "current_state": self._sanitize_state(self.current_state) if self.current_state else None,
            "state_history": self.state_history,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save state to {self.output_file}: {e}")
    
    def _notify_callbacks(self, event_type: str, data: Dict[str, Any] = None):
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback({
                    "event_type": event_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Warning: Callback error: {e}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution."""
        if not self.state_history:
            return {"status": "no_execution"}
        
        # Calculate agent statistics
        agent_stats = {}
        total_duration = 0
        
        for record in self.state_history:
            if "error_type" in record:  # Skip error records
                continue
                
            agent = record.get("agent", "UNKNOWN")
            duration = record.get("duration", 0)
            
            if agent not in agent_stats:
                agent_stats[agent] = {"count": 0, "total_duration": 0}
            
            agent_stats[agent]["count"] += 1
            agent_stats[agent]["total_duration"] += duration
            total_duration += duration
        
        # Get current turn info
        current_turn = 0
        max_turns = 0
        if self.current_state:
            current_turn = self.current_state.get("current_turn", 0)
            max_turns = self.current_state.get("max_turns", 0)
        
        return {
            "status": "active" if not self.execution_metadata["end_time"] else "completed",
            "session_id": self.execution_metadata.get("session_id", "unknown"),
            "start_time": self.execution_metadata["start_time"],
            "end_time": self.execution_metadata["end_time"],
            "total_duration": self.execution_metadata["total_duration"],
            "node_count": self.execution_metadata["node_count"],
            "error_count": self.execution_metadata["error_count"],
            "current_turn": current_turn,
            "max_turns": max_turns,
            "progress": (current_turn / max_turns * 100) if max_turns > 0 else 0,
            "agent_stats": agent_stats,
            "total_conversation_entries": len(self.current_state.get("conversation_history", [])) if self.current_state else 0
        }


class MonitoredOrchestrator:
    """Wrapper for AgentBasedOrchestrator with monitoring capabilities."""
    
    def __init__(self, monitor: StateMonitor = None):
        """Initialize with optional state monitor."""
        from uap_podcast.workflow import AgentBasedOrchestrator
        
        self.orchestrator = AgentBasedOrchestrator()
        self.monitor = monitor or StateMonitor()
        
    async def generate_podcast_with_monitoring(self, context: str, session_id: str = None, 
                                             max_turns: int = 3) -> Dict[str, Any]:
        """Generate podcast with real-time monitoring."""
        if not session_id:
            session_id = f"monitored_session_{int(time.time())}"
        
        # Prepare initial state
        initial_state = {
            "messages": [],
            "current_speaker": "NEXUS",
            "topic": "Monitored Podcast Generation",
            "context": {"summary": context},
            "interrupted": False,
            "audio_segments": [],
            "conversation_history": [],
            "current_turn": 0.0,
            "max_turns": max_turns,
            "session_id": session_id,
            "node_history": [],
            "current_node": "nexus_intro",
            "script_lines": [],
            "nexus_state": None,
            "reco_state": None,
            "stat_state": None
        }
        
        # Start monitoring
        self.monitor.start_execution(session_id, initial_state)
        
        try:
            # Get compiled graph
            graph = self.orchestrator.get_compiled_graph()
            
            # Execute with async monitoring
            final_state = None
            async for event in graph.astream(initial_state, config={"recursion_limit": 60}):
                for node_name, node_output in event.items():
                    # Record node execution
                    start_time = time.time()
                    # Simulate some processing time for demonstration
                    await asyncio.sleep(0.01)
                    duration = time.time() - start_time
                    
                    self.monitor.record_node_execution(
                        node_name=node_name,
                        input_state=initial_state,
                        output_state=node_output,
                        duration=duration
                    )
                    
                    final_state = node_output
                    initial_state = node_output  # Update for next iteration
            
            # End monitoring
            self.monitor.end_execution(final_state or initial_state)
            
            return {
                "status": "success",
                "final_state": final_state,
                "execution_summary": self.monitor.get_execution_summary()
            }
            
        except Exception as e:
            self.monitor.record_error("unknown", e, initial_state)
            self.monitor.end_execution(initial_state)
            
            return {
                "status": "error",
                "error": str(e),
                "execution_summary": self.monitor.get_execution_summary()
            }


def print_status_callback(event: Dict[str, Any]):
    """Simple callback to print status updates."""
    event_type = event["event_type"]
    timestamp = event["timestamp"]
    
    if event_type == "execution_started":
        print(f"🚀 [{timestamp}] Execution started")
    elif event_type == "node_executed":
        data = event["data"]
        print(f"📊 [{timestamp}] {data['agent']} executed {data['node_name']} (turn: {data['turn']})")
    elif event_type == "error_occurred":
        data = event["data"]
        print(f"❌ [{timestamp}] Error in {data['node_name']}: {data['error_message']}")
    elif event_type == "execution_completed":
        print(f"✅ [{timestamp}] Execution completed")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Create monitor with callback
    monitor = StateMonitor("podcast_execution_state.json")
    monitor.add_callback(print_status_callback)
    
    # Create monitored orchestrator
    orchestrator = MonitoredOrchestrator(monitor)
    
    # Sample context
    context = """
    {
        "metrics": [
            {"metric_name": "Patient Satisfaction", "value": 4.2, "target": 4.5},
            {"metric_name": "Readmission Rate", "value": 0.08, "target": 0.05}
        ]
    }
    """
    
    async def run_demo():
        print("Starting monitored podcast generation...")
        result = await orchestrator.generate_podcast_with_monitoring(
            context=context,
            session_id="demo_session",
            max_turns=2
        )
        
        print("\n📊 Execution Summary:")
        summary = result["execution_summary"]
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # Run the demo
    asyncio.run(run_demo())
