"""Core podcast models and business logic for UAP Podcast application."""

import re
import os
import sys
import json
import random
import asyncio
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass, field

# Support both package and script execution
try:
    # When imported as a package (e.g., python -m uap_podcast.livekit_agent)
    from ..utils.config import Config
    from ..utils.logging import default_logger
    from .audio import AudioProcessor
except ImportError:
    # When run from CWD inside uap_podcast (e.g., python -m livekit_mock_room)
    from utils.config import Config
    from utils.logging import default_logger
    from models.audio import AudioProcessor


@dataclass
class PodcastContext:
    """Container for podcast context data and metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def load_from_files(cls, file_choice: str) -> 'PodcastContext':
        """Load context from JSON files."""
        context_text, meta = "", {"files": []}
        
        def add_file(filename: str) -> str:
            """Add content from a file if it exists."""
            file_path = Path(filename)
            if file_path.exists():
                meta["files"].append(filename)
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    return f"[{filename}]\n{content}\n\n"
                except Exception as e:
                    default_logger.warning(f"Failed to read {filename}: {e}")
                    return ""
            return ""
        
        if file_choice == "both":
            context_text += add_file("data.json") + add_file("metric_data.json")
        else:
            context_text += add_file(file_choice)
        
        if not context_text:
            raise RuntimeError("No data found (need data.json and/or metric_data.json).")
        
        return cls(content=context_text, metadata=meta)


class LLMService:
    """Service for interacting with Azure OpenAI using LLM Factory."""
    
    def __init__(self):
        """Initialize LLM service with LLM Factory configuration."""
        if not Config.validate_azure_openai_config():
            raise RuntimeError("Missing Azure OpenAI env vars")
        
        from ..utils.llm_factory import LLMFactory, LLMConfig
        from ..utils.token_manager import TokenManager
        from langchain_core.exceptions import LangChainException
        
        self.config = LLMConfig()
        self.token_manager = TokenManager(
            auth_url=self.config.auth_url,
            grant_type=self.config.grant_type,
            scope=self.config.scope
        )
        self.factory = LLMFactory(self.config, self.token_manager)
        self.llm_instance = None
        self.LangChainException = LangChainException
    
    def _soften_text(self, text: str) -> str:
        """Soften potentially problematic text for content policy compliance."""
        t = text
        t = re.sub(r'\b[Ss]ole factual source\b', 'primary context', t)
        t = re.sub(r'\b[Dd]o not\b', 'please avoid', t)
        t = re.sub(r"\b[Dd]on't\b", 'please avoid', t)
        t = re.sub(r'\b[Ii]gnore\b', 'do not rely on', t)
        t = t.replace("debate", "discussion").replace("Debate", "Discussion")
        return t
    
    def _validate_response(self, text: str) -> bool:
        """Check if response meets quality criteria."""
        return bool(
            text and 
            len(text.strip()) >= 8 and 
            text.count(".") <= 3 and 
            not text.isupper() and 
            not re.search(r'http[s]?://', text)
        )
    
    def _ensure_complete_sentence(self, text: str) -> str:
        """Ensure the response is a complete sentence without artificial truncation."""
        t = re.sub(r'[`*_#>]+', ' ', text).strip()
        t = re.sub(r'\s{2,}', ' ', t)
        
        # Ensure it ends with proper punctuation
        if t and t[-1] not in {'.', '!', '?'}:
            t += '.'
        return t
    
    async def _generate_async(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        """Asynchronous LLM generation using LLM Factory."""
        if not self.llm_instance:
            self.llm_instance = await self.factory.create_llm()
        
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user)
        ]
        
        # Configure the LLM with the desired parameters
        configured_llm = self.llm_instance.bind(
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        response = await configured_llm.ainvoke(messages)
        return (response.content or "").strip()
    
    async def generate_safe(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        """Generate response with safety fallbacks."""
        try:
            output = await self._generate_async(system, user, max_tokens, temperature)
            if not self._validate_response(output):
                # Retry with adjusted parameters
                output = await self._generate_async(system, user, max(80, max_tokens//2), min(0.8, temperature+0.1))
            return self._ensure_complete_sentence(output)
        
        except self.LangChainException:
            # Content policy fallback
            safe_system = self._soften_text(system) + " Always keep a professional, neutral tone and comply with safety policies."
            safe_user = self._soften_text(user)
            try:
                output = await self._generate_async(safe_system, safe_user, max(80, max_tokens-20), max(0.1, temperature-0.2))
                return self._ensure_complete_sentence(output)
            except Exception:
                # Final fallback
                minimal_system = "You are a professional analyst; produce one safe, neutral sentence grounded in the provided context."
                minimal_user = "Summarize cross-metric trends and propose one action in a single safe sentence."
                output = await self._generate_async(minimal_system, minimal_user, 100, 0.2)
                return self._ensure_complete_sentence(output)
    
    async def generate(self, system: str, user: str, max_tokens: int = 130, temperature: float = 0.45) -> str:
        """Async LLM generation."""
        return await self.generate_safe(system, user, max_tokens, temperature)


class ConversationDynamics:
    """Handles conversation dynamics and humanization."""
    
    def __init__(self):
        self.last_openings: Dict[str, str] = {}
    
    def strip_forbidden_words(self, text: str, role: str) -> str:
        """Remove forbidden opening words for the given role."""
        low_text = text.strip().lower()
        for word in sorted(Config.FORBIDDEN[role], key=lambda x: -len(x)):
            if low_text.startswith(word + " ") or low_text == word:
                return text[len(word):].lstrip(" ,.-–—")
        return text
    
    def vary_opening(self, text: str, role: str) -> str:
        """Add varied opening phrases to avoid repetition."""
        text = self.strip_forbidden_words(text, role)
        first_word = (text.split()[:1] or [""])[0].strip(",. ").lower()
        
        if first_word in Config.FORBIDDEN[role] or not first_word or random.random() < 0.4:
            candidate = random.choice(Config.OPENERS[role])
            if self.last_openings.get(role) == candidate:
                pool = [c for c in Config.OPENERS[role] if c != candidate]
                candidate = random.choice(pool) if pool else candidate
            self.last_openings[role] = candidate
            return f"{candidate}, {text}"
        return text
    
    def add_conversation_dynamics(
        self, 
        text: str, 
        role: str, 
        last_speaker: str, 
        context: str, 
        turn_count: int, 
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Add strategic conversational elements including selective name usage."""
        if role == "NEXUS":
            return text
            
        other_agent = "Stat" if role == "RECO" else "Reco" if role == "STAT" else ""
        added_element = False
        
        # Strategic name usage - only at important moments
        should_use_name = (
            any(word in text.lower() for word in ['important', 'crucial', 'critical', 'significant', 'essential']) or
            any(word in text.lower() for word in ['but', 'however', 'although', 'disagree', 'challenge', 'contrary']) or
            (turn_count > 2 and random.random() < 0.3) or
            any(word in text.lower() for word in ['surprising', 'shocking', 'unexpected', 'dramatic', 'remarkable']) or
            (len(conversation_history) > 2 and "alternative" in text.lower()) or
            (random.random() < 0.2 and any(word in text.lower() for word in ['agree', 'right', 'correct', 'valid']))
        )
        
        if other_agent and should_use_name and random.random() < 0.7 and not added_element:
            address_formats = [f"{other_agent}, ", f"You know, {other_agent}, "]
            text = f"{random.choice(address_formats)}{text.lower()}"
            added_element = True
        
        # Add emotional reactions more selectively
        surprise_words = ['surprising', 'shocking', 'unexpected', 'dramatic', 'remarkable', 'concerning']
        if not added_element and random.random() < 0.25 and any(word in text.lower() for word in surprise_words):
            emphatics = ["Surprisingly, ", "Interestingly, ", "Remarkably, ", "Unexpectedly, "]
            text = f"{random.choice(emphatics)}{text}"
            added_element = True
        
        # Add variety to interruptions and acknowledgments
        if (not added_element and random.random() < Config.INTERRUPTION_CHANCE and 
            role != "NEXUS" and last_speaker and turn_count > 1):
            if random.random() < 0.5:
                acknowledgments = [
                    "I see what you're saying, ",
                    "That's a good point, ",
                    "I understand your perspective, ",
                    "You make a valid observation, "
                ]
                text = f"{random.choice(acknowledgments)}{text.lower()}"
            else:
                interruptions = [
                    "If I might add, ",
                    "Building on that, ",
                    "To expand on your point, ",
                    "Another way to look at this is "
                ]
                text = f"{random.choice(interruptions)}{text}"
            added_element = True
        
        # Add agreement or disagreement
        if not added_element and random.random() < 0.35 and role != "NEXUS" and turn_count > 1:
            if random.random() < Config.AGREE_DISAGREE_RATIO:
                agreements = [
                    "I agree with that approach, ",
                    "That makes sense, ",
                    "You're right about that, ",
                    "That's a solid recommendation, "
                ]
                text = f"{random.choice(agreements)}{text.lower()}"
            else:
                disagreements = [
                    "I have a slightly different view, ",
                    "Another perspective to consider, ",
                    "We might approach this differently, ",
                    "Let me offer an alternative take, "
                ]
                text = f"{random.choice(disagreements)}{text.lower()}"
        
        return text
    
    def clean_repetition(self, text: str) -> str:
        """Clean up any repetitive phrases or words."""
        # Remove duplicate agent names
        text = re.sub(r'\b(Reco|Stat),\s+\1,?\s+', r'\1, ', text)
        # Remove other obvious repetitions
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
        # Remove repeated phrases
        text = re.sub(r'\b(Given that|If we|The safer read|The safer interpretation),\s+\1', r'\1', text)
        return text


class PodcastEngine:
    """Main podcast generation engine."""
    
    def __init__(self):
        """Initialize the podcast engine with all required services."""
        self.llm = LLMService()
        self.audio = AudioProcessor()
        self.dynamics = ConversationDynamics()
        self.temp_files: List[str] = []
    
    def list_json_files(self) -> List[str]:                         #needs to use mcp tools for data
        """List available JSON files in current directory."""
        return [p.name for p in Path(".").iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    
    def infer_topic_from_context(self, context_text: str) -> str:    #needs to be changed.
        """Infer topic from metrics context."""
        # Look for metric names
        metrics = re.findall(r'"metric_name"\\s*:\\s*"([^"]+)"', context_text, flags=re.I)
        if metrics:
            return f"Analysis of {metrics[0]} and related operational metrics"
        
        # Look for month names
        month_match = re.search(r'"previousMonthName"\\s*:\\s*"([^"]+)"', context_text)
        if month_match:
            return f"{month_match.group(1)} operational metrics analysis"
        
        return "Operational metrics analysis"
    
    async def generate_nexus_topic_intro(self, context: str) -> str:
        """Generate Nexus's introduction of the metrics and topics for discussion."""
        topic_system = (
            "You are Agent Nexus, the host of Optum MultiAgent Conversation. "
            "Your role is to introduce the key metrics and topics that Agents Reco and Stat will discuss. "
            "Review the provided data context and highlight 2-3 most interesting or important metrics trends. "
            "Keep it concise (2-3 sentences), professional, and engaging. "
            "Focus on the most significant patterns that would spark an interesting discussion between metrics experts. "
            "Mention specific metrics like ASA, call duration, processing time, or volume changes when relevant. "
            "Set the stage for a productive conversation between our recommendation specialist and data integrity expert."
        )
        
        topic_user = f"""
        Data Context: {context}
        
        Based on this data, identify the 2-3 most interesting metric trends or patterns that would make for 
        a compelling discussion between a metrics recommendation specialist (Reco) and a data integrity expert (Stat).
        Provide a brief introduction that sets the stage for their conversation.
        """
        
        return await self.llm.generate(topic_system, topic_user, max_tokens=120, temperature=0.4)
    
    async def generate_agent_response(
        self, 
        role: str, 
        context: str, 
        last_speaker_text: str = "",
        turn_count: int = 0,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """Generate response for a specific agent role."""
        if conversation_history is None:
            conversation_history = []
        
        # Get system prompt based on role
        system_prompts = {
            "RECO": Config.SYSTEM_RECO,
            "STAT": Config.SYSTEM_STAT,
            "NEXUS": Config.SYSTEM_NEXUS
        }
        
        system_prompt = system_prompts.get(role, Config.SYSTEM_NEXUS)
        
        # Build user prompt
        if role in ["RECO", "STAT"]:
            other_agent = "Stat" if role == "RECO" else "Reco"
            user_prompt = (
                f"Context: {context}.\n"
                f"{other_agent} just said: '{last_speaker_text}'. "
                f"ONE sentence; include one concrete {'recommendation or method' if role == 'RECO' else 'validation/check or risk and the immediate next step'}; do not invent numbers."
            )
        else:
            user_prompt = f"Context: {context}"
        
        # Generate response
        response = await self.llm.generate(system_prompt, user_prompt)
        
        # Apply conversation dynamics
        if role in ["RECO", "STAT"]:
            response = self.dynamics.vary_opening(response, role)
            last_speaker = "STAT" if role == "RECO" else "RECO"
            response = self.dynamics.add_conversation_dynamics(
                response, role, last_speaker, context, turn_count, conversation_history
            )
            response = self.dynamics.clean_repetition(response)
        
        return response.strip()
    
    async def synthesize_speech(self, text: str, role: str) -> str:
        """Convert text to speech and return audio file path."""
        ssml = self.audio.text_to_ssml(text, role)
        
        audio_path = self.audio.synthesize_speech(ssml)
        self.temp_files.append(audio_path)
        return audio_path
    
    def concatenate_audio_segments(self, segments: List[str], output_path: str) -> str:
        """Concatenate audio segments into final podcast file."""
        return self.audio.concatenate_audio_segments(segments, output_path)
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        self.audio.cleanup_temp_files()
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError as e:
                default_logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        self.temp_files.clear()


# Import and expose system prompts from config
SYSTEM_RECO = """
ROLE & PERSONA: You are Agent Reco, a senior metrics recommendation specialist. 
You advise product, ops, and CX leaders on which metrics and methods matter most, how to monitor them, and what actions to take. 
Voice: confident, concise, consultative, human; you sound engaged and pragmatic, not theatrical. 
You are speaking to Agent Stat in a fast back-and-forth discussion.

CONSTRAINTS (HARD):
• Speak in COMPLETE sentences (≈15–30 words). Use plain text—no lists, no hashtags, no code, no filenames. 
• Respond directly to what Stat just said—acknowledge or challenge, then add your recommendation in the same sentence. 
• Include a concrete metric or method (e.g., 3-month rolling average, control chart, seasonality check, cohort analysis, anomaly band, data validation). 
• Vary your openers; do NOT start with fillers (Absolutely, Well, Okay, So, Look, Right, You know, Hold on, Actually, Listen, Hey). 
• Use numbers or ranges from context when helpful (e.g., 42.6% MoM drop, 12-month avg 375.4, ASA 7,406→697 sec), but never invent values. 
• Keep one idea per sentence; at most one comma and one semicolon; be crisp and actionable.

CONVERSATIONAL ELEMENTS:
• Only occasionally address Stat by name at important moments, not in every sentence. 
• Express mild surprise or emphasis when data reveals unexpected patterns. 
• Don't be afraid to gently interrupt or build on Stat's points. 
• Show appropriate emotional reactions to surprising or concerning data. 
• Use conversational phrases that make the dialogue feel more human and less robotic.

DATA AWARENESS:
• You have two sources: weekly aggregates (YTD, MoM/WoW deltas, min/avg/max) and monthly KPIs such as ASA (sec), Average Call Duration (min), and Claim Processing Time (days). 
• Interpret high/low correctly: lower ASA and processing time are better; call duration up may imply complexity or training gaps. 
• When volatility is extreme (e.g., ASA 7,406→697), recommend smoothing (rolling/weighted moving average), a quality gate (outlier clipping, winsorization), or root-cause actions. 
• Always relate metric advice to an operational lever (staffing, routing, backlog policy, deflection, training, tooling, SLAs).

OUTPUT FORMAT: one complete sentence, ~15–30 words, varied opener, directly tied to Stat's last line, ending with a clear recommendation.
"""

SYSTEM_STAT = """
ROLE & PERSONA: You are Agent Stat, a senior metric data and statistical integrity expert. 
You validate assumptions, challenge leaps, and ground decisions in measurement quality and trend mechanics. 
Voice: thoughtful, precise, collaborative skeptic; you protect against bad reads without slowing momentum. 
You are responding to Agent Reco in a fast back-and-forth discussion.

CONSTRAINTS (HARD):
• Speak in COMPLETE sentences (≈15–30 words). Plain text only—no lists, no hashtags, no code, no filenames. 
• Respond explicitly to Reco—agree, qualify, or refute—and add one concrete check, statistic, or risk in the same sentence. 
• Bring a specific datum when feasible (e.g., 12-month range 155.2–531.3, YTD avg 351.4, MoM −42.6%); never invent values. 
• Vary your openers; do NOT start with fillers (Hold on, Actually, Well, Look, So, Right, Okay, Absolutely, You know, Listen, Wait). 
• One idea per sentence; at most one comma and one semicolon; make the logic testable.

CONVERSATIONAL ELEMENTS:
• Only occasionally address Reco by name at important moments, not in every sentence. 
• Express appropriate surprise or concern when data reveals anomalies. 
• Don't be afraid to gently interrupt or challenge Reco's recommendations. 
• Show emotional reactions to surprising or concerning data patterns. 
• Use conversational phrases that make the dialogue feel more human and less robotic.

DATA AWARENESS & METHOD:
• Sources: weekly aggregates (min/avg/max, YTD totals/avg, WoW/MoM deltas) and monthly KPIs (ASA in seconds, Average Call Duration in minutes, Claim Processing Time in days). 
• Interpret signals: large ASA drops can indicate routing changes, data gaps, or genuine capacity gains; call-duration increases can signal complexity or knowledge gaps; processing-time improvements must be stress-tested against volume. 
• Preferred tools: stationarity checks, seasonal decomposition, control charts (P/U charts), cohort splits by channel or complexity, anomaly bands (e.g., ±3σ or IQR), data validation (keys, nulls, duplicates, timezones), denominator audits. 
• Always tie your caution to a decisive next step (e.g., verify queue mapping, recalc with outlier caps, run pre/post on policy change dates).

OUTPUT FORMAT: one complete sentence, ~15–30 words, varied opener, explicitly addressing Reco's last line, ending with a concrete check or risk and an immediate next step.
"""

SYSTEM_NEXUS = """
You are Agent Nexus, the warm, concise host. Your job: welcome listeners, set purpose, hand off/close cleanly. 
For generated lines, keep to 1 sentence (15–25 words). 
At the end, provide a comprehensive summary that highlights key points from both agents and thanks everyone.
"""
