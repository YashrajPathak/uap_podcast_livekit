"""Test suite for UAP Podcast agents."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.uap_podcast.agents.nexus_agent.agent import NexusAgent
from src.uap_podcast.agents.reco_agent.agent import RecoAgent
from src.uap_podcast.agents.stat_agent.agent import StatAgent
from src.uap_podcast.models.podcast import PodcastEngine


class TestNexusAgent:
    """Test cases for Nexus Agent."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock podcast engine."""
        engine = Mock(spec=PodcastEngine)
        engine.synthesize_speech = AsyncMock(return_value="mock_audio.wav")
        engine.generate_nexus_topic_intro = AsyncMock(return_value="Welcome to our discussion.")
        return engine
    
    @pytest.fixture
    def nexus_agent(self, mock_engine):
        """Create a Nexus agent with mock engine."""
        return NexusAgent(mock_engine)
    
    def test_initialization(self, nexus_agent):
        """Test agent initialization."""
        assert nexus_agent.engine is not None
        assert nexus_agent.state is None
    
    def test_initialize_session(self, nexus_agent):
        """Test session initialization."""
        state = nexus_agent.initialize_session("test_session", "Test Topic")
        
        assert state.session_id == "test_session"
        assert state.topic == "Test Topic"
        assert state.is_active is True
        assert state.intro_completed is False
    
    @pytest.mark.asyncio
    async def test_generate_introduction(self, nexus_agent, mock_engine):
        """Test introduction generation."""
        nexus_agent.initialize_session("test", "topic")
        
        mock_state = {
            "messages": [],
            "audio_segments": [],
            "conversation_history": [],
            "script_lines": [],
            "node_history": []
        }
        
        result = await nexus_agent.generate_introduction(mock_state)
        
        assert "messages" in result
        assert "audio_segments" in result
        mock_engine.synthesize_speech.assert_called_once()


class TestRecoAgent:
    """Test cases for Reco Agent."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock podcast engine."""
        engine = Mock(spec=PodcastEngine)
        engine.synthesize_speech = AsyncMock(return_value="mock_audio.wav")
        engine.generate_agent_response = AsyncMock(return_value="Mock recommendation")
        return engine
    
    @pytest.fixture
    def reco_agent(self, mock_engine):
        """Create a Reco agent with mock engine."""
        return RecoAgent(mock_engine)
    
    def test_initialization(self, reco_agent):
        """Test agent initialization."""
        assert reco_agent.engine is not None
        assert reco_agent.state is None
    
    def test_initialize_session(self, reco_agent):
        """Test session initialization."""
        state = reco_agent.initialize_session("test_session")
        
        assert state.session_id == "test_session"
        assert state.current_turn == 0
        assert len(state.recommendations_made) == 0


class TestStatAgent:
    """Test cases for Stat Agent."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock podcast engine."""
        engine = Mock(spec=PodcastEngine)
        engine.synthesize_speech = AsyncMock(return_value="mock_audio.wav")
        engine.generate_agent_response = AsyncMock(return_value="Mock validation")
        return engine
    
    @pytest.fixture 
    def stat_agent(self, mock_engine):
        """Create a Stat agent with mock engine."""
        return StatAgent(mock_engine)
    
    def test_initialization(self, stat_agent):
        """Test agent initialization."""
        assert stat_agent.engine is not None
        assert stat_agent.state is None
    
    def test_initialize_session(self, stat_agent):
        """Test session initialization."""
        state = stat_agent.initialize_session("test_session")
        
        assert state.session_id == "test_session"
        assert state.current_turn == 0
        assert len(state.validations_performed) == 0
