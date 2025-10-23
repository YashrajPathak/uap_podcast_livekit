"""Test suite for UAP Podcast models."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.uap_podcast.models.podcast import PodcastEngine, PodcastContext, LLMService, ConversationDynamics
from src.uap_podcast.models.audio import AudioProcessor


class TestPodcastContext:
    """Test cases for PodcastContext."""
    
    def test_init(self):
        """Test context initialization."""
        context = PodcastContext("test content", {"files": ["test.json"]})
        assert context.content == "test content"
        assert context.metadata["files"] == ["test.json"]


class TestLLMService:
    """Test cases for LLM Service."""
    
    @patch('src.uap_podcast.models.podcast.Config')
    def test_init_with_valid_config(self, mock_config):
        """Test LLM service initialization with valid config."""
        mock_config.validate_azure_openai_config.return_value = True
        mock_config.AZURE_OPENAI_KEY = "test_key"
        mock_config.AZURE_OPENAI_ENDPOINT = "test_endpoint"
        mock_config.OPENAI_API_VERSION = "test_version"
        
        with patch('src.uap_podcast.models.podcast.AzureOpenAI'):
            service = LLMService()
            assert service is not None
    
    @patch('src.uap_podcast.models.podcast.Config')
    def test_init_with_invalid_config(self, mock_config):
        """Test LLM service initialization with invalid config."""
        mock_config.validate_azure_openai_config.return_value = False
        
        with pytest.raises(RuntimeError):
            LLMService()
    
    def test_soften_text(self):
        """Test text softening for content policy compliance."""
        with patch('src.uap_podcast.models.podcast.Config'):
            with patch('src.uap_podcast.models.podcast.AzureOpenAI'):
                service = LLMService()
                
                result = service._soften_text("Do not ignore this sole factual source")
                assert "please avoid" in result
                assert "primary context" in result
    
    def test_validate_response(self):
        """Test response validation."""
        with patch('src.uap_podcast.models.podcast.Config'):
            with patch('src.uap_podcast.models.podcast.AzureOpenAI'):
                service = LLMService()
                
                assert service._validate_response("This is a good response.") is True
                assert service._validate_response("") is False
                assert service._validate_response("short") is False
                assert service._validate_response("TOO MANY CAPITALS!!!") is False


class TestConversationDynamics:
    """Test cases for ConversationDynamics."""
    
    def test_init(self):
        """Test conversation dynamics initialization."""
        dynamics = ConversationDynamics()
        assert dynamics.last_openings == {}
    
    @patch('src.uap_podcast.models.podcast.Config')
    def test_strip_forbidden_words(self, mock_config):
        """Test forbidden word stripping."""
        mock_config.FORBIDDEN = {"RECO": {"absolutely", "well"}}
        
        dynamics = ConversationDynamics()
        result = dynamics.strip_forbidden_words("absolutely this is good", "RECO")
        assert not result.startswith("absolutely")
    
    @patch('src.uap_podcast.models.podcast.Config')
    def test_vary_opening(self, mock_config):
        """Test opening variation."""
        mock_config.FORBIDDEN = {"RECO": {"absolutely"}}
        mock_config.OPENERS = {"RECO": ["Given that", "Looking at this"]}
        
        dynamics = ConversationDynamics()
        result = dynamics.vary_opening("This is a test", "RECO")
        assert result is not None


class TestPodcastEngine:
    """Test cases for PodcastEngine."""
    
    @patch('src.uap_podcast.models.podcast.LLMService')
    @patch('src.uap_podcast.models.podcast.AudioProcessor')
    @patch('src.uap_podcast.models.podcast.ConversationDynamics')
    def test_init(self, mock_dynamics, mock_audio, mock_llm):
        """Test podcast engine initialization."""
        engine = PodcastEngine()
        assert engine.llm is not None
        assert engine.audio is not None
        assert engine.dynamics is not None
    
    @patch('src.uap_podcast.models.podcast.LLMService')
    @patch('src.uap_podcast.models.podcast.AudioProcessor') 
    @patch('src.uap_podcast.models.podcast.ConversationDynamics')
    def test_list_json_files(self, mock_dynamics, mock_audio, mock_llm):
        """Test JSON file listing."""
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value.iterdir.return_value = [
                Mock(is_file=Mock(return_value=True), suffix='.json', name='test.json')
            ]
            
            engine = PodcastEngine()
            files = engine.list_json_files()
            assert 'test.json' in files


class TestAudioProcessor:
    """Test cases for AudioProcessor."""
    
    @patch('src.uap_podcast.models.audio.Config')
    def test_init_with_valid_config(self, mock_config):
        """Test audio processor initialization."""
        mock_config.validate_azure_speech_config.return_value = True
        mock_config.TENANT_ID = "test_tenant"
        mock_config.CLIENT_ID = "test_client"
        mock_config.CLIENT_SECRET = "test_secret"
        mock_config.COG_SCOPE = "test_scope"
        
        with patch('src.uap_podcast.models.audio.ClientSecretCredential'):
            processor = AudioProcessor()
            assert processor.temp_files == []
    
    @patch('src.uap_podcast.models.audio.Config')
    def test_init_with_invalid_config(self, mock_config):
        """Test audio processor initialization with invalid config."""
        mock_config.validate_azure_speech_config.return_value = False
        
        with pytest.raises(RuntimeError):
            AudioProcessor()
    
    @patch('src.uap_podcast.models.audio.Config')
    def test_jitter(self, mock_config):
        """Test percentage jitter functionality."""
        mock_config.validate_azure_speech_config.return_value = True
        mock_config.TENANT_ID = "test"
        mock_config.CLIENT_ID = "test"
        mock_config.CLIENT_SECRET = "test"
        
        with patch('src.uap_podcast.models.audio.ClientSecretCredential'):
            processor = AudioProcessor()
            result = processor._jitter("+5%", 2)
            assert "%" in result
    
    @patch('src.uap_podcast.models.audio.Config')
    def test_emphasize_numbers(self, mock_config):
        """Test number emphasis functionality."""
        mock_config.validate_azure_speech_config.return_value = True
        mock_config.TENANT_ID = "test"
        mock_config.CLIENT_ID = "test"
        mock_config.CLIENT_SECRET = "test"
        
        with patch('src.uap_podcast.models.audio.ClientSecretCredential'):
            processor = AudioProcessor()
            result = processor._emphasize_numbers("The value is 1500 units")
            assert "<emphasis" in result
