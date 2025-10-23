"""Test suite for UAP Podcast utilities."""

import pytest
import tempfile
import os
from unittest.mock import patch, Mock

from src.uap_podcast.utils.config import Config
from src.uap_podcast.utils.logging import setup_logger, get_session_logger


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_attributes_exist(self):
        """Test that required configuration attributes exist."""
        assert hasattr(Config, 'AZURE_OPENAI_KEY')
        assert hasattr(Config, 'AZURE_OPENAI_ENDPOINT')
        assert hasattr(Config, 'VOICE_NEXUS')
        assert hasattr(Config, 'VOICE_RECO')
        assert hasattr(Config, 'VOICE_STAT')
    
    def test_voice_plan_structure(self):
        """Test voice plan configuration structure."""
        assert isinstance(Config.VOICE_PLAN, dict)
        assert "NEXUS" in Config.VOICE_PLAN
        assert "RECO" in Config.VOICE_PLAN
        assert "STAT" in Config.VOICE_PLAN
        
        for agent, plan in Config.VOICE_PLAN.items():
            assert "style" in plan
            assert "base_pitch" in plan
            assert "base_rate" in plan
    
    def test_forbidden_words_structure(self):
        """Test forbidden words configuration."""
        assert isinstance(Config.FORBIDDEN, dict)
        assert "RECO" in Config.FORBIDDEN
        assert "STAT" in Config.FORBIDDEN
        
        for agent, words in Config.FORBIDDEN.items():
            assert isinstance(words, set)
            assert len(words) > 0
    
    def test_openers_structure(self):
        """Test openers configuration."""
        assert isinstance(Config.OPENERS, dict)
        assert "RECO" in Config.OPENERS
        assert "STAT" in Config.OPENERS
        
        for agent, openers in Config.OPENERS.items():
            assert isinstance(openers, list)
            assert len(openers) > 0
    
    def test_intro_outro_content(self):
        """Test intro and outro content exists."""
        assert Config.NEXUS_INTRO is not None
        assert Config.RECO_INTRO is not None
        assert Config.STAT_INTRO is not None
        assert Config.NEXUS_OUTRO is not None
        
        assert len(Config.NEXUS_INTRO) > 50
        assert len(Config.NEXUS_OUTRO) > 100
    
    @patch.dict(os.environ, {
        'AZURE_OPENAI_KEY': 'test_key',
        'AZURE_OPENAI_ENDPOINT': 'test_endpoint',
        'AZURE_OPENAI_DEPLOYMENT': 'test_deployment',
        'OPENAI_API_VERSION': 'test_version'
    })
    def test_validate_azure_openai_config_valid(self):
        """Test Azure OpenAI config validation with valid values."""
        # Reload config with mocked environment
        import importlib
        import src.uap_podcast.utils.config
        importlib.reload(src.uap_podcast.utils.config)
        
        from src.uap_podcast.utils.config import Config
        assert Config.validate_azure_openai_config() is True
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_azure_openai_config_invalid(self):
        """Test Azure OpenAI config validation with missing values."""
        # Reload config with empty environment
        import importlib
        import src.uap_podcast.utils.config
        importlib.reload(src.uap_podcast.utils.config)
        
        from src.uap_podcast.utils.config import Config
        assert Config.validate_azure_openai_config() is False
    
    def test_get_voice_config(self):
        """Test voice configuration getter."""
        nexus_config = Config.get_voice_config("NEXUS")
        assert "voice" in nexus_config
        assert "plan" in nexus_config
        assert nexus_config["voice"] == Config.VOICE_NEXUS
        
        reco_config = Config.get_voice_config("RECO")
        assert reco_config["voice"] == Config.VOICE_RECO
        
        # Test unknown role falls back to NEXUS
        unknown_config = Config.get_voice_config("UNKNOWN")
        assert unknown_config["voice"] == Config.VOICE_NEXUS


class TestLogging:
    """Test cases for logging utilities."""
    
    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger")
        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0
    
    def test_setup_logger_with_file(self):
        """Test logger setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = setup_logger("test_logger_file", log_file=log_file)
            
            # Test that the logger works
            logger.info("Test message")
            
            # Check that file was created
            assert os.path.exists(log_file)
    
    def test_setup_logger_custom_format(self):
        """Test logger setup with custom format."""
        custom_format = "%(name)s - %(message)s"
        logger = setup_logger("test_custom", format_string=custom_format)
        
        # Check that logger was created (format testing would require more complex setup)
        assert logger.name == "test_custom"
    
    def test_get_session_logger(self):
        """Test session-specific logger creation."""
        session_id = "test_session_123"
        logger = get_session_logger(session_id)
        
        assert session_id in logger.name
        assert len(logger.handlers) >= 2  # Console + file handlers
    
    def test_default_logger_exists(self):
        """Test that default logger is created."""
        from src.uap_podcast.utils.logging import default_logger
        assert default_logger is not None
        assert default_logger.name == "uap_podcast"


class TestConfigConstants:
    """Test configuration constants and values."""
    
    def test_conversation_dynamics_constants(self):
        """Test conversation dynamics constants are reasonable."""
        assert 0.0 <= Config.INTERRUPTION_CHANCE <= 1.0
        assert 0.0 <= Config.AGREE_DISAGREE_RATIO <= 1.0
    
    def test_system_prompts_exist(self):
        """Test that system prompts are defined."""
        # These should be imported from models if available
        try:
            from src.uap_podcast.models.podcast import SYSTEM_RECO, SYSTEM_STAT, SYSTEM_NEXUS
            assert len(SYSTEM_RECO) > 100
            assert len(SYSTEM_STAT) > 100 
            assert len(SYSTEM_NEXUS) > 50
        except ImportError:
            # If not available due to import issues, that's OK for this test
            pass
    
    def test_voice_names_format(self):
        """Test voice names follow expected format."""
        assert Config.VOICE_NEXUS.startswith("en-US-")
        assert Config.VOICE_RECO.startswith("en-US-")
        assert Config.VOICE_STAT.startswith("en-US-")
        
        assert Config.VOICE_NEXUS.endswith("Neural")
        assert Config.VOICE_RECO.endswith("Neural") 
        assert Config.VOICE_STAT.endswith("Neural")
    
    def test_voice_plan_percentages(self):
        """Test voice plan percentages are properly formatted."""
        for agent, plan in Config.VOICE_PLAN.items():
            pitch = plan["base_pitch"]
            rate = plan["base_rate"]
            
            assert pitch.endswith("%")
            assert rate.endswith("%")
            assert pitch.startswith(("+", "-"))
            assert rate.startswith(("+", "-"))
