"""
Azure Speech-to-Text integration for Chainlit podcast app
"""

import asyncio
import tempfile
import os
from typing import Optional
import azure.cognitiveservices.speech as speechsdk
from azure.identity import ClientSecretCredential

from uap_podcast.utils.config import Config
from uap_podcast.utils.logging import default_logger


class SpeechToTextService:
    """Service for converting audio to text using Azure Speech Services."""
    
    def __init__(self):
        """Initialize the speech service with Azure credentials."""
        self.tenant_id = Config.TENANT_ID
        self.client_id = Config.CLIENT_ID
        self.client_secret = Config.CLIENT_SECRET
        self.speech_region = Config.SPEECH_REGION
        self.resource_id = Config.RESOURCE_ID
        
        if not all([self.tenant_id, self.client_id, self.client_secret, self.speech_region]):
            raise RuntimeError("Missing Azure Speech credentials")
        
        self.credential = ClientSecretCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret
        )
    
    def _get_auth_token(self) -> str:
        """Get authentication token for Azure Speech service."""
        token = self.credential.get_token("https://cognitiveservices.azure.com/.default").token
        return f"aad#{self.resource_id}#{token}" if self.resource_id else token
    
    async def audio_bytes_to_text(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to text using Azure Speech Recognition."""
        try:
            # Create temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # Configure speech recognition
                speech_config = speechsdk.SpeechConfig(
                    auth_token=self._get_auth_token(),
                    region=self.speech_region
                )
                speech_config.speech_recognition_language = "en-US"
                
                # Create audio config from file
                audio_config = speechsdk.audio.AudioConfig(filename=temp_path)
                
                # Create speech recognizer
                speech_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=speech_config,
                    audio_config=audio_config
                )
                
                # Perform recognition
                result = await asyncio.to_thread(speech_recognizer.recognize_once)
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    default_logger.info(f"Speech recognition successful: {result.text}")
                    return result.text
                elif result.reason == speechsdk.ResultReason.NoMatch:
                    default_logger.warning("No speech could be recognized")
                    return "Sorry, I couldn't understand the audio. Please try speaking more clearly."
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = result.cancellation_details
                    default_logger.error(f"Speech recognition canceled: {cancellation_details.reason}")
                    return f"Speech recognition failed: {cancellation_details.error_details}"
                else:
                    default_logger.error(f"Unexpected speech recognition result: {result.reason}")
                    return "Speech recognition failed due to an unexpected error."
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            default_logger.error(f"Error in speech to text conversion: {e}")
            return f"Error converting speech to text: {str(e)}"


# Global instance for the Chainlit app
speech_service = None

def get_speech_service() -> SpeechToTextService:
    """Get or create the global speech service instance."""
    global speech_service
    if speech_service is None:
        speech_service = SpeechToTextService()
    return speech_service


async def speech_to_text(audio_bytes: bytes) -> str:
    """Convert audio bytes to text - main function for Chainlit integration."""
    try:
        service = get_speech_service()
        return await service.audio_bytes_to_text(audio_bytes)
    except Exception as e:
        default_logger.error(f"Failed to initialize speech service: {e}")
        return f"Speech recognition service unavailable: {str(e)}"
