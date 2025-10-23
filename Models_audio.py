"""Audio processing models and utilities for UAP Podcast application."""

import os
import re
import wave
import tempfile
import random
import datetime
from typing import Tuple
import azure.cognitiveservices.speech as speechsdk
from azure.identity import ClientSecretCredential

from utils.config import Config
from utils.logging import default_logger


class AudioProcessor:
    """Handles audio processing including TTS synthesis and audio file manipulation."""
    
    def __init__(self):
        """Initialize audio processor with Azure Speech configuration."""
        if not Config.validate_azure_speech_config():
            raise RuntimeError("Missing AAD Speech env vars (TENANT_ID, CLIENT_ID, CLIENT_SECRET, SPEECH_REGION)")
        
        self.cred = ClientSecretCredential(
            tenant_id=Config.TENANT_ID,
            client_id=Config.CLIENT_ID,
            client_secret=Config.CLIENT_SECRET
        )
        self.temp_files = []
    
    def get_auth_token(self) -> str:
        """Get authentication token for Azure Speech service."""
        try:
            default_logger.debug(f"Attempting to get token with scope: {Config.COG_SCOPE}")
            default_logger.debug(f"Using tenant: {Config.TENANT_ID}, client: {Config.CLIENT_ID}")
            tok = self.cred.get_token(Config.COG_SCOPE).token
            return f"aad#{Config.RESOURCE_ID}#{tok}" if Config.RESOURCE_ID else tok
        except Exception as e:
            default_logger.error(f"Failed to get Azure Speech token: {e}")
            default_logger.error(f"Check these environment variables:")
            default_logger.error(f"- TENANT_ID: {'SET' if Config.TENANT_ID else 'MISSING'}")
            default_logger.error(f"- CLIENT_ID: {'SET' if Config.CLIENT_ID else 'MISSING'}")
            default_logger.error(f"- CLIENT_SECRET: {'SET' if Config.CLIENT_SECRET else 'MISSING'}")
            default_logger.error(f"- SPEECH_REGION: {'SET' if Config.SPEECH_REGION else 'MISSING'}")
            raise
    
    # def _jitter(self, pct: str, spread: int = 3) -> str:
    #     """Apply random jitter to percentage values for more natural speech."""
    #     m = re.match(r'([+-]?\d+)%', pct.strip())
    #     base = int(m.group(1)) if m else 0
    #     j = random.randint(-spread, spread)
    #     return f"{base+j}%"
    
    # def _emphasize_numbers(self, text: str) -> str:
    #     """Add emphasis to numbers in text for better TTS pronunciation."""
    #     wrap = lambda s: f'<emphasis level="moderate">{s}</emphasis>'
    #     t = re.sub(r'\b\d{3,}(\.\d+)?\b', lambda m: wrap(m.group(0)), text)
    #     t = re.sub(r'\b-?\d+(\.\d+)?%\b', lambda m: wrap(m.group(0)), t)
    #     return t

    def _jitter(self, pct: str, spread: int = 3) -> str:
        """Apply random jitter to percentage values for more natural variation."""
        m = re.match(r'([+-]?\d+)%', pct.strip())
        base = int(m.group(1)) if m else 0
        j = random.randint(-spread, spread)
        return f"{base + j}%"

    def _emphasize_numbers(self, text: str) -> str:
        """Add emphasis to numbers in text for better TTS pronunciation."""
        def wrap(s: str) -> str:
            return f'<emphasis level="moderate">{s}</emphasis>'
        t = re.sub(r'\b\d{3,}(\.\d+)?\b', lambda m: wrap(m.group(0)), text)
        t = re.sub(r'\b-?\d+(\.\d+)?%\b', lambda m: wrap(m.group(0)), t)
        return t
    
    # def _add_clause_pauses(self, text: str) -> str:
    #     """Add natural pauses at clause boundaries."""
    #     t = re.sub(r',\s', ',<break time="220ms"/> ', text)
    #     t = re.sub(r';\s', ';<break time="260ms"/> ', t)
    #     t = re.sub(r'\bHowever\b', 'However,<break time="220ms"/>', t, flags=re.I)
    #     t = re.sub(r'\bBut\b', 'But,<break time="220ms"/>', t, flags=re.I)
    #     return t
    
    # def _calculate_inflection(self, text: str, role: str) -> Tuple[str, str]:
    #     """Calculate pitch and rate inflections based on text content and role."""
    #     voice_plan = Config.VOICE_PLAN[role]
    #     base_pitch = voice_plan["base_pitch"]
    #     base_rate = voice_plan["base_rate"]
    #     pitch = self._jitter(base_pitch, 3)
    #     rate = self._jitter(base_rate, 2)
        
    #     # More natural pitch variations based on content
    #     if text.strip().endswith("?"):
    #         try:
    #             p = int(pitch.replace('%', ''))
    #             pitch = f"{p+4}%"
    #         except ValueError:
    #             pitch = "+4%"
    #     elif re.search(r'\bhowever\b|\bbut\b', text, re.I):
    #         try:
    #             p = int(pitch.replace('%', ''))
    #             pitch = f"{p-2}%"
    #         except ValueError:
    #             pitch = "-2%"
    #     elif any(word in text.lower() for word in ['surprising', 'shocking', 'unexpected', 'dramatic']):
    #         try:
    #             p = int(pitch.replace('%', ''))
    #             pitch = f"{p+3}%"
    #         except ValueError:
    #             pitch = "+3%"
        
    #     return pitch, rate
    
    def _generate_ssml_nexus(self, text: str) -> str:
        """Generate SSML for Nexus agent using HD DragonHDLatestNeural voice."""
        return f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>
    <voice name='en-US-Emma2:DragonHDLatestNeural' parameters='temperature=0.6'>
    <mstts:silence type="Sentenceboundary" value="200ms"/>
    {text}
    </voice>
</speak>"""

    def _generate_ssml_reco(self, text: str) -> str:
        """Generate SSML for Reco agent using HD DragonHDLatestNeural voice."""
        return f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>
    <voice name='en-US-Ava3:DragonHDLatestNeural' parameters='temperature=0.6'>
    <mstts:silence type="Sentenceboundary" value="200ms"/>
    {text}
    </voice>
</speak>"""

    def _generate_ssml_stat(self, text: str) -> str:
        """Generate SSML for Stat agent using HD DragonHDLatestNeural voice."""
        return f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>
    <voice name='en-US-Andrew3:DragonHDLatestNeural' parameters='temperature=0.6'>
    <mstts:silence type="Sentenceboundary" value="200ms"/>
    {text}
    </voice>
</speak>"""
    
    def text_to_ssml(self, text: str, role: str) -> str:
        """Convert text to SSML markup for a specific agent role - Simplified version."""
        # Use simplified SSML generation based on role
        if role.upper() == "NEXUS":
            return self._generate_ssml_nexus(text)
        elif role.upper() == "RECO":
            return self._generate_ssml_reco(text)
        elif role.upper() == "STAT":
            return self._generate_ssml_stat(text)
        else:
            # Default to Nexus if role not recognized
            return self._generate_ssml_nexus(text)
    
    def synthesize_speech(self, ssml: str) -> str:
        """
        Synthesize speech from SSML and return path to audio file.
        
        Args:
            ssml: SSML markup for speech synthesis
            
        Returns:
            Path to generated audio file
            
        Raises:
            RuntimeError: If TTS synthesis fails
        """
        cfg = speechsdk.SpeechConfig(auth_token=self.get_auth_token(), region=Config.SPEECH_REGION)
        cfg.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
        
        # Create temporary file
        fd, tmp_path = tempfile.mkstemp(prefix="seg_", suffix=".wav")
        os.close(fd)
        self.temp_files.append(tmp_path)
        
        out = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=out)
        
        # Try SSML synthesis first
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            default_logger.debug(f"SSML synthesis successful: {tmp_path}")
            return tmp_path
        
        # Fallback to plain text
        default_logger.warning("SSML synthesis failed, attempting plain text fallback")
        plain_text = re.sub(r'<[^>]+>', ' ', ssml)
        result = synthesizer.speak_text_async(plain_text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            default_logger.info(f"Plain text synthesis successful: {tmp_path}")
            return tmp_path
        
        # Clean up failed file
        try:
            os.remove(tmp_path)
            self.temp_files.remove(tmp_path)
        except (OSError, ValueError):
            pass
        
        raise RuntimeError(f"TTS synthesis failed for both SSML and plain text: {result.reason}")
    
    def get_wav_duration(self, path: str) -> float:
        """Get duration of WAV file in seconds."""
        try:
            with wave.open(path, "rb") as wav_file:
                frame_rate = wav_file.getframerate() or 24000
                return wav_file.getnframes() / float(frame_rate)
        except Exception as e:
            default_logger.error(f"Failed to get WAV duration for {path}: {e}")
            return 0.0
    
    def concatenate_audio_segments(self, segments: list[str], output_path: str, sample_rate: int = 24000) -> str:
        """
        Concatenate multiple audio segments into a single file.
        
        Args:
            segments: List of paths to audio segment files
            output_path: Path for output file
            sample_rate: Audio sample rate (default: 24000)
            
        Returns:
            Path to final concatenated audio file
            
        Raises:
            RuntimeError: If concatenation fails
        """
        fd, tmp_path = tempfile.mkstemp(prefix="final_", suffix=".wav")
        os.close(fd)
        
        try:
            with wave.open(tmp_path, "wb") as output_wav:
                output_wav.setnchannels(1)
                output_wav.setsampwidth(2)
                output_wav.setframerate(sample_rate)
                
                for segment_path in segments:
                    try:
                        with wave.open(segment_path, "rb") as segment_wav:
                            # Verify format compatibility
                            if (segment_wav.getframerate(), segment_wav.getnchannels(), 
                                segment_wav.getsampwidth()) != (sample_rate, 1, 2):
                                raise RuntimeError(f"Segment format mismatch: {segment_path}")
                            
                            # Copy audio data
                            output_wav.writeframes(segment_wav.readframes(segment_wav.getnframes()))
                    except Exception as e:
                        default_logger.error(f"Failed to process segment {segment_path}: {e}")
                        raise
            
            # Move to final location
            try:
                os.replace(tmp_path, output_path)
                default_logger.info(f"Audio concatenation successful: {output_path}")
                return output_path
            except PermissionError:
                # Handle file locked scenario
                base, ext = os.path.splitext(output_path)
                alt_path = f"{base}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
                os.replace(tmp_path, alt_path)
                default_logger.warning(f"Output was locked; wrote to {alt_path}")
                return alt_path
                
        except Exception as e:
            # Clean up temporary file on error
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise RuntimeError(f"Audio concatenation failed: {e}")
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError as e:
                default_logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        self.temp_files.clear()
