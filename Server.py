"""FastAPI server for UAP Podcast application."""

import os
import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .utils.config import Config
from .utils.logging import default_logger, get_session_logger
from .models.podcast import PodcastEngine
from .livekit_agent import run_cli as run_livekit_cli


# Initialize FastAPI app
app = FastAPI(
    title="UAP Podcast Generator",
    description="Multi-agent podcast generation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
podcast_engine: Optional[PodcastEngine] = None


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for generating AI responses."""
    system_prompt: str
    user_prompt: str
    max_tokens: int = 150
    temperature: float = 0.45


class AudioRequest(BaseModel):
    """Request model for generating audio."""
    text: str
    role: str = "NEXUS"


class PodcastGenerationRequest(BaseModel):
    """Request model for full podcast generation."""
    topic: Optional[str] = None
    max_turns: int = 6
    file_choice: str = "both"
    session_id: Optional[str] = None


class PodcastResponse(BaseModel):
    """Response model for podcast generation."""
    session_id: str
    audio_file: str
    script_file: str
    duration: float
    success: bool
    message: str


# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the podcast engine on startup."""
    global podcast_engine
    try:
        default_logger.info("Initializing podcast engine...")
        podcast_engine = PodcastEngine()
        default_logger.info("Podcast engine initialized successfully")
    except Exception as e:
        default_logger.error(f"Failed to initialize podcast engine: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global podcast_engine
    if podcast_engine:
        podcast_engine.cleanup_temp_files()
        default_logger.info("Podcast engine cleaned up")


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "uap-podcast-generator",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "name": "UAP Podcast Generator",
        "description": "Multi-agent podcast generation system",
        "agents": ["Nexus", "Reco", "Stat"],
        "capabilities": [
            "AI-powered conversation generation",
            "Text-to-speech synthesis",
            "Multi-agent orchestration",
            "Metrics analysis discussion"
        ],
        "endpoints": {
            "health": "/health",
            "generate_response": "/generate-response",
            "generate_audio": "/generate-audio", 
            "generate_podcast": "/generate-podcast",
            "list_files": "/list-files",
            "livekit_start": "/livekit/start"
        }
    }


@app.post("/generate-response")
async def generate_response(request: GenerateRequest):
    """Generate AI response using LLM."""
    if not podcast_engine:
        raise HTTPException(status_code=500, detail="Podcast engine not initialized")
    
    try:
        response = await podcast_engine.llm.generate(
            request.system_prompt,
            request.user_prompt,
            request.max_tokens,
            request.temperature
        )
        return {"text": response, "success": True}
    except Exception as e:
        default_logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/livekit/start")
async def start_livekit(background_tasks: BackgroundTasks):
    """Start the LiveKit worker in a background task.

    Note: This expects you to run the LiveKit CLI simulator to provide the room context:
      python -m livekit.agents.cli simulate
    """
    try:
        # Run the LiveKit CLI worker in a background thread so API remains responsive
        background_tasks.add_task(run_livekit_cli)
        return {"success": True, "message": "LiveKit worker starting; launch the LiveKit CLI simulator to connect."}
    except Exception as e:
        default_logger.error(f"Error starting LiveKit worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-audio")
async def generate_audio_endpoint(request: AudioRequest):
    """Generate audio from text."""
    if not podcast_engine:
        raise HTTPException(status_code=500, detail="Podcast engine not initialized")
    
    try:
        audio_path = await podcast_engine.synthesize_speech(request.text, request.role)
        return {
            "audio_file": os.path.basename(audio_path),
            "audio_path": audio_path,
            "success": True
        }
    except Exception as e:
        default_logger.error(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-files")
async def list_json_files():
    """List available JSON files for context."""
    if not podcast_engine:
        raise HTTPException(status_code=500, detail="Podcast engine not initialized")
    
    try:
        files = podcast_engine.list_json_files()
        return {"files": files, "success": True}
    except Exception as e:
        default_logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_podcast_background(
    request: PodcastGenerationRequest,
    session_logger
) -> Dict[str, Any]:
    """Background task for podcast generation."""
    if not podcast_engine:
        raise RuntimeError("Podcast engine not initialized")
    
    try:
        session_logger.info(f"Starting podcast generation with parameters: {request}")
        
        # Load context
        from .models.podcast import PodcastContext
        context = PodcastContext.load_from_files(request.file_choice)
        
        # Infer topic if not provided
        topic = request.topic or podcast_engine.infer_topic_from_context(context.content)
        
        # Generate conversation segments
        segments = []
        script_lines = []
        conversation_history = []
        
        # Introduction sequence
        session_logger.info("Generating introduction sequence")
        
        # Nexus intro
        nexus_intro_audio = await podcast_engine.synthesize_speech(Config.NEXUS_INTRO, "NEXUS")
        segments.append(nexus_intro_audio)
        script_lines.append(f"Agent Nexus: {Config.NEXUS_INTRO}")
        conversation_history.append({"speaker": "NEXUS", "text": Config.NEXUS_INTRO})
        
        # Reco intro
        reco_intro_audio = await podcast_engine.synthesize_speech(Config.RECO_INTRO, "RECO")
        segments.append(reco_intro_audio)
        script_lines.append(f"Agent Reco: {Config.RECO_INTRO}")
        conversation_history.append({"speaker": "RECO", "text": Config.RECO_INTRO})
        
        # Stat intro
        stat_intro_audio = await podcast_engine.synthesize_speech(Config.STAT_INTRO, "STAT")
        segments.append(stat_intro_audio)
        script_lines.append(f"Agent Stat: {Config.STAT_INTRO}")
        conversation_history.append({"speaker": "STAT", "text": Config.STAT_INTRO})
        
        # Topic introduction
        session_logger.info("Generating topic introduction")
        topic_intro = await podcast_engine.generate_nexus_topic_intro(context.content)
        topic_intro_audio = await podcast_engine.synthesize_speech(topic_intro, "NEXUS")
        segments.append(topic_intro_audio)
        script_lines.append(f"Agent Nexus: {topic_intro}")
        conversation_history.append({"speaker": "NEXUS", "text": topic_intro})
        
        # Main conversation turns
        session_logger.info(f"Generating {request.max_turns} conversation turns")
        
        for turn in range(request.max_turns):
            session_logger.info(f"Generating turn {turn + 1}/{request.max_turns}")
            
            # Reco turn
            last_stat_text = ""
            for entry in reversed(conversation_history):
                if entry["speaker"] == "STAT":
                    last_stat_text = entry["text"]
                    break
            
            reco_response = await podcast_engine.generate_agent_response(
                role="RECO",
                context=context.content,
                last_speaker_text=last_stat_text,
                turn_count=turn,
                conversation_history=conversation_history
            )
            
            reco_audio = await podcast_engine.synthesize_speech(reco_response, "RECO")
            segments.append(reco_audio)
            script_lines.append(f"Agent Reco: {reco_response}")
            conversation_history.append({"speaker": "RECO", "text": reco_response})
            
            # Stat turn
            stat_response = await podcast_engine.generate_agent_response(
                role="STAT",
                context=context.content,
                last_speaker_text=reco_response,
                turn_count=turn,
                conversation_history=conversation_history
            )
            
            stat_audio = await podcast_engine.synthesize_speech(stat_response, "STAT")
            segments.append(stat_audio)
            script_lines.append(f"Agent Stat: {stat_response}")
            conversation_history.append({"speaker": "STAT", "text": stat_response})
        
        # Conclusion
        session_logger.info("Generating conclusion")
        outro_audio = await podcast_engine.synthesize_speech(Config.NEXUS_OUTRO, "NEXUS")
        segments.append(outro_audio)
        script_lines.append(f"Agent Nexus: {Config.NEXUS_OUTRO}")
        conversation_history.append({"speaker": "NEXUS", "text": Config.NEXUS_OUTRO})
        
        # Generate final files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = request.session_id or f"podcast_{timestamp}"
        
        # Audio file
        audio_file = f"podcast_{session_id}.wav"
        final_audio_path = podcast_engine.concatenate_audio_segments(segments, audio_file)
        
        # Script file
        script_file = f"podcast_script_{session_id}.txt"
        with open(script_file, "w", encoding="utf-8") as f:
            f.write("\\n".join(script_lines))
        
        # Calculate duration
        duration = podcast_engine.audio.get_wav_duration(final_audio_path)
        
        session_logger.info(f"Podcast generation completed: {audio_file}")
        
        return {
            "session_id": session_id,
            "audio_file": audio_file,
            "script_file": script_file,
            "duration": duration,
            "success": True,
            "message": "Podcast generated successfully"
        }
        
    except Exception as e:
        session_logger.error(f"Error during podcast generation: {e}")
        raise


@app.post("/generate-podcast", response_model=PodcastResponse)
async def generate_podcast_endpoint(
    request: PodcastGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate a complete podcast."""
    if not podcast_engine:
        raise HTTPException(status_code=500, detail="Podcast engine not initialized")
    
    # Create session logger
    session_id = request.session_id or f"podcast_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_logger = get_session_logger(session_id)
    
    try:
        # Run generation
        result = await generate_podcast_background(request, session_logger)
        return PodcastResponse(**result)
        
    except Exception as e:
        session_logger.error(f"Podcast generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Development server function
def run_server(host: str = "0.0.0.0", port: int = 8001, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "src.uap_podcast.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(reload=True)
