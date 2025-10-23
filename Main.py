#!/usr/bin/env python3
"""
Main entry point for UAP Podcast application.

This script provides CLI interface for generating podcasts using the
multi-agent conversation system.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from uap_podcast.models.podcast import PodcastEngine, PodcastContext
from uap_podcast.utils.config import Config
from uap_podcast.utils.logging import setup_logger
from uap_podcast.server import run_server


async def generate_podcast_cli(
    topic: str = None,
    max_turns: int = 6,
    file_choice: str = "both",
    output_dir: str = "."
):
    """Generate podcast using CLI."""
    logger = setup_logger("podcast_cli")
    logger.info("Starting podcast generation via CLI")
    
    try:
        # Initialize engine
        engine = PodcastEngine()
        
        # Load context
        context = PodcastContext.load_from_files(file_choice)
        
        # Infer topic if not provided
        if not topic:
            topic = engine.infer_topic_from_context(context.content)
        
        logger.info(f"Generating podcast: {topic}")
        logger.info(f"Max turns: {max_turns}")
        logger.info(f"Context files: {context.metadata.get('files', [])}")
        
        # Generate segments
        segments = []
        script_lines = []
        
        # Introduction sequence
        logger.info("Generating introductions...")
        
        # Nexus intro
        nexus_intro_audio = await engine.synthesize_speech(Config.NEXUS_INTRO, "NEXUS")
        segments.append(nexus_intro_audio)
        script_lines.append(f"Agent Nexus: {Config.NEXUS_INTRO}")
        
        # Reco intro  
        reco_intro_audio = await engine.synthesize_speech(Config.RECO_INTRO, "RECO")
        segments.append(reco_intro_audio)
        script_lines.append(f"Agent Reco: {Config.RECO_INTRO}")
        
        # Stat intro
        stat_intro_audio = await engine.synthesize_speech(Config.STAT_INTRO, "STAT")
        segments.append(stat_intro_audio)
        script_lines.append(f"Agent Stat: {Config.STAT_INTRO}")
        
        # Topic introduction
        logger.info("Generating topic introduction...")
        topic_intro = await engine.generate_nexus_topic_intro(context.content)
        topic_intro_audio = await engine.synthesize_speech(topic_intro, "NEXUS")
        segments.append(topic_intro_audio)
        script_lines.append(f"Agent Nexus: {topic_intro}")
        
        # Main conversation
        conversation_history = [
            {"speaker": "NEXUS", "text": Config.NEXUS_INTRO},
            {"speaker": "RECO", "text": Config.RECO_INTRO},
            {"speaker": "STAT", "text": Config.STAT_INTRO},
            {"speaker": "NEXUS", "text": topic_intro}
        ]
        
        logger.info(f"Generating {max_turns} conversation turns...")
        
        for turn in range(max_turns):
            logger.info(f"Turn {turn + 1}/{max_turns}")
            
            # Reco turn
            last_stat_text = ""
            for entry in reversed(conversation_history):
                if entry["speaker"] == "STAT":
                    last_stat_text = entry["text"]
                    break
            
            reco_response = await engine.generate_agent_response(
                role="RECO",
                context=context.content,
                last_speaker_text=last_stat_text,
                turn_count=turn,
                conversation_history=conversation_history
            )
            
            reco_audio = await engine.synthesize_speech(reco_response, "RECO")
            segments.append(reco_audio)
            script_lines.append(f"Agent Reco: {reco_response}")
            conversation_history.append({"speaker": "RECO", "text": reco_response})
            
            # Stat turn
            stat_response = await engine.generate_agent_response(
                role="STAT", 
                context=context.content,
                last_speaker_text=reco_response,
                turn_count=turn,
                conversation_history=conversation_history
            )
            
            stat_audio = await engine.synthesize_speech(stat_response, "STAT")
            segments.append(stat_audio)
            script_lines.append(f"Agent Stat: {stat_response}")
            conversation_history.append({"speaker": "STAT", "text": stat_response})
        
        # Conclusion
        logger.info("Generating conclusion...")
        outro_audio = await engine.synthesize_speech(Config.NEXUS_OUTRO, "NEXUS")
        segments.append(outro_audio)
        script_lines.append(f"Agent Nexus: {Config.NEXUS_OUTRO}")
        
        # Generate output files
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Audio file
        audio_file = Path(output_dir) / f"podcast_{timestamp}.wav"
        final_audio_path = engine.concatenate_audio_segments(segments, str(audio_file))
        
        # Script file
        script_file = Path(output_dir) / f"podcast_script_{timestamp}.txt"
        with open(script_file, "w", encoding="utf-8") as f:
            f.write("\\n".join(script_lines))
        
        # Calculate duration
        duration = engine.audio.get_wav_duration(final_audio_path)
        
        logger.info("Podcast generation completed!")
        logger.info(f"Audio file: {final_audio_path}")
        logger.info(f"Script file: {script_file}")
        logger.info(f"Duration: {duration:.1f} seconds")
        
        # Display script
        print("\\n" + "="*60)
        print("PODCAST SCRIPT")
        print("="*60)
        for line in script_lines:
            print(line)
            print()
        
        # Cleanup
        engine.cleanup_temp_files()
        
        return {
            "audio_file": final_audio_path,
            "script_file": script_file,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Podcast generation failed: {e}")
        raise


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="UAP Podcast Generator - Multi-agent conversation system"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command (original direct approach)
    gen_parser = subparsers.add_parser("generate", help="Generate a podcast using direct approach")
    gen_parser.add_argument("--topic", type=str, help="Podcast topic (auto-inferred if not provided)")
    gen_parser.add_argument("--turns", type=int, default=6, help="Number of conversation turns (default: 6)")
    gen_parser.add_argument("--files", choices=["data.json", "metric_data.json", "both"], 
                           default="both", help="Context files to use (default: both)")
    gen_parser.add_argument("--output", type=str, default=".", help="Output directory (default: current)")
    
    # LangGraph command (new orchestration approach)
    langgraph_parser = subparsers.add_parser("langgraph", help="Generate a podcast using LangGraph orchestration")
    langgraph_parser.add_argument("--topic", type=str, help="Podcast topic (auto-inferred if not provided)")
    langgraph_parser.add_argument("--turns", type=int, default=6, help="Number of conversation turns (default: 6)")
    langgraph_parser.add_argument("--files", choices=["data.json", "metric_data.json", "both"], 
                                 default="both", help="Context files to use (default: both)")
    langgraph_parser.add_argument("--recursion-limit", type=int, default=60, help="LangGraph recursion limit (default: 60)")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the web server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    server_parser.add_argument("--port", type=int, default=8001, help="Port to bind to (default: 8001)")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        # Generate podcast using direct approach
        result = asyncio.run(generate_podcast_cli(
            topic=args.topic,
            max_turns=args.turns,
            file_choice=args.files,
            output_dir=args.output
        ))
        print(f"\\nGeneration complete! Audio saved to: {result['audio_file']}")
        
    elif args.command == "langgraph":
        # Generate podcast using Agent-based LangGraph orchestration
        async def run_langgraph():
            from uap_podcast.workflow import AgentBasedOrchestrator
            orchestrator = AgentBasedOrchestrator()
            
            return await orchestrator.generate_podcast(
                topic=args.topic,
                max_turns=args.turns,
                file_choice=args.files,
                recursion_limit=args.recursion_limit
            )
        
        result = asyncio.run(run_langgraph())
        
        print(f"\\n🎉 Agent-based LangGraph generation complete!")
        print(f"   Audio: {result['audio_file']}")
        print(f"   Script: {result['script_file']}")
        print(f"   Duration: {result['duration_seconds']} seconds")
        print(f"   Session ID: {result['session_id']}")
        print(f"   Workflow: {result['workflow_type']}")
        
    elif args.command == "server":
        # Start server
        print(f"Starting UAP Podcast server on {args.host}:{args.port}")
        run_server(host=args.host, port=args.port, reload=args.reload)
        
    elif args.command == "info":
        # Show info
        print("UAP Podcast Generator")
        print("="*40)
        print(f"Version: 1.0.0")
        print(f"Agents: Nexus (Host), Reco (Recommendations), Stat (Statistics)")
        print(f"Capabilities: Multi-agent AI conversations, TTS synthesis, Podcast generation")
        print()
        print("Available files in current directory:")
        engine = PodcastEngine()
        files = engine.list_json_files()
        if files:
            for file in files:
                print(f"  - {file}")
        else:
            print("  No JSON files found")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
