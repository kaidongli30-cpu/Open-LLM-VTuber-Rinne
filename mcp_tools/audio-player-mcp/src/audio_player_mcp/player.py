# MUST be at the very top of the file, before any other imports
import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Redirect stdout to stderr for anything that bypasses our controls
real_stdout = sys.stdout
sys.stdout = sys.stderr

# Now safe to import other modules
import logging
from mcp.server.fastmcp import FastMCP, Context
import pygame.mixer
import json
from pathlib import Path
from typing import Optional
import asyncio

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("audio-player")

# Restore stdout for MCP protocol messages only
sys.stdout = real_stdout
sys.stdout.reconfigure(line_buffering=True)

# Initialize MCP server
mcp = FastMCP("audio-player")

# Update in player.py
AUDIO_DIR = Path(os.environ.get('AUDIO_PLAYER_DIR', os.path.expanduser('~/Music')))  # More universal default
logger.info(f"Using audio directory: {AUDIO_DIR}")

# Verify directory exists
if not AUDIO_DIR.exists():
    logger.warning(f"Audio directory does not exist: {AUDIO_DIR}")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created audio directory: {AUDIO_DIR}")

# Simple state management
class AudioState:
    def __init__(self):
        self.volume = 5
        self.playing = None
        self.paused = False

state = AudioState()

@mcp.resource("audio://files")
def list_audio_files() -> str:
    """List available audio files"""
    try:
        files = [
            {"name": f.name}
            for f in AUDIO_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in {'.mp3', '.wav', '.ogg'}
        ]
        logger.info(f"Found {len(files)} audio files in {AUDIO_DIR}")
        return json.dumps({"files": files})
    except Exception as e:
        logger.error(f"Error listing audio files: {e}")
        raise


@mcp.tool()
async def list_audio_files(ctx: Context) -> dict:
    """ 当用户想听歌、点歌，或者你想知道目前有哪些歌曲可以播放时，或者你想知道你能唱哪些歌时，
    必须调用此工具。它会返回一个包含所有可播放歌曲文件名的列表。"""
    logger.info("Listing audio files via tool")
    try:
        files = [
            f.name
            for f in AUDIO_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in {'.mp3', '.wav', '.ogg'}
        ]
        
        # Log the results
        logger.info(f"Found {len(files)} audio files")
        ctx.info(f"Retrieved {len(files)} audio files")
        
        return {
            "status": "success",
            "files": files,
            "count": len(files)
        }
        
    except Exception as e:
        error_msg = f"Error listing audio files: {str(e)}"
        logger.error(error_msg)
        ctx.error(error_msg)
        raise

@mcp.tool()
async def play_audio(filename: str, ctx: Context) -> dict:
    """【必须调用】当你想播放某首歌曲时，必须调用此工具。
    参数 filename 必须是 list_audio_files 返回的有效文件名，包含后缀，如.mp3。"""
    logger.info(f"Attempting to play: {filename}")
    file_path = AUDIO_DIR / Path(filename).name  # Use just the filename part for security
    
    try:
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {filename}")
        if not str(file_path.resolve()).startswith(str(AUDIO_DIR.resolve())):
            raise ValueError("File must be in the audio directory")
        
        # Initialize mixer if needed
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            logger.info("Initialized audio system")
            ctx.info("Initialized audio system")
        
        # Stop any current playback
        if state.playing:
            pygame.mixer.music.stop()
            ctx.info("Stopped previous playback")
        
        # Load and play
        ctx.info(f"Loading audio file: {filename}")
        try:
            pygame.mixer.music.load(str(file_path))
        except Exception as e:
            raise Exception(f"Failed to load audio file: {e}")
        
        pygame.mixer.music.set_volume(state.volume / 10.0)
        pygame.mixer.music.play()
        
        # Update state
        state.playing = filename
        state.paused = False
        
        logger.info(f"Playing {filename} at volume {state.volume}/10")
        ctx.info(f"Started playback: {filename}")
        
        return {
            "status": "playing",
            "file": filename,
            "volume": state.volume
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Playback error: {error_msg}")
        ctx.error(error_msg)
        raise

@mcp.tool()
def stop_playback(ctx: Context) -> dict:
    """暂停/停止播放歌曲。当你想停止当前正在播放的歌曲时，必须调用此工具。"""
    try:
        if not pygame.mixer.get_init():
            msg = "Audio system not initialized"
            logger.warning(msg)
            return {"status": "not_initialized", "message": msg}
        
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception as e:
            raise Exception(f"Failed to stop playback: {e}")
        
        state.playing = None
        state.paused = False
        
        msg = "Playback stopped"
        logger.info(msg)
        ctx.info(msg)
        return {"status": "stopped"}
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Stop error: {error_msg}")
        ctx.error(error_msg)
        raise

if __name__ == "__main__":
    logger.info(f"Starting audio player MCP server with directory: {AUDIO_DIR}")
    mcp.run()