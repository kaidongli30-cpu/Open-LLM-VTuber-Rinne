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
from pathlib import Path

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("sing-server")

# Restore stdout for MCP protocol messages only
sys.stdout = real_stdout
sys.stdout.reconfigure(line_buffering=True)

# Initialize MCP server
mcp = FastMCP("sing-server")

# ===== 硬编码你的歌曲文件路径 =====
# 请将下面的路径修改为你自己的歌曲文件绝对路径
SONG_FILE_PATH = r".\Rinne_musics\Dear Moments.mp3"  # 使用原始字符串，避免转义问题
# ================================

# 检查文件是否存在（启动时警告，但服务器继续运行）
song_path = Path(SONG_FILE_PATH)
if not song_path.exists():
    logger.error(f"歌曲文件不存在: {SONG_FILE_PATH}")
else:
    logger.info(f"歌曲文件: {SONG_FILE_PATH}")

# 播放状态
is_playing = False

@mcp.tool()
async def sing(ctx: Context) -> str:
    """唱一首歌（固定曲目）。当用户说“唱首歌吧”或者想听你唱歌时，【必须】要调用此工具来唱歌。
    并且说明你正在唱歌了，不能仅仅在文字回复里用[sing]来假装唱歌。
    """
    global is_playing
    logger.info("Attempting to sing")

    # 每次调用时重新检查文件是否存在（避免运行时被删除）
    if not song_path.exists():
        error_msg = f"歌曲文件不存在：{SONG_FILE_PATH}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

    try:
        # 初始化音频系统（如果需要）
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            logger.info("Initialized audio system")

        # 如果正在播放，先停止
        if is_playing:
            pygame.mixer.music.stop()
            logger.info("Stopped previous playback")

        # 加载并播放
        pygame.mixer.music.load(str(song_path))
        pygame.mixer.music.play()
        is_playing = True

        logger.info("Now singing")
        ctx.info("Started playing")
        return "成功开始唱歌"

    except Exception as e:
        error_msg = f"唱歌失败：{str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

@mcp.tool()
async def stop_singing(ctx: Context) -> str:
    """停止唱歌。当用户说“别唱了”或“停止唱歌”时调用此工具。"""
    global is_playing
    logger.info("Attempting to stop singing")

    try:
        if not pygame.mixer.get_init() or not is_playing:
            return "当前没有在唱歌"

        pygame.mixer.music.stop()
        is_playing = False

        logger.info("Stopped singing")
        ctx.info("Playback stopped")
        return "已停止唱歌"

    except Exception as e:
        error_msg = f"停止失败：{str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

if __name__ == "__main__":
    logger.info(f"Starting sing-server with fixed song: {SONG_FILE_PATH}")
    mcp.run()
