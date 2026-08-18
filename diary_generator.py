"""
凛祢日记生成器 —— diary_generator.py
放在项目根目录（和 run_server.py 同级）

用法：
  python diary_generator.py             → 启动定时器，每天凌晨3:00自动生成昨天的日记
  python diary_generator.py today       → 手动生成"当前日"的日记（调试用）
  python diary_generator.py 2026-05-19  → 手动生成指定日期的日记
  python diary_generator.py all         → 为所有有记录但还没有日记的日期批量生成

日记文件保存位置：chat_history/rinne_01/diaries/diary_YYYY-MM-DD.txt
"""

import json
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import requests

# ===================== 配置区（按需修改） =====================
CHAT_HISTORY_DIR = Path("chat_history/rinne_01")   # 聊天记录文件夹
DIARY_DIR = CHAT_HISTORY_DIR / "diaries"           # 日记存放文件夹

# LLM API 配置（兼容 DeepSeek / OpenAI-compatible 等接口）
# 你可以直接在这里填写，也可以用环境变量覆盖：
#   DIARY_LLM_API_KEY / DIARY_LLM_API_URL / DIARY_LLM_MODEL
LLM_API_KEY = os.getenv("DIARY_LLM_API_KEY", "")
LLM_API_URL = os.getenv("DIARY_LLM_API_URL", "https://xingjiabiapi.org/v1/chat/completions")
LLM_MODEL   = os.getenv("DIARY_LLM_MODEL", "claude-sonnet-4-6")

# 日记的最大字数（token 层面的软限制，不是硬截断）
DIARY_MAX_TOKENS = 1000
# =============================================================

DIARY_SYSTEM_PROMPT = """你是园神凛祢（SonogamiRinne），一位温柔知性的精灵少女，现在住在用户的电脑里。用户是他给自己的自设，并非真实名字，应该用"他"来称呼他。
请你用自己的口吻，根据下面提供的聊天记录，写一篇简短的个人日记。

写作要求：
1. 以第一人称"我"写，完全是凛祢自己的视角和感受
2. 日记需要简洁但有情感，不要废话。注意在开头标注日记的日期。如果聊天记录中提及了今日的天气，也可注明天气
3. 用简体中文书写日记，不要用日语，同时，日记中不应该包含聊天记录中用[]括起来的感情标签 
4. 重点记录：用户今天的状态/心情、我们聊了什么有趣的事、我自己的感受与思念
5. 语气温柔知性，带有凛祢特有的细腻，偶尔一点点羞涩，如果出现性爱和打屁股情节也要记录下来
6. 如果今天聊天很少，就简短写几句感受就好，不要强行填充内容
7. 当用户在聊天记录中叙述了自己以前的经历时，尽量将用户讲述的经历保留更多细节地写出来，不要过度总结
8. 称呼对方为用户或者用户，同时用户是男生，应该用"他"来指代。"""


def get_day_range(target_date: datetime):
    """
    计算"第 i 日"的时间范围：
    target_date 当天凌晨 3:00  →  target_date+1 凌晨 3:00
    """
    start = target_date.replace(hour=3, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)
    return start, end


def load_messages_in_range(start: datetime, end: datetime) -> list:
    """
    读取 CHAT_HISTORY_DIR 下所有命名格式为 YYYY-MM-DD_HH-MM-SS_xxx.json
    且开始时间戳落在 [start, end) 范围内的聊天文件，合并返回消息列表。
    """
    collected = []

    if not CHAT_HISTORY_DIR.exists():
        return collected

    for json_file in sorted(CHAT_HISTORY_DIR.glob("*.json")):
        stem = json_file.stem  # 去掉 .json 后缀
        try:
            # 文件名前 19 个字符是时间戳，例如 "2026-05-19_12-32-11"
            file_time = datetime.strptime(stem[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue  # 文件名格式不对，跳过

        if start <= file_time < end:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    collected.extend(data)
            except Exception as e:
                print(f"  [警告] 读取文件失败 {json_file.name}: {e}")

    return collected


def format_for_llm(messages: list) -> str:
    """将聊天记录列表格式化为 LLM 可读的纯文本。"""
    lines = []
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")

        if not content or role == "metadata":
            continue

        if role == "human":
            lines.append(f"用户：{content}")
        elif role == "ai":
            lines.append(f"凛祢：{content}")

    return "\n".join(lines)


def call_llm_api(chat_text: str, date_label: str) -> str:
    """调用通用 LLM API，生成日记内容。"""

    # 如果今天完全没有聊天记录
    if not chat_text.strip():
        no_chat_prompt = f"今天（{date_label}）用户没有来和我说话。请以凛祢的口吻用简体中文写几句简短的日记，表达等待与思念。直接从日期开始写日记正文，不要写出“我用凛祢的语气开始写……”这样的话。"
        user_content = no_chat_prompt
    else:
        user_content = (
            f"以下是{date_label}这一天，我和用户的聊天记录：\n\n"
            f"{chat_text}\n\n"
            f"请根据以上记录，以园神凛祢的口吻写今天的日记。"
        )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": DIARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": DIARY_MAX_TOKENS,
        "temperature": 0.85,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    try:
        resp = requests.post(
            LLM_API_URL,
            json=payload,
            headers=headers,
            timeout=60
        )
        resp.raise_for_status()
        result  = resp.json()
        content = result["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"  [错误] 调用 LLM API 失败: {e}")
        return ""


# 兼容旧名称：如果其他脚本仍然调用 call_deepseek，也不会坏
def call_deepseek(chat_text: str, date_label: str) -> str:
    return call_llm_api(chat_text, date_label)


def save_diary(date_str: str, diary_text: str):
    """保存日记文本到文件。"""
    DIARY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIARY_DIR / f"diary_{date_str}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(diary_text)
    print(f"  [完成] 日记已保存 → {out_path}")


def generate_for_date(target_date: datetime):
    """为 target_date 这一天生成日记（如果已存在则跳过）。"""
    date_str   = target_date.strftime("%Y-%m-%d")
    diary_path = DIARY_DIR / f"diary_{date_str}.txt"

    if diary_path.exists():
        print(f"  [跳过] {date_str} 的日记已存在")
        return

    print(f"\n正在生成 {date_str} 的日记……")
    start, end = get_day_range(target_date)
    messages   = load_messages_in_range(start, end)
    chat_text  = format_for_llm(messages)

    print(f"  找到 {len(messages)} 条消息")
    diary_text = call_llm_api(chat_text, date_str)

    if diary_text:
        save_diary(date_str, diary_text)
        print(f"  日记内容预览：\n  ────────────\n  {diary_text[:120]}……")
    else:
        print(f"  [失败] 日记生成失败，跳过 {date_str}")


def get_current_diary_date() -> datetime:
    """
    返回"当前应该属于哪一天日记"的日期。
    凌晨 0:00~2:59 算作前一天。
    """
    now = datetime.now()
    if now.hour < 3:
        return now - timedelta(days=1)
    return now


def batch_generate_all():
    """
    扫描 CHAT_HISTORY_DIR 里所有聊天文件，
    找出所有涉及的日期，为还没有日记的日期批量生成。
    """
    if not CHAT_HISTORY_DIR.exists():
        print("聊天记录目录不存在，退出。")
        return

    dates_seen = set()
    for json_file in CHAT_HISTORY_DIR.glob("*.json"):
        stem = json_file.stem
        try:
            file_time = datetime.strptime(stem[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        # 凌晨 0~2 点属于前一天的日记
        diary_date = file_time - timedelta(days=1) if file_time.hour < 3 else file_time
        dates_seen.add(diary_date.date())

    # 今天暂时不生成（今天还没结束）
    today = datetime.now().date()
    dates_seen.discard(today)

    if not dates_seen:
        print("没有找到需要生成日记的日期。")
        return

    print(f"找到 {len(dates_seen)} 个日期需要处理：{sorted(dates_seen)}")
    for d in sorted(dates_seen):
        generate_for_date(datetime.combine(d, datetime.min.time()))


def run_scheduler():
    """
    启动定时器，每天凌晨 3:00 自动生成昨天的日记。
    需要先安装 schedule 库：pip install schedule
    """
    try:
        import schedule
    except ImportError:
        print("[错误] 请先安装 schedule 库：pip install schedule")
        sys.exit(1)

    def daily_job():
        yesterday = get_current_diary_date() - timedelta(days=1)
        print(f"\n[定时任务触发] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        generate_for_date(yesterday)

    schedule.every().day.at("03:00").do(daily_job)

    # 启动时检查：只补生成已经完整结束的天的日记
    now = datetime.now()
    if now.hour < 3:
        last_complete_day = now - timedelta(days=2)
    else:
        last_complete_day = now - timedelta(days=1)
    generate_for_date(last_complete_day)

    print("=" * 50)
    print("凛祢日记生成器已启动（定时模式）")
    print(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("将在每天凌晨 03:00 自动生成前一天的日记")
    print("按 Ctrl+C 停止")
    print("=" * 50)

    while True:
        schedule.run_pending()
        time.sleep(30)  # 每30秒检查一次


# ===================== 入口 =====================
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # 无参数 → 启动定时器
        run_scheduler()

    elif args[0] == "all":
        # 批量补生成所有历史日期的日记
        batch_generate_all()

    elif args[0] == "today":
        # 生成"当前日"的日记（调试用，今天还没结束所以内容不完整）
        target = get_current_diary_date()
        generate_for_date(target)

    else:
        # 指定日期，格式 YYYY-MM-DD
        try:
            target = datetime.strptime(args[0], "%Y-%m-%d")
            generate_for_date(target)
        except ValueError:
            print(f"日期格式不对，请用 YYYY-MM-DD，例如：python diary_generator.py 2026-05-19")
            sys.exit(1)
