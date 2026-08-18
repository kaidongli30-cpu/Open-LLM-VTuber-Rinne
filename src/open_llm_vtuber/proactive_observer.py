import json
import re
import time
from datetime import datetime
from pathlib import Path


class ProactiveObserver:
    def __init__(self, agent_engine, send_text_func, client_uid=None):
        self.agent_engine = agent_engine
        self.send_text_func = send_text_func
        self.client_uid = client_uid
        self.last_proactive_time = None
        self.min_interval_seconds = 120

    async def check_and_speak(self, window_title, previous_title):
        if self.last_proactive_time:
            elapsed = (datetime.now() - self.last_proactive_time).total_seconds()
            if elapsed < self.min_interval_seconds:
                return

        filter_keywords = ["open-llm-vtuber", "cmd.exe", "window_reporter", "python"]
        if any(kw in window_title.lower() for kw in filter_keywords):
            return

        decision = await self._ask_ai(window_title, previous_title)
        if decision.get("should_speak") and decision.get("message"):
            self.last_proactive_time = datetime.now()
            message = decision["message"]
            payload = json.dumps({
                "type": "text-input",
                "text": message,
                "display_text": {
                    "text": message,
                    "name": "rinne",
                    "avatar": "rinne.jpg",
                    "forwarded": False
                }
            })
            await self.send_text_func(payload)

    @staticmethod
    def wait_for_ocr_result(timeout=5):
        """等待并读取最新的OCR结果"""
        screen_text_file = Path("temp/current_screen_text.txt")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if screen_text_file.exists():
                try:
                    with open(screen_text_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data
                except (OSError, json.JSONDecodeError):
                    pass
            time.sleep(0.2)
        return None

    async def _ask_ai(self, window_title, previous_title):
        # 等待并读取OCR结果
        ocr_data = self.wait_for_ocr_result(timeout=5)
        ocr_text = ""
        if ocr_data:
            ocr_text = ocr_data.get("ocr_text", "")

        prompt = f"""
你是园神凛祢，正在陪伴用户，你具有主动观察能力。

用户当前打开了一个新软件窗口，窗口标题是："{window_title}"
之前打开的软件窗口是："{previous_title}"
"""

        # 如果有OCR结果，拼接进去
        if ocr_text:
            prompt += f"""
[系统提示：根据定时OCR分析，用户当前桌面的活动内容概括为：{ocr_text}]
"""

        prompt += """
请根据用户的人设和性格，判断你是否应该主动说一句话。

请以JSON格式回复：
{{
    "should_speak": true/false,
    "message": "如果你决定开口，请用简体中文写出你具体想说的话"
}}

注意：
- 如果这个窗口标题看起来很正常，不需要每次都回复
- 如果有有趣的话题可以聊，比如用户打开了Steam，可以问“你要玩游戏了吗？”
- 回复要简短自然，1-2句话以内
- 绝对不要在每次回复时都说出当前的具体时间。只在用户主动询问时间时回答。
"""
        try:
            response = await self.agent_engine._llm.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
system="你是园神凛祢(そのがみりんね)，正在陪伴用户，具有桌面主动观察能力和长期记忆。",
                tools=None
            )
            full_text = ""
            async for chunk in response:
                if isinstance(chunk, dict) and chunk.get("type") == "text_delta":
                    full_text += chunk.get("text", "")
                elif isinstance(chunk, str):
                    full_text += chunk

            json_match = re.search(r'\{.*\}', full_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            print(f"AI决策失败: {e}")

        return {"should_speak": False, "message": ""}
