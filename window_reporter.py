import pygetwindow as gw
import time
import os

REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "current_window.txt")
CHANGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "window_changed.txt")

print(f"窗口报告员已启动，正在向 {REPORT_FILE} 写入数据...")
last_title = ""
while True:
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            title = active_window.title
            if title and title != last_title:
                # 保存当前窗口标题
                with open(REPORT_FILE, 'w', encoding='utf-8') as f:
                    f.write(title)
                print(f"已记录窗口: {title}")

                # 新增：写换行符表示“变了”
                with open(CHANGE_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"{title}\n{last_title}")

                last_title = title
    except Exception as e:
        print(f"出错了: {e}")
    time.sleep(2)