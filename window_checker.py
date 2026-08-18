import pygetwindow as gw
import time

print("窗口检测器已启动，每5秒检测一次...")
last_title = ""
while True:
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            title = active_window.title
            if title and title != last_title:
                print(f"当前活动窗口: {title}")
                last_title = title
    except Exception as e:
        print(f"出错了: {e}")
    time.sleep(5)