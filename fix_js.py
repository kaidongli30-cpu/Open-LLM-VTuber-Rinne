# fix_js.py - 修复摄像头镜像问题
path = "frontend/assets/main-nu7uwxNJ.js"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 找到翻转相关的CSS
old = 'scaleX(-1)'
count = content.count(old)
print(f"找到 {count} 处 scaleX(-1)")

new = 'scaleX(1)'
content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("替换完成！")