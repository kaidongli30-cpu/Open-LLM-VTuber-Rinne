# Open-LLM-VTuber-Rinne

这是一个以 **Open-LLM-VTuber** 为基础整理出的凛祢桌宠版本。它保留了当前可运行状态中的 Live2D、云端 LLM 对话、语音识别、GPT-SoVITS 日语语音、表情参考音频，以及“聊天记录 + 日记 + 周记 + 月记”长期记忆流程。

本仓库不包含任何作者的 API Key、私人聊天记录、私人日记/周记/月记或个人电脑绝对路径。第一次安装时，你需要填入自己的 API、GPT-SoVITS 路径和模型权重。

> 本教程目前以 **Windows 10/11 64 位** 为主要验收环境。macOS/Linux 仍可参考上游文档，但本项目的 V4-B-R1 语音管理批处理以 Windows 为主。

## 你会得到什么

- 凛祢 Live2D 模型、头像与表情映射
- 浏览器聊天界面，以及连接上游桌面客户端后使用透明桌宠模式的能力
- OpenAI-compatible 云端 LLM 对话
- 本地 SenseVoice 语音识别（首次使用时可能自动下载模型）
- GPT-SoVITS V4-B-R1 日语语音；V4 异常时自动回退至 V2
- 中文回复先经本地 Ollama 翻译成日语，再交给 GPT-SoVITS
- 聊天记录、日记、周记、月记全部作为云端 LLM 上下文的当前稳定记忆方案
- 局域网或 HTTPS 隧道访问，方便手机聊天

## 仓库内容说明

本仓库直接公开后端代码、记忆系统、配置模板、Live2D 运行资源、参考音频和语音管理脚本。

`Open-LLM-VTuber-Web` 的前端源码不在本仓库重复发布。`frontend` 是一个 Git submodule（子模块，可以理解为“指向另一个 Git 仓库特定版本的引用”），指向上游公开的前端 **build 分支**。后端需要这些构建好的网页文件，因此克隆时必须带 `--recursive`。

桌面 `.exe` 也不在本仓库重新打包。需要透明桌宠模式时，请从上游 [Open-LLM-VTuber Releases](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/releases) 下载 Windows 安装程序；只使用浏览器时，不需要安装 `.exe`。

## 一、安装前确认

### 1. 建议硬件

- Windows 10/11 64 位
- 至少 16 GB 内存更稳妥
- 至少预留 15 GB 磁盘空间（GPT-SoVITS 整合包、依赖与模型会占用较多空间）
- 推荐 NVIDIA 显卡运行 GPT-SoVITS；无 CUDA 时可以使用 CPU，但语音生成会明显变慢
- 麦克风和扬声器/耳机

### 2. 检查前置软件

打开 PowerShell，逐条执行：

```powershell
git --version
uv --version
ffmpeg -version
```

如果命令能正常显示版本，说明程序已进入 PATH。

#### Git

从 [Git for Windows](https://git-scm.com/download/win) 安装，或执行：

```powershell
winget install --id Git.Git -e
```

安装后关闭并重新打开 PowerShell。

#### uv 与 Python

本项目要求 Python `>=3.10,<3.13`，推荐用 uv 安装 Python 3.12，不需要自己维护虚拟环境。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.12
```

uv 的官方安装说明见 [Astral uv 文档](https://docs.astral.sh/uv/getting-started/installation/)。

#### FFmpeg

可以先查找可用的软件包，再安装：

```powershell
winget search ffmpeg
winget install --id Gyan.FFmpeg -e
```

也可以从 [FFmpeg 官方下载页](https://ffmpeg.org/download.html) 选择 Windows 构建。安装后再次运行 `ffmpeg -version`。

#### Ollama（完整日语语音需要）

从 [Ollama Windows 下载页](https://ollama.com/download/windows) 安装。安装完成后打开新的 PowerShell：

```powershell
ollama --version
ollama pull qwen3.5:4b-q4_K_M
ollama list
```

当前 `conf.yaml` 默认启用 Ollama 翻译。如果只想先验证文字聊天，可以临时把 `translator_config` 下的 `translate_audio: True` 改为 `False`；这时中文回复不会先翻译成日语，不建议把它当作最终语音效果。

## 二、下载项目并安装 Python 依赖

不要使用 GitHub 的 “Download ZIP”，因为 ZIP 不会自动取得 `frontend` 子模块。

```powershell
cd D:\你准备存放项目的目录
git clone --recursive https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne.git
cd Open-LLM-VTuber-Rinne
git submodule status
uv sync
```

`git submodule status` 应当能看到 `frontend` 的提交号。若 `frontend` 目录为空：

```powershell
git submodule sync --recursive
git submodule update --init --recursive
```

运行本仓库自带的安装检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_rinne_installation.ps1 -SkipVoice
```

此时语音尚未安装，因此先使用 `-SkipVoice`。脚本只判断 API 环境变量是否存在，不会输出 Key 内容。

## 三、配置云端 LLM API

你需要一个提供 OpenAI-compatible `/chat/completions` 接口的 API。请从你的服务商后台取得：

- Base URL，例如 `https://服务商地址/v1`
- API Key
- 模型名

### 推荐方式：私有启动脚本

复制模板：

```powershell
New-Item -ItemType Directory -Force .\local_config
Copy-Item .\config_templates\start_rinne_private.example.ps1 .\local_config\start_rinne_private.ps1
notepad .\local_config\start_rinne_private.ps1
```

把其中占位内容替换成自己的值：

```powershell
$env:RINNE_LLM_BASE_URL = "https://你的服务商/v1"
$env:RINNE_LLM_API_KEY = "你的私人 API Key"
$env:RINNE_LLM_MODEL = "服务商提供的模型名"

$env:DIARY_LLM_API_URL = "https://你的服务商/v1/chat/completions"
$env:DIARY_LLM_API_KEY = $env:RINNE_LLM_API_KEY
$env:DIARY_LLM_MODEL = $env:RINNE_LLM_MODEL
```

区别是：主对话的 `RINNE_LLM_BASE_URL` 通常写到 `/v1`；日记生成器的 `DIARY_LLM_API_URL` 要写完整的 `/v1/chat/completions`。

`local_config` 已被 `.gitignore` 排除，不会进入 Git。不要截图、上传或把这个文件发给别人。周记和月记默认沿用日记 API；需要单独服务商时，再设置 `MEMORY_LLM_API_KEY`、`MEMORY_LLM_API_URL`、`MEMORY_LLM_MODEL`。

### 备选方式：直接编辑 conf.yaml

也可以把 `conf.yaml` 中 `openai_compatible_llm` 的三个 `${RINNE_LLM_*}` 占位符替换为自己的配置。不过这种方式更容易误提交 Key，因此只建议不会再上传该副本的普通使用者采用。

## 四、安装 GPT-SoVITS 与凛祢语音

### 1. 下载 GPT-SoVITS 整合包

从 [GPT-SoVITS 官方仓库](https://github.com/RVC-Boss/GPT-SoVITS) 下载 Windows 整合包。本项目当前按 `GPT-SoVITS-v2pro-20250604` 的目录结构验收。

推荐解压到项目目录，使最终结构为：

```text
Open-LLM-VTuber-Rinne\
└─ GPT-SoVITS-v2pro-20250604\
   └─ GPT-SoVITS-v2pro-20250604\
      ├─ runtime\python.exe
      ├─ api_v2.py
      └─ GPT_SoVITS\
```

如果你已有一份可运行的 GPT-SoVITS，不必复制。在 `local_config\start_rinne_private.ps1` 中设置实际路径：

```powershell
$env:RINNE_GPT_SOVITS_ROOT = "D:\你的路径\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604"
```

### 2. 下载四个凛祢权重

权重文件超过 GitHub 普通 Git 单文件限制，因此放在独立资源仓库的 Release：

[下载 Open-LLM-VTuber-Rinne-Assets / rinne-voice-v1](https://github.com/kaidongli30-cpu/Open-LLM-VTuber-Rinne-Assets/releases/tag/rinne-voice-v1)

下载后放到 GPT-SoVITS 根目录对应位置：

```text
GPT_weights_v2\rinne_e15.ckpt
SoVITS_weights_v2\rinne_e8_s456.pth
GPT_weights_v4\rinne_v4-e10.ckpt
SoVITS_weights_v4\rinne_v4_e1_s1297_l32.pth
```

四段运行参考音频已经随本仓库放在：

```text
Rinne_model\rinne_voice_runtime_bundle\emotion_references\
```

不需要再次下载原始游戏音频或训练数据。

### 3. 检查并启动语音服务

如果 GPT-SoVITS 位于默认目录，双击：

```text
Rinne_model\rinne_voice_runtime_bundle\检查保留文件与当前模式.bat
Rinne_model\rinne_voice_runtime_bundle\切换到V4-B-R1.bat
```

如果 GPT-SoVITS 在外部目录，请从已经设置 `$env:RINNE_GPT_SOVITS_ROOT` 的 PowerShell 启动：

```powershell
& ".\Rinne_model\rinne_voice_runtime_bundle\检查保留文件与当前模式.bat"
& ".\Rinne_model\rinne_voice_runtime_bundle\切换到V4-B-R1.bat"
```

V4-B-R1 会在本机启动：

- V4 后端：`127.0.0.1:9883`
- 稳定 V2 回退：`127.0.0.1:9882`
- 桌宠统一访问的代理：`127.0.0.1:9880`

检查健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:9880/health
```

无 NVIDIA/CUDA 时，在私有启动脚本中取消下面两行的注释：

```powershell
$env:RINNE_TTS_DEVICE = "cpu"
$env:RINNE_TTS_HALF = "false"
```

CPU 模式会慢很多。停止本项目启动的语音服务时，双击 `停止凛祢语音服务.bat`。

## 五、第一次启动并开始聊天

### 1. 完整安装检查

先在私有启动脚本所在配置下设置好变量，或在当前 PowerShell 手动设置，然后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_rinne_installation.ps1
```

### 2. 启动顺序

1. 确认 Ollama 正在运行，且 `ollama list` 中存在 `qwen3.5:4b-q4_K_M`。
2. 启动 `切换到V4-B-R1.bat`，确认 `http://127.0.0.1:9880/health` 正常。
3. 在项目根目录运行私人启动脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\local_config\start_rinne_private.ps1
```

4. 等终端显示服务启动后，在浏览器打开：

```text
http://127.0.0.1:12393
```

5. 第一次加载语音识别或其他本地模型时会等待下载，请观察后端终端，不要立即关闭。

### 3. 验收清单

按顺序确认：

- 页面能显示凛祢 Live2D，而不是空白或默认角色
- 输入文字后，云端 LLM 能返回回复
- 回复字幕正常，能够听到日语语音
- 点击麦克风并授权后，语音能被识别为文字
- `chat_history` 下产生新的本地会话数据
- 达到日记/周记/月记生成条件后，文件只写入本机；API 请求成功
- 关闭并重新启动后，历史会话仍能继续，记忆上下文能够进入新对话

如果只验证“能开始聊天”，前两项通过就表示 LLM 主链路已打通；其余项目用于确认完整桌宠体验。

## 六、安装 Windows 桌宠 `.exe`

1. 打开上游 [Open-LLM-VTuber Releases](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/releases)。
2. 下载 Windows 的 `open-llm-vtuber-...-setup.exe`。
3. 按安装向导完成安装。
4. 保持本仓库后端运行在 `127.0.0.1:12393`。
5. 打开桌面客户端，在连接设置中使用本机后端地址。
6. 普通窗口聊天通过后，再切换到透明背景/桌宠模式。

`.exe` 是桌面外壳，LLM、记忆、TTS 和角色配置仍由本仓库后端提供。若桌面客户端版本与本仓库前端不兼容，先用浏览器完成验收，并尝试上游 Releases 中与 v1 后端接近的版本。

## 七、手机部署

### 方案 A：同一局域网，先验证页面与文字聊天

电脑和手机连接同一个路由器。电脑执行：

```powershell
ipconfig
```

找到电脑的 IPv4 地址，例如 `192.168.1.20`。允许 Windows 防火墙中 Python/本项目通过专用网络后，手机打开：

```text
http://192.168.1.20:12393
```

因为浏览器通常只允许在 `https` 或 `localhost` 安全环境使用麦克风，所以局域网 HTTP 更适合先验证页面和文字聊天；手机语音请用下面的 HTTPS 方案。

### 方案 B：Cloudflare Quick Tunnel，适合首次测试

安装 [cloudflared](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/)，保持后端运行，然后执行：

```powershell
cloudflared tunnel --url http://localhost:12393
```

终端会显示一个随机 `https://xxxx.trycloudflare.com` 地址。用手机打开该地址；若前端要求单独填写服务器地址：

- Base URL：`https://xxxx.trycloudflare.com`
- WebSocket：`wss://xxxx.trycloudflare.com/client-ws`

手机第一次进入页面后点击一次页面，并允许麦克风权限，这也能解除移动浏览器的自动播放限制。

Quick Tunnel 是临时测试通道：地址每次可能改变，终端关闭后失效，不适合长期公开服务。官方说明见 [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。

### 方案 C：固定域名的 Named Tunnel

长期使用需要 Cloudflare 账号及一个接入 Cloudflare 的域名：

```powershell
cloudflared tunnel login
cloudflared tunnel create rinne
cloudflared tunnel route dns rinne rinne.你的域名.com
```

在 `%USERPROFILE%\.cloudflared\config.yml` 写入创建命令返回的 Tunnel ID 和凭据路径：

```yaml
tunnel: 你的-Tunnel-ID
credentials-file: C:\Users\你的用户名\.cloudflared\你的-Tunnel-ID.json

ingress:
  - hostname: rinne.你的域名.com
    service: http://localhost:12393
  - service: http_status:404
```

启动：

```powershell
cloudflared tunnel run rinne
```

然后手机访问 `https://rinne.你的域名.com`。不要把 `.cloudflared` 中的凭据文件上传到 GitHub。

## 八、常见问题

### 页面空白或 frontend 目录为空

```powershell
git submodule sync --recursive
git submodule update --init --recursive
```

### API 报 401/403

检查 Key 是否有效、模型是否有权限，以及 `RINNE_LLM_BASE_URL` 是否只写到服务商要求的 API 根路径。不要在 Issue 中粘贴完整 Key。

### 日记生成失败但主对话正常

主对话和日记使用的 URL 形式不同。确认 `DIARY_LLM_API_URL` 是完整 `/chat/completions` 地址，并且三个 `DIARY_LLM_*` 变量已在启动后端的同一个 PowerShell 进程中设置。

### 9880 端口被占用

```powershell
Get-NetTCPConnection -LocalPort 9880 -ErrorAction SilentlyContinue
```

不要直接结束不明进程。先确认它是否是你已启动的 GPT-SoVITS；若是本项目语音管理器启动的实例，使用 `停止凛祢语音服务.bat`。

### 没有日语语音

依次检查：

```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:9880/health
```

同时查看后端终端是否出现翻译超时、权重缺失、参考音频路径错误或 CUDA 显存不足。

### 手机能打开但麦克风不可用

不要使用普通局域网 HTTP 做最终语音测试。改用 Cloudflare Tunnel 等 HTTPS 入口，并在浏览器地址栏权限中允许麦克风。

## 九、更新、备份与回退

更新前先备份自己的本地数据：

- `chat_history`
- 本地日记/周记/月记文件
- `local_config`
- 自己修改过的 `conf.yaml`

查看当前版本：

```powershell
git status
git log -1 --oneline
```

不要在有未提交私人改动时直接强制重置。若只想安全试用新版本，重新克隆到另一个目录，再把本地配置和记忆复制过去。

## 十、隐私与开源资源说明

- API Key 只应存在于你的本机环境变量、`local_config` 或本地配置中。
- 聊天记录和日记类文件默认属于用户私人数据，不应提交到公开仓库。
- 当前记忆方案会把聊天记录、日记、周记、月记发送给你配置的云端 LLM。使用前请自行确认服务商的隐私政策。
- Live2D 相关资源还受仓库内 `LICENSE-Live2D.txt` 约束。
- 项目代码沿用上游许可证，详见 `LICENSE.txt`。
- GPT-SoVITS、上游桌面客户端及其模型/资源各自遵循原项目许可；请在再分发前分别核对。

## 致谢与上游项目

- [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
- [Open-LLM-VTuber-Web](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber-Web)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- 安装教程结构参考：[Open-LLM-VTuber-ATRI](https://github.com/ZL-Tian/Open-LLM-VTuber-ATRI)

这个仓库是可继续研究的开源基线，不代表上游项目的官方角色发行版。
