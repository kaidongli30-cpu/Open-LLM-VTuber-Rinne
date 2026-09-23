# 在 QQ 与凛祢对话

先按主 [README](../README.md) 完成后端、语音和桌面客户端安装，确认桌面端能回复。QQ 有两种接法：官方 QQ Bot 适合纯文字私聊；NapCat／AstrBot 接个人 QQ 小号，支持文字、语音、图片和文档，发送单独一条 `over` 才会让凛祢回复。两种方式都使用同一套凛祢后端和本地记忆，但要分别配置，不能把官方 Bot 的 OpenID 当作个人 QQ 号。

以下命令都在 **PowerShell** 中执行。个人 QQ 通道的监护脚本需要 [PowerShell 7](https://learn.microsoft.com/powershell/scripting/install/installing-powershell-on-windows)；安装后，在项目文件夹地址栏输入 `pwsh` 并回车。路径 `D:\RinneQQ` 只是示例；换成自己有空间的文件夹。不要把账号、密码、Token、二维码、聊天记录或运行数据放进 Git 仓库。

## 方式一：QQ 官方机器人

到 [腾讯官方 SDK 页面](https://github.com/tencent-connect/qqbot-agent-sdk)创建自己的 QQ Bot，或用下方扫码启动器绑定。它只接收白名单账号的一对一文字消息；默认不主动发消息，不接收群聊、语音和文件。

在项目根目录执行：

```powershell
uv pip install --python .\.venv\Scripts\python.exe "qqbot-agent-sdk==1.2.2"
$env:RINNE_DATA_ROOT = (Get-Location).Path
.\.venv\Scripts\python.exe .\tools\start_rinne_qq.py
```

`RINNE_DATA_ROOT` 指向存放 `chat_history` 的目录。沿用项目内记忆时用上例；如果桌面端已经把它指向另一个目录，QQ 端也填写**同一个目录**。首次运行会打开绑定页面；用手机 QQ 按页面提示完成后，本次后端自动启动。关掉窗口后，本次绑定凭据不会留在项目文件里。已有完整的 `RINNE_QQ_APP_ID`、`RINNE_QQ_CLIENT_SECRET`、`RINNE_QQ_ALLOWED_USER_OPENIDS` 环境变量时，启动器可跳过扫码；密钥不要写入脚本。`uv sync` 之后若提示缺少 QQ SDK，重新执行上面的 `uv pip install`。

如果桌面后端已经占用 `12393`，先按 `Ctrl+C` 停掉桌面后端，再启动这个官方 Bot 入口；它会重新启动同一个后端。手机发文字，收到凛祢回复后即完成验收。

## 方式二：个人 QQ 小号

这条通道需要自己的小号登录 NapCat，主号与小号私聊。NapCat 与 AstrBot 是独立软件，不包含在本仓库。先阅读 [NapCat Windows 安装说明](https://napcat.napneko.icu/guide/boot/Shell.html)、[AstrBot 安装说明](https://docs.astrbot.app/deploy/astrbot/package.html) 和 [AstrBot 的 OneBot 连接说明](https://docs.astrbot.app/platform/aiocqhttp.html)。QQ 登录可能触发平台安全验证，不能通过本项目绕过。

### 1. 准备运行目录

在 PowerShell 中选择不在项目仓库内的目录：

```powershell
$qqHome = 'D:\RinneQQ'
$qqRuntime = Join-Path $qqHome 'runtime'
$qqData = Join-Path $qqHome 'data'
New-Item -ItemType Directory -Path $qqRuntime,$qqData -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $qqData 'qq_private') -Force | Out-Null
```

把 NapCat 的 Windows 无头包安装到 `$qqRuntime\napcat-node`，确认该文件夹**直接**包含 `node.exe` 和 `index.js`。首次按 NapCat 的说明启动和扫码，设置本机 WebUI 端口 `6099`，让 NapCat 在 `$qqData\napcat\config` 生成 `webui.json` 和当前小号的配置。不要公开 WebUI 端口。

用 `uv` 安装 AstrBot，放到本项目启动器约定的目录：

```powershell
$env:UV_TOOL_DIR = Join-Path $qqRuntime 'astrbot-tool'
$env:UV_TOOL_BIN_DIR = Join-Path $qqRuntime 'bin'
uv tool install astrbot --python 3.12
$astrbotExe = Join-Path $qqRuntime 'astrbot-tool\astrbot\Scripts\astrbot.exe'
Test-Path -LiteralPath $astrbotExe
```

最后一行应显示 `True`。然后建立 AstrBot 实例：

```powershell
$astrbotInstance = Join-Path $qqRuntime 'astrbot-instance'
New-Item -ItemType Directory -Path $astrbotInstance -Force | Out-Null
Set-Location $astrbotInstance
& $astrbotExe init
& $astrbotExe run
```

首次运行时打开 `http://127.0.0.1:6185`，按照 AstrBot 窗口提示登录后台并修改初始密码。按其 OneBot 说明创建适配器，反向 WebSocket 监听地址填 `127.0.0.1`、端口填 `6199`；NapCat 网络配置创建 WebSocket 客户端，连接 `ws://127.0.0.1:6199/ws`。如果设置了 OneBot Token，两边必须填相同值。看到 AstrBot 提示适配器已连接后继续。

### 2. 配置凛祢插件

停下首次运行的 AstrBot，回到凛祢项目文件夹，将插件复制进去：

```powershell
Set-Location '你的凛祢项目文件夹绝对路径'
$pluginParent = Join-Path $astrbotInstance 'data\plugins'
New-Item -ItemType Directory -Path $pluginParent -Force | Out-Null
$pluginTarget = Join-Path $astrbotInstance 'data\plugins\astrbot_plugin_rinne_private'
New-Item -ItemType Directory -Path $pluginTarget -Force | Out-Null
Copy-Item -Path .\integrations\astrbot_plugin_rinne_private\* -Destination $pluginTarget -Recurse -Force
```

再次运行 `& $astrbotExe run`，在 AstrBot 后台启用插件，并填写：

- `allowed_sender_id`：唯一允许与小号对话的**主 QQ 号**，只填数字；
- `data_root`：`D:\RinneQQ\data\qq_private`，改成自己的实际路径；
- `bridge_url`：`http://127.0.0.1:12394/agent/private/v1/turn`。

插件配置会保存在 `$astrbotInstance\data\config\astrbot_plugin_rinne_private_config.json`。确认该文件存在，再停止手动运行的 AstrBot 和 NapCat。小号登录数据、主号白名单、媒体缓存、聊天输入和日志都留在自己电脑上。

### 3. 启动与使用

回到凛祢项目根目录，运行：

```powershell
.\tools\start_private_qq.ps1 -RuntimeRoot $qqRuntime -DataRoot $qqData -OpenRecoveryPage
```

启动器会在本机运行独立的 QQ 后端 `12394`、AstrBot `6185/6199`、NapCat `6099` 和扫码页 `12395`；不会占用桌面后端的 `12393`。默认只在本机打开扫码页，不建立公网隧道。出现 `ready` 后，用主号给小号发消息，再单独发一条 `over`。`/status` 看缓存条数；`/cancel` 清空尚未提交的内容；`/end` 封存本次 QQ 聊天，下次消息另开记录。

如果桌面端用 `RINNE_DATA_ROOT` 指向了独立的记忆目录，在启动命令后加 `-ConversationDataRoot '你的绝对记忆目录'`，让 QQ 与桌面共用同一份记忆；QQ 运行缓存仍由 `-DataRoot` 单独保存。停用时执行：

```powershell
.\tools\stop_private_qq.ps1 -DataRoot $qqData
```

`restart_private_qq_brain.ps1 -DataRoot $qqData` 只重启 QQ 后端和 AstrBot，不停止 NapCat 登录。要在登录 Windows 后自动监护，先确认手动启动正常，再运行 `register_private_qq_guardian.ps1 -RuntimeRoot $qqRuntime -DataRoot $qqData`；用 `unregister_private_qq_guardian.ps1` 移除该计划任务。

如果你自己已配置 Cloudflare Tunnel 和 HTTPS 域名，才在启动命令中**额外**加入 `-EnableRemoteRecoveryTunnel -CloudflaredRoot '你的 cloudflared 目录' -RecoveryPublicBaseUrl 'https://你的域名'`。该目录须包含 `cloudflared.exe` 和你的 `config.yml`。远程扫码页仅用于恢复 QQ 登录，不提供聊天服务；不要把 NapCat WebUI 或 AstrBot 后台暴露到公网。没有这些配置时不要加该选项。
