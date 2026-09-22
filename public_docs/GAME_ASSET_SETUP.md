# 用自己的游戏文件启用游戏原画凛祢

公开仓库和发布包不包含凛祢的游戏原文件、转换纹理、语音片段或生成后的运行包。导入器只在本机读取用户自己选择的文件，不联网上传，也不会修改游戏安装目录。

## 先理解三个目录

- **游戏 PCK 目录**：游戏安装目录下的 `Data\Data\Mp\1st`（第 1、2 套）或 `Mp\2nd`（第 3、4 套）。每套需要对应的 15 个 `MP060x01.pck` 至 `MP060x15.pck` 文件。
- **SDK 根目录**：项目已经附带自制的格式读取与转换程序，普通用户不需要另行下载 SDK。需要替换转换实现的开发者才会用到 `--sdk-directory`。
- **本地运行资源目录**：SDK 从 PCK 生成、供桌宠 WebGL 渲染器读取的文件。它不是 Live2D/Cubism 模型，也不能改名伪装成 `.model3.json`。

SDK 和桌宠程序都只读 PCK。默认生成后的大文件放在：

```text
%LOCALAPPDATA%\Open-LLM-VTuber-Rinne\game-assets\mp_summer_uniform
```

如需与另一套正在使用的凛祢并行，先在启动导入器和后端的同一个 PowerShell 窗口设置独立路径：

```powershell
$env:RINNE_GAME_ASSET_ROOT = 'D:\RinnePublicTest\game-assets'
$env:RINNE_RENDERER_SETTINGS_PATH = 'D:\RinnePublicTest\settings\rinne-legacy-renderer.json'
$env:RINNE_LEGACY_SETTINGS_PATH = $env:RINNE_RENDERER_SETTINGS_PATH
```

这些环境变量只对当前窗口及其启动的进程生效。桌面客户端也应从这个窗口启动，才能读取同一份独立设置。

## 方法 A：从原版 PCK 转换并安装

在项目根目录运行以下命令，默认转换第 1 套服装：

PowerShell：

```powershell
uv run python setup_rinne_game_assets.py build `
  'D:\Games\DATE A LIVE Rio Reincarnation\Data\Data\Mp\1st'
```

CMD：

```bat
uv run python setup_rinne_game_assets.py build "D:\Games\DATE A LIVE Rio Reincarnation\Data\Data\Mp\1st"
```

不想手输路径时可以打开两个文件夹选择窗口：

```powershell
uv run python setup_rinne_game_assets.py build --gui
```

要让换装菜单显示其他游戏服装，分别执行：

```powershell
uv run python setup_rinne_game_assets.py build 'D:\Games\DATE A LIVE Rio Reincarnation\Data\Data\Mp\1st' --outfit-number 2
uv run python setup_rinne_game_assets.py build 'D:\Games\DATE A LIVE Rio Reincarnation\Data\Data\Mp\2nd' --outfit-number 3
uv run python setup_rinne_game_assets.py build 'D:\Games\DATE A LIVE Rio Reincarnation\Data\Data\Mp\2nd' --outfit-number 4
```

每次成功安装会增加一个本地服装入口，不会删掉先前安装的入口；项目仓库仍不包含从游戏生成的任何运行资源。

转换会逐个校验并生成 15 个肖像，耗时和磁盘占用都明显高于普通安装。中途失败时，导入器删除未完成的临时输出，不改动 PCK，也不会覆盖已有安装。

如果目标目录已经存在，命令默认停止。确认要用新生成包替换旧包时加 `--replace`。替换先把旧目录移到同盘临时备份，只有新包通过完整校验后才删除备份。

## 方法 B：导入已经生成的运行包

如果已经按 SDK 流程取得包含 `first-outfit-manifest.json`、`MP060101` 至 `MP060115` 的目录：

PowerShell：

```powershell
uv run python setup_rinne_game_assets.py install 'D:\RinneAssets\first-outfit'
```

CMD：

```bat
uv run python setup_rinne_game_assets.py install "D:\RinneAssets\first-outfit"
```

或者使用选择窗口：

```powershell
uv run python setup_rinne_game_assets.py install --gui
```

默认会复制到本地应用数据目录。已有运行包很大、不想再复制一份时，可以让桌宠直接读取原位置：

```powershell
uv run python setup_rinne_game_assets.py install `
  'D:\RinneAssets\first-outfit' `
  --use-in-place
```

就地引用的目录不由导入器管理，执行移除命令时绝不会删除它。

## 检查是否安装完整

完整检查会验证清单、文件长度和每个大文件的 SHA-256：

```powershell
uv run python setup_rinne_game_assets.py status
```

只想快速检查目录结构和长度时：

```powershell
uv run python setup_rinne_game_assets.py status --fast
```

成功后必须完全退出桌面客户端再重新打开。仅刷新网页不能让 Electron 主进程重新加载本地资源设置。Live Mode 和 Pet Mode 使用同一套本地资源指针。

## 手动配置接口

自动导入失败但运行包已经通过 SDK 自带校验时，可以手动创建项目根目录下的 `local_config\rinne_game_assets.json`。该目录已被 Git 忽略，不要提交。

```json
{
  "version": 1,
  "assets": {
    "mp_summer_uniform": {
      "kind": "legacy_first_outfit",
      "path": "D:\\RinneAssets\\first-outfit",
      "managed_by_setup": false
    }
  }
}
```

再编辑 `%APPDATA%\open-llm-vtuber\rinne-legacy-renderer.json`：

```json
{
  "version": 2,
  "profile_id": "mp_summer_uniform",
  "renderer": "rinne",
  "outfit_id": "mp_summer_uniform",
  "outfit_dir": "D:/RinneAssets/first-outfit",
  "first_outfit_dir": "D:/RinneAssets/first-outfit",
  "asset_kind": "outfit"
}
```

Windows JSON 中反斜杠必须写成 `\\`；也可以像第二个示例一样使用 `/`。

## 移除与重建

只移除两处本地指针、保留所有大文件：

```powershell
uv run python setup_rinne_game_assets.py remove
```

同时删除由本工具复制的第 1 套生成数据：

```powershell
uv run python setup_rinne_game_assets.py remove --delete-generated-data --yes
```

其他服装可用 `--profile mp_red_white_ruffled_casual`、`mp_red_cardigan_brown_skirt` 或 `mp_dark_navy_winter_uniform` 指定。

删除前会核对隐藏安装标记。就地引用、手工目录、游戏安装目录和 SDK 目录都不会被删除。移除后桌面设置回到 `live2d`；若公开发行版没有安装任何用户自备 Live2D 模型，前端会显示缺少模型，而不会从仓库恢复游戏资源。

## 失败时先看什么

- “缺少 15 个 PCK”：选择的是错误层级；应直接选择包含 `MP060101.pck` 的 `Mp\1st`。
- “SDK 目录缺少导出器”：如果主动传入了 `--sdk-directory`，应选择含 `rinne_legacy_runtime` 和 `tools` 的 SDK 根目录；普通用户删除该参数即可使用项目自带转换器。
- “文件哈希不符”：运行包不完整或被改写，重新从只读 PCK 生成。
- “安装目录已存在”：先运行 `remove --delete-generated-data --yes`，或确认后用 `--replace`。
- 客户端仍显示旧形象：完全退出客户端进程，再重新启动；不要只关闭 Live Mode 页面。

导入器不会帮助下载游戏，不会绕过所有权验证，也不会把任何本地资源上传到项目维护者或第三方服务。
