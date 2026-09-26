# 游戏资源重建工具（可选）

**正常安装不需要运行这个工具。** 项目已经包含五套游戏服装，以及四套自制服装。

只有想用自己电脑上的游戏文件重新生成资源时，才需要这里的程序。它不会下载、上传或改写游戏源文件。

## 两种用法

- `build`：从游戏 PCK 文件生成运行资源，再安装。默认处理第 1 套，也可以选择第 2–4 套。
- `install`：安装已经生成的运行包，入口文件是 `first-outfit-manifest.json`。也可以只引用原位置，不复制大文件。

转换所需的 SDK 已随项目提供，不需要另外下载。

## 资源保存位置

默认安装到：

```text
%LOCALAPPDATA%\Open-LLM-VTuber-Rinne\game-assets\mp_summer_uniform
```

把上面的路径粘贴到文件资源管理器地址栏，即可打开对应位置。

项目的 `local_config\rinne_game_assets.json` 记录资源位置。客户端设置保存在：

```text
%APPDATA%\open-llm-vtuber\rinne-legacy-renderer.json
```

## 查看操作方法

完整步骤见[游戏资源重建说明](../../public_docs/GAME_ASSET_SETUP.md)，包括 PowerShell、CMD、手动配置、检查和移除。

只想查看命令帮助，可以在项目目录执行：

```text
uv run python setup_rinne_game_assets.py --help
uv run python setup_rinne_game_assets.py build --help
uv run python setup_rinne_game_assets.py install --help
```
