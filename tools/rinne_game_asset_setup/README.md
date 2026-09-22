# Rinne 本地游戏资源导入器

这里的程序只负责把用户自己持有的游戏源文件接入公开版凛祢。程序不会下载、上传、改写或提交游戏资源。

项目已附带自制转换 SDK，不需要普通用户另行下载。支持两种输入：

- `build`：调用项目内附带的 `rinne_legacy_runtime` SDK，从用户选择的游戏 PCK 生成运行包并安装；默认第 1 套，也可选择第 2–4 套。
- `install`：导入 SDK 已经生成完成的 `first-outfit-manifest.json` 运行包；也可以只建立就地引用，不复制大文件。

默认安装到 `%LOCALAPPDATA%\Open-LLM-VTuber-Rinne\game-assets\mp_summer_uniform`，不进入 Git 仓库。项目内只生成一个被 `.gitignore` 排除的 `local_config\rinne_game_assets.json` 指针；桌面客户端设置写入 `%APPDATA%\open-llm-vtuber\rinne-legacy-renderer.json`。

完整 PowerShell、CMD、手动配置、状态检查和移除方法见 [`public_docs/GAME_ASSET_SETUP.md`](../../public_docs/GAME_ASSET_SETUP.md)。

快速查看帮助：

```powershell
uv run python setup_rinne_game_assets.py --help
uv run python setup_rinne_game_assets.py install --help
uv run python setup_rinne_game_assets.py build --help
```
