# CHANGELOG

本仓库改动按 **Git First-Parent 提交顺序倒序**记录，PR 与自有提交统一排列；仅记录实际完成的变更。

## 2026-10-19

| Commit | 来源 | 具体改动 |
| --- | --- | --- |
| 本次提交 | 自有 | 建立统一的 CHANGELOG，后续随代码变更持续更新。 |
| `567cb8d` | [PR #159](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/pull/159) | `convert_utils.py`：GLB 保存日志改为英文，避免部分终端的中文编码异常。 |
| `39acee4` | [PR #201](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/pull/201) | 6 个文件、10 处 `except:` → `except Exception:`，避免吞掉中断及退出异常。 |
| `0892686` | [PR #217](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/pull/217) | `pipelines.py`：meta device + `assign=True` 跳过无用权重初始化，优化模型加载时间；保留常规加载回退。 |
| `da09704` | [PR #215](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/pull/215) | Linux 依赖：`bpy` 可选及 trimesh GLB 回退；`setuptools<81`；分离 `deepspeed`；移除默认 CuPy / 额外镜像；CUDA 扩展 `--no-build-isolation`。 |
| `4aea7a7` | [PR #207](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/pull/207) | `Pipeline.to()` 返回 `self`，支持链式调用；以独立修复提交纳入。 |
| `dadc79a` | 自有 | 新增 `.gitmessage`，规范 Git Commit 消息。 |

## 维护约定

- 每次修改代码时，同步更新本文件并一起提交；新记录置于同日记录顶部。
- 每行包含 Commit、PR 来源（或自有）与具体改动；不写未实施的计划。
- “本次提交”代表更新本文件所在的 Commit，可由 `git log -1` 查询。
- 只有完成实测才记录性能结论，并注明环境与数据。
