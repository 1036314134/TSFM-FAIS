# research-writing-skill 使用门槛

论文迭代 prompt 必须明确要求使用 `Norman-bury/research-writing-skill`。这不是普通润色建议，而是修改论文前必须执行的写作流程。

## 必需模块

- `using-research-writing`：确认任务属于论文写作，并选择后续模块。
- `paper-orchestration`：中型及以上改动使用，维护项目概览、章节结构、任务包或进度审计。
- `evidence-driven-writing`：涉及摘要、引言、相关工作、贡献、定位和引用驱动段落时使用。
- `writing-core`：涉及正文改写时使用，检查自然表达、去 AI 味、机械过渡词和过强 claim。
- `latex-output`：涉及 LaTeX 文件时使用，并在提交前编译。
- `figures-diagram`：涉及流程图、框架图、概念图时使用，生成 Gemini / Nano Banana / image2 可用提示词。
- `figures-python`：涉及数据图时使用，优先生成可复现的 Python 图表。
- `verification`：完成前使用，要求运行真实验证命令后才能宣称完成。

## Codex 与 Claude 的差异

- Claude Code 如果支持 Skill 工具，应优先通过 Skill 工具调用对应模块。
- Codex CLI 如果不能直接调用该 skill，应读取本地 `RESEARCH_WRITING_SKILL_REPO` 或常见安装目录下的对应 `SKILL.md`，并按其中流程执行。
- 如果找不到 `research-writing-skill`，本轮必须停止，不能用普通 prompt 继续改论文。

## prompt 中必须写清楚的事项

- 本轮需要哪些模块。
- 每个模块对应读取的文件路径或调用方式。
- 哪些 evidence map、coverage note、chapter blueprint、task packet 或 progress audit 需要更新。
- 哪些验证命令必须运行。
- commit body 中如何记录 capability-use audit。

## 禁止替代

- 不能只说“参考 research-writing-skill”，却不要求读取或调用。
- 不能只用通用写作建议替代 evidence-driven-writing。
- 不能把概念图提示词当成真实数据图。
- 不能在未运行验证命令时宣称编译、图表或实验已经完成。
