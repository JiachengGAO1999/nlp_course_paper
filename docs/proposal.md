# 课程论文 Proposal

## 暂定题目

对话历史压缩对多轮推理问答可靠性的影响：一项小规模可控实验

## 研究背景

多轮 LLM 系统在实际使用中会不断积累用户消息、模型回复、局部状态、
偏好、约束和中间结论。当上下文预算逐渐紧张时，系统通常无法无限保留
完整历史，而必须选择某种历史压缩或保留策略。

问题在于，不同压缩方式可能保留不同类型的信息，也可能丢失状态更新、
否定约束、时间顺序或硬约束，从而影响模型后续回答的可靠性。本文关注
这一问题在轻量多轮推理问答中的表现。

## 研究问题

在整体上下文预算受限、完整历史不可持续保留的情境下，不同对话历史
压缩策略如何影响模型在后续多轮推理问答中的：

- 答案正确率；
- 输入 token 成本；
- 输出 reasoning/content 长度；
- 证据使用情况；
- 错误类型分布。

## 核心定位

本文不是泛泛比较几种 prompt 格式，也不是构造通用多轮对话 benchmark。
本文构造一个小规模 controlled synthetic diagnostic testbed，用于可控诊断：

> 在相同 compressed-history budget 下，不同历史压缩/保留策略会怎样影响
> 模型对多轮状态、约束、证据和干扰信息的使用？

## 数据设计

主实验使用英文 synthetic diagnostic testbed，课程论文用中文撰写。

每条样本包含：

- 6-8 轮历史对话；
- 完整历史约 1,500-2,500 tokens；
- 2-4 条 required evidence；
- 2-4 类诊断性 distractors；
- 3-4 个 multiple-choice options；
- 自动计算的 gold answer；
- 每个选项的 `option_diagnostics`。

四类诊断现象：

1. 分散事实整合；
2. 状态更新 / 覆盖；
3. 否定与排除约束；
4. 顺序 / 时间依赖。

为了避免 synthetic 数据过于简单，生成器将借鉴现有多轮评测中常见的失败
模式，但不直接把外部 benchmark 作为主数据集。具体包括：早期隐式约束保持、
需要中间推理才能使用的约束、长距离证据、未显式标注的 hard/soft preference
冲突、细微状态更新，以及分散候选属性整合。正式 pilot 前会先构造 hard smoke
v2，用少量高难样本校准 Qwen3-8B thinking 模型是否存在 ceiling effect。

## 历史压缩条件

主实验比较五个条件：

1. Full History；
2. User-only History；
3. Oracle Fact-State Summary；
4. LLM-generated Summary；
5. Hybrid Summary + Recent Turns。

除 Full History 作为无压缩质量/成本上界外，其余条件均控制在相近
compressed-history budget 下，计划约为 600 tokens，允许 500-800 tokens。

其中 `Sliding Window` 降级为 supplementary naive baseline，仅用于展示纯
recency deletion 的下界；`Oracle Dialogue Summary` 可在 pilot 中作为冗余
检查，如果它和 `Oracle Fact-State Summary` 高度重合，则不进入正式主分析。

## 模型与推理设置

主实验复用现有研究项目的服务器环境：

- model: Qwen3-8B；
- serving: vLLM OpenAI-compatible API；
- reasoning parser: Qwen3；
- temperature: 0.0；
- top_p: 1.0；
- thinking enabled；
- fixed thinking budget: 512 或 1024 tokens，优先从 512 开始。

回答格式统一为：

```text
Final Answer: <A/B/C/D>
Explanation: ...
```

Prompt 要求 brief evidence-based explanation，但不要求 step-by-step reasoning，
也不显式要求模型执行 verification。

## 评价指标

- Accuracy；
- input tokens；
- compressed-history tokens；
- compression ratio；
- output reasoning tokens / content tokens；
- parse success rate；
- error type distribution from `option_diagnostics`；
- required evidence retention；
- answerability after compression；
- hard/soft constraint preservation；
- stale-state handling；
- hallucinated fact count；
- phenomenon-wise performance；
- accuracy-cost trade-off。

## 预期贡献

课程论文层面的贡献包括：

1. 构造一个小规模、可复现、可诊断的 synthetic testbed；
2. 在 budget-aware setting 下比较规则状态压缩、LLM 摘要压缩和
   hybrid summary + recent turns 等策略；
3. 分析压缩产物的中间质量，包括关键证据保留、hard/soft constraint
   区分、stale state 处理和 hallucination，而不只报告 accuracy；
4. 为后续博士研究中的 multi-turn reasoning reliability 和 inference-time
   context/state control 提供一个可控实验原型。

## 实验推进

本文不直接从简单 smoke 扩展到正式实验，而是采用：

```text
basic smoke -> hard smoke v2 -> pilot -> formal
```

其中 hard smoke v2 用于检查任务是否过于简单。如果 Full History 条件仍接近
100% 正确率，则需要增加生成器难度；如果 Full History 明显低于合理上界，则
说明任务本身过难，应将这类样本作为 supplementary hard subset，而不是主实验
分布。

## 局限性

- 数据为 synthetic diagnostic testbed，不代表真实多轮对话分布；
- Oracle Fact-State Summary 是理想化规则压缩上界，不能直接等同于真实自动摘要；
- 主实验规模较小，主要用于课程论文和诊断性分析；
- 后续需要通过真实/半真实历史、更多模型、progressive summary、retrieval
  memory 和预算档位检验外部有效性。
