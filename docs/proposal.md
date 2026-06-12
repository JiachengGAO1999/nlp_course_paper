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

## 历史压缩条件

主实验比较五个条件：

1. Full History；
2. Sliding Window；
3. User-only History；
4. Oracle Dialogue Summary；
5. Oracle Fact-State Summary。

除 Full History 作为无压缩质量/成本上界外，其余条件均控制在相近
compressed-history budget 下，计划约为 600 tokens，允许 500-800 tokens。

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
- phenomenon-wise performance；
- accuracy-cost trade-off。

## 预期贡献

课程论文层面的贡献包括：

1. 构造一个小规模、可复现、可诊断的 synthetic testbed；
2. 在 budget-aware setting 下系统比较多种历史压缩策略；
3. 分析压缩策略导致的错误类型，而不只报告 accuracy；
4. 为后续博士研究中的 multi-turn reasoning reliability 和 inference-time
   context/state control 提供一个可控实验原型。

## 局限性

- 数据为 synthetic diagnostic testbed，不代表真实多轮对话分布；
- Oracle summaries 是理想化压缩，不能直接等同于真实自动摘要；
- 主实验规模较小，主要用于课程论文和诊断性分析；
- 后续需要通过 LLM-generated summaries、真实/半真实历史、更多模型和预算
  档位检验外部有效性。
