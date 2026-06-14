# 项目快速说明：Keep Recent or Keep Relevant?

这份说明面向第一次接触本项目的人。目标是用最少背景理解：这个研究在问什么、为什么重要、一次 run 的结果该怎么看。

## 1. 这个项目在研究什么？

大语言模型在多轮对话或 agent 任务中会积累很长的历史上下文。实际系统通常不能无限保留全部历史，所以会做“历史压缩”：

- 把之前的对话总结成较短摘要；
- 或者总结较早历史，同时保留最近一轮原文。

这个项目研究的问题是：

> 在固定压缩预算下，把最近轮次原文保留下来，是一种稳定有效的历史压缩策略，还是一种依赖证据位置的预算分配取舍？

我们的核心发现是：**它不是稳定收益，而是条件性 trade-off**。保留最近轮次能保护最近出现的关键证据；但如果真正有用的证据在较早轮次，它会挤占旧历史摘要预算，反而更容易丢失早期证据。

## 2. Motivation：为什么这个问题重要？

很多系统默认把“最近”当成“相关”：

```text
+ 最近一轮原文
+ 较早历史摘要
```

这个做法看起来合理，因为用户最近说的话往往重要。但在多跳问答、多轮检索、多轮推理里，答案所需证据可能出现在很早之前。

因此，本项目想诊断一种常见风险：

> recency 不等于 relevance。最近的信息不一定是最相关的信息。

这不是简单比较哪个 prompt 写得更好，而是在研究**固定上下文预算应该如何分配给历史信息**。

## 3. Research Question

主问题：

> 在相同压缩预算、相同压缩 prompt 下，最近轮次原文保留是否会因为关键证据位置不同而产生条件性收益和代价？

更具体地说：

- 当关键证据在最近窗口内，recent-verbatim retention 是否保护了证据？
- 当关键证据在较早历史中，recent-verbatim retention 是否挤占了旧历史摘要预算？
- 压缩失败主要来自证据遗漏、干扰项误导，还是模型推理错误？

## 4. 实验怎么做？

数据来自 MuSiQue 多跳问答。我们把原始开放问答转换成 4 选 1 multiple-choice QA。

每个样本被构造成一个 6-8 轮的“source-note collection dialogue”：

- 用户逐步提供 source notes；
- 关键证据被放在早期、中期、跨轮次或晚期；
- 还会加入一些 distractors；
- 最终问题和选项在对话阶段隐藏，只在最后回答时出现。

比较 3 种历史条件：

| Condition | 含义 | 作用 |
|---|---|---|
| `full_history` | 完整对话历史 | 上界：确认原始历史可答 |
| `one_shot_summary` | 全部历史一次性压缩 | 全局压缩 baseline；不显式偏向最近轮次 |
| `hybrid_summary_recent` | 旧历史摘要 + 最近一轮原文 | recency-biased 预算分配策略 |

两个压缩条件使用同一个压缩 prompt，压缩历史预算约 800 tokens。主要模型是 Qwen3-8B。

这里的 one-shot 和 hybrid 不是为了做“谁整体更强”的排行榜，而是两个受控对照：

- one-shot 表示把约 800 tokens 都给全局摘要；
- hybrid 表示把预算拆成“旧历史摘要 + 最近原文”。

两者的差异用来诊断 recency-biased budget allocation 的影响。

## 5. 这次 100-case run 的主结果

正式 run：100 个样本 × 3 个条件 = 300 次推理。

整体准确率：

| Condition | Correct / Total | Accuracy |
|---|---:|---:|
| `full_history` | 100 / 100 | 100% |
| `one_shot_summary` | 85 / 100 | 85% |
| `hybrid_summary_recent` | 86 / 100 | 86% |

只看整体，hybrid 只比 one-shot 高 1 个百分点，看起来差不多。

但按“关键证据是否在最近窗口内”分层后：

| Evidence placement | One-shot | Hybrid | Hybrid gap |
|---|---:|---:|---:|
| Critical evidence in recent window | 43 / 50 | 50 / 50 | +14pp |
| Critical evidence outside recent window | 42 / 50 | 36 / 50 | -12pp |

核心解读：

> hybrid 的 aggregate gap 不是小到没有意义，而是方向相反的两个效应被平均掉了。关键证据最近时，recent-verbatim retention 明显有利；关键证据较早时，它会因为旧历史摘要预算变小而带来损失。

## 6. 怎么看 run 目录？

如果你收到压缩包，重点看这些文件：

```text
runs/layer1_scale100_qwen3_8b_budget800_20260614/
  formal/
    formal_selected_dialogues.jsonl   # 100 个正式样本的对话
    formal_variants.jsonl             # 3 种历史条件构造后的输入
    formal_selection_audit.json       # 样本选择统计
    formal_variant_audit.json         # 历史长度统计
    22_critical_annotation.md         # 关键错误样本人工标注
    inference/
      summary.json                    # 主结果统计，先看这个
      generations.parsed.jsonl        # 每条推理的解析结果
      generations.raw.jsonl           # request + parsed 结果
```

最先读：

1. `formal/inference/summary.json`
2. `formal/22_critical_annotation.md`
3. `formal/formal_selection_audit.json`
4. `formal/formal_variant_audit.json`

## 7. generations.parsed.jsonl 里每行怎么看？

每一行是一条样本在一个条件下的模型回答。重要字段：

| 字段 | 含义 |
|---|---|
| `source_id` | 样本 ID |
| `condition` | 使用哪种历史条件 |
| `gold` | 正确选项 |
| `parsed_answer` | 模型解析出的选项 |
| `is_correct` | 是否答对 |
| `dialogue_profile` | 证据位置类型 |
| `critical_evidence_in_recent_turn` | 关键证据是否在 hybrid 最近窗口 |
| `history_tokens` | 历史表示长度 |
| `response_content` | 模型最终回答 |
| `response_reasoning` | Qwen3 thinking/reasoning 内容 |

注意：`response_reasoning` 是后来重跑 inference 后保存的。当前标准路径 `formal/inference/` 已经是带 reasoning 的版本。

## 8. 人工标注文件怎么看？

`22_critical_annotation.md` 只标注 compressed 条件里的关键失败样本。

主要看三类：

| Group | 含义 |
|---|---|
| `OS✗HY✓` | one-shot 错，hybrid 对 |
| `HY✗OS✓` | hybrid 错，one-shot 对 |
| `both✗` | 两个压缩条件都错 |

标注结论：

- `OS✗HY✓` 大多发生在关键证据最近；
- `HY✗OS✓` 和 `both✗` 都发生在关键证据不最近；
- 主失败机制是 evidence omitted，也就是压缩后答案关键证据丢失。

## 9. 一句话结论

这个项目说明：

> 最近轮次原文保留不是免费的提升。它是一种固定预算下的信息分配策略：保护最近证据，同时可能牺牲较早证据。多轮推理历史压缩更应该保留 relevant reasoning state，而不是默认保留 recent text。

## 10. 如果只想花 10 分钟了解

推荐阅读顺序：

1. 本文件。
2. `README.md` 的 Current Framing。
3. `paper/6min_nlp_class_report.md`。
4. run 压缩包里的 `formal/inference/summary.json`。
5. run 压缩包里的 `formal/22_critical_annotation.md`。
