# Generator Implementation Specification

本文档定义六个 benchmark-inspired 难度模式的规则化模板策略、Assistant 行为矩阵、
样本分配表，以及 `llm_generated_summary` 和 `hybrid_summary_recent` 的压缩 prompt
模板。

---

## 1. Difficulty Mode Template Strategies

### 1.1 implicit_constraint_tracking

**定义**：早期约束仅在其引入 turn 中出现一次，后续 turn 不再重复。

**生成参数**：

```yaml
constraint_repetition: once
  # 合法值: once | every_turn | every_other_turn
  # 当前 smoke 使用的是 every_turn（每条 user 消息都带元注释）
metadata_prompt: disabled
  # 关闭 "Some pieces may become stale..." 元注释
  # 仅在约束引入 turn 保留提示角色，后续 turn 中完全不提
```

**模板策略**：

- 约束在 `turn_1` 或 `turn_2` 中以自然对话方式引入一次
- 后续 user 消息讨论别的主题，不回头 reaffirm 该约束
- assistant 在后续 turn 中不做隐含提示（不写 "given the earlier constraint about X"）
- 约束仍然写入 `required_evidence`，只是关键词在后续对话中不再出现

**预期效果**：压缩策略如果丢失了早期 turn，该约束将完全不可用。

### 1.2 derived_constraint

**定义**：约束需要通过 1-2 步推理才能使用，而不是直接陈述。

**生成参数**：

```yaml
constraint_type: derived
reasoning_chain:
  - premise: "Ben's lab duty ends at 12:00 noon"
    turn: turn_1
    surface: "Ben told us he's stuck in the lab until noon every day this week."
  - premise: "Friday afternoon slots are 2:00-5:00 PM"
    turn: turn_3
    surface: "The only open slots Friday are the afternoon block, 2 to 5."
  - target_conclusion: "Ben can attend Friday afternoon"
    # 该结论不出现在对话中，但用于计算 gold answer
```

**模板策略**：

- 生成器从 world state 中的 `reasoning_chain` 提取 premise
- 只将 premise 渲染到对话中，不渲染 target_conclusion
- gold answer 仍然从 world state 计算（不受影响）
- 每个 `derived_constraint` 样本需要 2-3 个 premise 分布在 2+ turn 中

**实现约束**：

- reasoning_chain 由生成器在 world state 构建时填充
- 每条 chain 的 premise 之间有明确的逻辑依赖
- 丢失任一 premise 都应导致约束不可推导

### 1.3 long_distance_evidence

**定义**：answer-critical evidence 远离最终问题，中间填充大量冗余或无关内容。

**生成参数**：

```yaml
evidence_distance_profile: far_early
  # 合法值: far_early | far_middle | mixed
  # far_early: 证据在 turn_1-2，填充在 turn_3-5，问题在 turn_6-7
  # far_middle: 填充在 turn_1-2 和 turn_6-7，证据夹在 turn_3-5
  # mixed: 各样本随机选择
redundancy_profile: dense_at_middle
  # 合法值: dense_at_middle | uniform | sparse_at_ends
  # 控制 assistant_redundancy 和 topical_distractor 的密度分布
```

**模板策略**：

- `far_early`：turn_1-2 放置 3-4 条 required evidence（密集），turn_3-5 放置 2-3 条 long topical distractors + assistant redundancy（纯填充），turn_6-7 放置问题
- `far_middle`：turn_1-2 和 turn_6-7 是填充/介绍/收尾，turn_3-5 放置 required evidence
- 每种 profile 至少 1 个样本，剩余样本用 mixed

**与 `scattered_fact_integration` 的区别**：scattered_fact 要求证据分布在多个 turn 中且需要组合；long_distance 关注的是**绝对距离**（最早证据到问题的 token 数），可以且常常与 scattered_fact 叠加使用。

### 1.4 soft_hard_conflict_without_labels

**定义**：硬约束和软偏好用相似的自然语言表达，不显式标注类型。

**生成参数**：

```yaml
constraint_labeling: implicit
  # 合法值: explicit (当前 smoke) | implicit
  # implicit 模式下禁止 "must", "hard requirement", "only a preference" 等词
constraint_conflict: enabled
  # 同一维度上有 >=2 个约束，至少 1 个 hard + 1 个 soft
  # 例如：两个时间约束、两个房间约束，不同类型
```

**模板措辞对比**：

```
Explicit（当前）:
  "Ben cannot attend before noon — this is a hard constraint."
  "Ben prefers rooms near the elevator — this is only a preference."

Implicit（新）:
  "Ben has a lab commitment every morning until 12, so mornings are out."
  "Ben mentioned he finds rooms near the elevator much easier to work in."
```

**模板策略**：

- 硬约束用后果语言表达（"X is out"、"won't work"、"can't make it"）
- 软偏好用舒适/便利语言表达（"finds it easier"、"would rather"、"it'd be nice if"）
- 不给任何类型标签
- 在 `required_evidence` 中标记 `evidence_type` 以支持自动 evaluation，但这些标签不进入对话文本

**软硬冲突的最小配置**：每个此类样本需要**同一维度上至少一对冲突的 hard+soft 约束**。例如：时间维度的 hard constraint（Ben cannot do mornings）+ 时间维度的 soft preference（Ben prefers afternoons but can do mornings if needed 的反向情况）。

### 1.5 subtle_state_update

**定义**：用自然措辞更新早期状态，不使用 "this replaces the old value" 等显式标记。

**生成参数**：

```yaml
update_style: subtle
  # 合法值: explicit (当前) | subtle | mixed
  # subtle 模式下禁止 explicit replacement language
```

**模板措辞对比**：

```
Explicit（当前）:
  "The organizer moved the review from Wednesday to Friday afternoon.
   This update replaces the old Wednesday placeholder."

Subtle（新）:
  "Actually, let's do Friday afternoon instead — it works better
   with everyone's schedule."
  # → 没有 "replaces"，没有 "old placeholder"，没有 "update"
```

**模板策略**：

- 使用对话自然转折词："Actually..."、"On second thought..."、"Let's go with..."、"Turns out Friday is better because..."
- 不显式提旧值，除非需要通过对比来传达更新
- 旧值仍然在 `required_evidence` 中标记为 `stale_information`
- 验证方式：检查压缩策略是否保留了最新状态而非旧状态

### 1.6 scattered_candidate_attributes

**定义**：每个候选的属性分散在多个 turn 中，需要跨 turn 组装。

**与当前 `scattered_fact_integration` 的关键区别**：

```
当前（一个 turn 介绍一个候选的完整信息）:
  turn_3: Lina knows Python, has annotation experience, available Friday only.
  turn_5: Priya knows Python, available Thursday, has annotation experience.

改造后（一个候选的属性分布在多个 turn）:
  turn_1: The assistant must know Python.                          (约束)
  turn_2: The assistant must be available Thursday.                (约束)
  turn_3: Lina knows Python and has annotation experience.          (Lina 属性 1)
  turn_4: Priya is available Thursday and has annotation experience. (Priya 属性 1)
  turn_5: Lina is only available Friday, not Thursday.              (Lina 属性 2)
  turn_6: Priya knows Python too.                                   (Priya 属性 2)
  → 组装：Lina: Python✅ Thurs❌ Annotation✅ | Priya: Python✅ Thurs✅ Annotation✅
```

**生成参数**：

```yaml
candidate_attribute_distribution: scattered
  # 合法值: grouped (当前) | scattered
  # scattered 模式下每个候选的属性分布在 >=2 turn 中
min_turns_per_candidate: 2
  # 每个候选的属性至少出现在 2 个不同 turn
```

**模板策略**：

- 生成器先构建完整的 candidate-attribute 矩阵（行=候选，列=属性）
- 然后将矩阵的列（属性维度）分配给不同的 turn
- 确保没有任何一个 turn 包含任一候选的**全部**属性
- 约束（属性的 required 版本）单独在早期 turn 给出

---

## 2. Assistant Behavior Matrix

当前 smoke 中的 assistant 是纯 echo chamber（重复 user 内容 + 元注释）。在 hard 版本中，
assistant 应承担选择性角色，增加压缩的难度和真实性。

### 2.1 Behavior Types

| Behavior | 描述 | 难度贡献 | 何时使用 |
|---|---|---|---|
| `echo` | 冗余确认 user 内容（当前行为） | token 成本，无信息增删 | 所有样本的基础行为 |
| `partial_summary` | 选择性总结部分 user 约束，省略其他 | 如果模型信任 assistant 总结 → 丢失未被总结的约束 | 压缩条件容易出错的样本 |
| `incorrect_inference` | 做出一个 nearly-correct 但缺少关键条件的推断 | near-miss 干扰 | 面向 `derived_constraint` 样本 |
| `stale_reiteration` | 在后期 turn 中重申已被更新的早期状态 | stale recirculation 干扰 | 面向 `state_update` 样本 |
| `neutral` | 简短确认（1-2 句），不做冗长回显 | 减少冗余，使其他难度模式更突出 | 至少 1/3 的样本 |

### 2.2 Behavior Assignment Rules

```yaml
assignment_rules:
  - echo: 每个样本至少有 2 个 assistant turn 使用 echo
  - partial_summary: scattered_fact_integration 样本中至少 1 个
  - incorrect_inference: derived_constraint 样本中至少 1 个
  - stale_reiteration: state_update 样本中至少 1 个
  - neutral: 每个样本至少有 1 个 turn 使用 neutral
```

### 2.3 Behavior Output Templates

**partial_summary**：
```
"Got it. So the key points so far are: [列举部分但非全部约束]."
→ 故意省略 1-2 个约束
→ 被省略的约束在 required_evidence 中标记为 at_risk_of_omission
```

**incorrect_inference**：
```
"So if I understand correctly, [nearly correct conclusion that misses one condition]."
→ 接近正确但缺一个条件
→ 被省略的条件标记为 assistant_omitted
```

**stale_reiteration**：
```
"Just to confirm, we're still planning for [旧状态] on [旧时间], right?"
→ 重申已被更新的状态
→ 该 turn 标记为 stale_reiteration distractor
```

**neutral**：
```
"Noted."
或
"Thanks, I've updated the notes."
→ 1-2 句，不做冗长回显
```

---

## 3. Phenomenon × Difficulty × Assistant × Evidence Position 分配表

以下给出 40 个 formal 样本的推荐分配。每个样本属于一个主现象，叠加 1-3 个难度模式，
分配一种 assistant 特殊行为，指定证据位置 profile。

### 3.1 Scattered Fact Integration（10 样本）

| # | 难度模式 | 证据位置 | Assistant 行为 | 备注 |
|---|---|---|---|---|
| 1 | scattered_candidate_attributes + long_distance_evidence (far_early) | early | partial_summary | assistant 省略 1 个关键候选属性 |
| 2 | scattered_candidate_attributes | cross_turn | neutral | 基准 scattered，无特殊行为 |
| 3 | scattered_candidate_attributes + implicit_constraint_tracking | early→late | partial_summary | 早期约束仅出现一次 |
| 4 | scattered_candidate_attributes + soft_hard_conflict_without_labels | cross_turn | echo | 软硬不标记 |
| 5 | scattered_candidate_attributes + long_distance_evidence (far_middle) | middle | neutral | 证据夹在中间 |
| 6 | scattered_candidate_attributes + derived_constraint | early + late | incorrect_inference | 需要推导 + assistant 做错误推断 |
| 7 | scattered_candidate_attributes + subtle_state_update | cross_turn | stale_reiteration | 候选属性被隐式更新 |
| 8 | scattered_candidate_attributes | cross_turn | echo | 纯 scattered，assistant 冗余 |
| 9 | scattered_candidate_attributes + implicit_constraint_tracking + soft_hard_conflict_without_labels | early + late | partial_summary | 高难度组合 |
| 10 | scattered_candidate_attributes + long_distance_evidence (mixed) | mixed | neutral | mixed profile |

### 3.2 State Update（10 样本）

| # | 难度模式 | 证据位置 | Assistant 行为 | 备注 |
|---|---|---|---|---|
| 1 | subtle_state_update | middle | stale_reiteration | 隐式更新 + assistant 重申旧状态 |
| 2 | subtle_state_update + implicit_constraint_tracking | early→late | neutral | 隐式约束不被重复 |
| 3 | subtle_state_update + long_distance_evidence (far_early) | early | echo | 早期更新，后期大量填充 |
| 4 | state_update（显式，作为对照） | middle | echo | 保留 1 个显式更新作为对照 |
| 5 | subtle_state_update + soft_hard_conflict_without_labels | cross_turn | partial_summary | 软硬不标记 |
| 6 | subtle_state_update + scattered_candidate_attributes | cross_turn | neutral | 更新 + 属性分散 |
| 7 | subtle_state_update + derived_constraint | early + late | incorrect_inference | 推导 + 错误推断 |
| 8 | subtle_state_update + long_distance_evidence (far_middle) | middle | echo | 证据在中间 |
| 9 | subtle_state_update + implicit_constraint_tracking + soft_hard_conflict_without_labels | early→late | partial_summary | 高难度组合 |
| 10 | state_update（显式）+ long_distance_evidence (mixed) | mixed | stale_reiteration | 混合 profile + 重申 |

### 3.3 Negation / Exclusion（10 样本）

| # | 难度模式 | 证据位置 | Assistant 行为 | 备注 |
|---|---|---|---|---|
| 1 | implicit_constraint_tracking | early | partial_summary | 早期排除约束仅出现一次 |
| 2 | derived_constraint | early + middle | incorrect_inference | 需要推导排除条件 |
| 3 | soft_hard_conflict_without_labels | cross_turn | echo | 软硬不标记，排除与偏好混淆 |
| 4 | long_distance_evidence (far_early) | early | neutral | 排除条件在早期 |
| 5 | subtle_state_update + negation | cross_turn | stale_reiteration | 排除条件被隐式修改 |
| 6 | scattered_candidate_attributes + negation | cross_turn | partial_summary | 候选属性分散 + 排除 |
| 7 | implicit_constraint_tracking + soft_hard_conflict_without_labels | early→late | echo | 组合 |
| 8 | derived_constraint + long_distance_evidence (far_middle) | middle | neutral | 推导 + 距离 |
| 9 | implicit_constraint_tracking + long_distance_evidence (far_early) | early | partial_summary | 高难度组合 |
| 10 | negation（显式，作为对照）| middle | echo | 保留 1 个显式排除作为对照 |

### 3.4 Temporal Order（10 样本）

| # | 难度模式 | 证据位置 | Assistant 行为 | 备注 |
|---|---|---|---|---|
| 1 | implicit_constraint_tracking | early | neutral | 早期依赖不重复 |
| 2 | subtle_state_update + temporal | cross_turn | stale_reiteration | 时间线被隐式修改 |
| 3 | derived_constraint | early + middle | incorrect_inference | 时间依赖需要推导 |
| 4 | long_distance_evidence (far_early) | early | echo | 依赖在早期 |
| 5 | soft_hard_conflict_without_labels + temporal | cross_turn | partial_summary | 时间约束与偏好交织 |
| 6 | scattered_candidate_attributes + temporal | cross_turn | neutral | 步骤与角色分散 |
| 7 | implicit_constraint_tracking + derived_constraint | early→late | echo | 隐式 + 推导 |
| 8 | long_distance_evidence (far_middle) | middle | neutral | 证据在中间 |
| 9 | subtle_state_update + implicit_constraint_tracking + soft_hard_conflict_without_labels | cross_turn | partial_summary | 高难度组合 |
| 10 | temporal（显式，作为对照）| middle | echo | 保留 1 个显式作为对照 |

### 3.5 覆盖度检查

| 维度 | 覆盖 |
|---|---|
| implicit_constraint_tracking | 10/40（每个现象 2-3） |
| derived_constraint | 8/40（每个现象 2） |
| long_distance_evidence | 12/40（每个现象 3，覆盖 far_early / far_middle / mixed） |
| soft_hard_conflict_without_labels | 10/40（每个现象 2-3） |
| subtle_state_update | 12/40（state_update 7，其他 3 个现象各 1-2） |
| scattered_candidate_attributes | 12/40（scattered 全部 10，其他现象各 1-2） |
| partial_summary | 12/40 |
| incorrect_inference | 4/40 main + up to 2 supplementary hard cases |
| stale_reiteration | 6/40 |
| neutral | 12/40 |
| echo（基础） | 剩余 ~10/40 |
| 显式对照 | 4/40（每个现象 1 个低难度显式样本） |

---

## 4. Compression Prompt Templates

### 4.1 LLM-generated Summary Prompt

用于 `llm_generated_summary` 条件。该 prompt 发给压缩模型（默认 Qwen3-8B，与回答模型相同；
可选更小的模型留待后续 ablation），要求它生成对话摘要。

```
You are compressing a multi-turn dialogue history for later use in answering
a multiple-choice question. The compressed history must fit within approximately
{target_tokens} tokens.

Preserve:
- All stated facts, constraints, exclusions, and preferences.
- The latest state for every entity or decision (updates override earlier values).
- Which constraints are hard requirements and which are soft preferences.
- Candidate entities, plans, or alternatives mentioned in the dialogue and their
  attributes, but do not infer or reference the final multiple-choice options
  (A/B/C/D) — those are part of the later question, not the dialogue history.
- The source turn for each piece of retained information.

Do NOT:
- Solve the final question or compare the candidates against it.
- State which option is correct or eliminate any option.
- Rank the options or make a final recommendation.
- Add any information not present in the original dialogue.
- Infer or discuss the final multiple-choice labels (A/B/C/D).

Original dialogue:
{dialogue_text}

Compressed history (under {target_tokens} tokens):
```

**参数说明**：
- `{target_tokens}`：600（或当前 compressed-history budget）
- `{dialogue_text}`：完整对话历史的渲染文本
- 压缩模型收到的是纯文本，不是结构化 sample。它不知道哪些信息是 `required_evidence`、哪些是 `distractor`

### 4.2 Hybrid Summary + Recent Turns Prompt

用于 `hybrid_summary_recent` 条件。旧历史（turn_1 到 turn_{N-1}）用 LLM 压缩，
最近 1 个完整 turn（user + assistant）保留原文。

**预算分配**：

```
compressed_history_budget: 600 tokens
recent_turns: 1 完整 turn（user + assistant），约 150-250 tokens
summary: 剩余预算，约 350-450 tokens
```

如果最近 turn 超过 250 tokens（例如 assistant 消息异常长），则只保留最近 user 消息原文，
不保留 assistant 消息。

**压缩 prompt**（只发给旧历史部分）：

```
You are compressing older dialogue history. A recent turn will be kept verbatim
separately, so you only need to summarize the older turns.

Summarize the older dialogue history below within approximately {summary_tokens} tokens.
Preserve all facts, constraints, exclusions, preferences, state updates, and candidate
attributes. Mark which information is a hard constraint, soft preference, or outdated.
Do not solve the final question or compare candidates.

Older dialogue history (turns before the most recent):
{older_history_text}

Compressed older history (under {summary_tokens} tokens):
```

**渲染顺序**（拼接为最终 compressed history）：
```
[Compressed older history]
---
[Recent turn — verbatim]
User: {latest_user_message}
Assistant: {latest_assistant_message}
```

**参数说明**：
- `{summary_tokens}`：400（600 - 预留 200 给 recent turn）
- `{older_history_text}`：去掉最近 1 个 turn 后的对话渲染
- `{latest_user_message}`、`{latest_assistant_message}`：最近 turn 的原文

---

## 5. Difficulty Calibration Protocol（Hard Smoke v2）

### 5.1 目标

在 6 个 hard smoke v2 样本上验证三个问题：

1. Ceiling check：full_history 在 6 个样本中是 6/6、5/6、4/6 还是 <4/6？
2. Differentiation check：oracle vs llm summary 是否有显着差异？
3. Failure attribution check：错误是否能被 compression quality 指标解释？

### 5.2 样本选择

从 40 样本分配表中取前 6 个（每个现象至少 1 个，包含至少 2 个高难度组合样本）：

| 样本 | 现象 | 难度组合 |
|---|---|---|
| H1 | scattered_fact | scattered_candidate + long_distance (far_early) |
| H2 | state_update | subtle_update + stale_reiteration |
| H3 | negation | implicit_constraint + derived_constraint |
| H4 | temporal | implicit_constraint + soft_hard_conflict |
| H5 | scattered_fact | scattered + implicit + soft_hard_conflict（高难度） |
| H6 | state_update | subtle_update + implicit + soft_hard_conflict（高难度） |

### 5.3 判定规则

6 条样本下准确率只能是离散值（0/6 → 6/6），不用百分比阈值。

| Full History Correct | 判定 | 行动 |
|---|---|---|
| 6/6 (100%) | Possible ceiling | 检查 explanation 质量和 summary 条件是否有区分度；如无区分度，增加难度后重新 smoke |
| 5/6 (~83%) | 理想 | 保持当前难度配置，进入 pilot |
| 4/6 (~67%) | 偏难但可用 | 进入 pilot，但标记为 hard subset；formal 中适当降低高难度组合比例 |
| <4/6 | 过难 | 降低难度（减少 derived/implicit 叠加），或分离为 supplementary hard subset |

### 5.4 不通过的处置

如果 hard smoke v2 不通过（过难或过易），调整生成器参数后重新生成 smoke 样本，
不直接进入 pilot。调整记录写入 GUIDE.md 的 Progress Log。

---

## 6. Summarizer Inference Settings

LLM-based compression conditions (`llm_generated_summary`, `hybrid_summary_recent`)
require the summarizer model to generate compression artifacts. The summarizer's
inference settings must be defined separately from the answer model.

### 6.1 Default Configuration

| Parameter | Value | Notes |
|---|---|---|
| `summarizer_model` | `qwen3-8b-budget` | Same model as answer model for v1; smaller model ablation reserved for supplementary |
| `summarizer_thinking_enabled` | `false` | Summary generation is extractive/restructuring, not reasoning; thinking disabled to reduce cost and avoid summary-internal reasoning loops |
| `summarizer_max_tokens` | `1024` | Upper bound on summary output length (summary text + compression overhead) |
| `summarizer_temperature` | `0.0` | Deterministic compression for reproducibility |
| `summarizer_top_p` | `1.0` | Same as answer model |

### 6.2 Cost Tracking

- Summarizer input tokens, output tokens, and total tokens are logged separately
  from answer-model usage.
- Summarizer output tokens count toward the compressed-history budget (the
  generated summary text is the compressed representation).
- If a supplementary experiment uses a smaller summarizer model, its cost is
  recorded as a separate column in the run summary.

### 6.3 Thinking Budget Interaction

The summarizer does not use thinking mode. The thinking budget (512 tokens) is
exclusive to the answer model. This avoids conflating summarizer-internal
reasoning cost with compression quality.

---

## 7. Compression Quality Evaluation Method (V1)

The `compression_quality` fields in `data_schema.md` need a concrete evaluation
method. V1 uses a hybrid rule-based + manual approach, appropriate for the scale
of smoke v2 and pilot (6-20 variants to audit).

### 7.1 Required Evidence Retention

**For oracle variants** (`oracle_fact_state_summary`, `oracle_dialogue_summary`):
- Auto-computed: retention = 1.0 by construction, since the generator reads
  `required_evidence` labels directly.
- Verification: during manual audit, spot-check 2-3 oracle artifacts to confirm
  all `evidence_id` spans are present in the rendered text.

**For LLM-based variants** (`llm_generated_summary`, `hybrid_summary_recent`,
`user_only_history`, `sliding_window`):
- V1 method: **evidence_id span matching**.
  - For each `required_evidence` item, check whether its `span` text appears as
    a substring (case-insensitive) in the compressed artifact text.
  - If yes → `retained`. If partial (≥50% token overlap) → `partial`. Else → `missing`.
- Fallback: for `derived_constraint` samples where evidence is not a verbatim span
  but an implicit premise, flag as `requires_manual` and audit by human.
- `retention_rate = retained / total_evidence`.

Important limitation: LLM summaries may paraphrase evidence. Therefore, hard
smoke v2 uses **full manual audit** for all LLM-generated compression artifacts:

```text
6 hard-smoke samples × 2 LLM compression conditions
= 12 compressed artifacts manually audited
```

For these 12 artifacts, substring/overlap checks are only pre-audit hints. The
final evidence-retention, hallucination, stale-state, and answerability labels
are set by manual inspection.

### 7.2 Constraint Preservation

- Compare the set of `evidence_type` labels present in the compressed artifact
  against the original sample's `required_evidence`.
- For `soft_hard_conflict_without_labels` samples, manual audit is required: a
  human reader labels whether hard constraints and soft preferences in the
  summary are correctly distinguished.
- V1 scope: auto-check for oracle variants; manual audit for LLM variants on
  hard smoke v2 (6 samples × 2 LLM conditions = 12 artifacts).

### 7.3 Stale-State Handling

- For each `stale_information` distractor in the sample, check whether the
  compressed artifact:
  - omits it entirely (correct);
  - includes it but marks it as outdated (correct);
  - includes it without any staleness indicator (incorrect — `stale_promoted_to_current`).
- V1 method: manual audit for hard smoke v2.

### 7.4 Hallucinated Fact Count

- V1 method: manual audit. A human reader compares the compressed artifact
  against the original dialogue text and flags any factual claim not supported
  by the source.
- LLM-assisted audit (optional): use a second model call to compare the
  summary against the original dialogue and flag unsupported claims. This is
  reserved for pilot/formal scale-up if manual audit becomes impractical.

### 7.5 Answerability

- If `required_evidence_retention < 1.0` for the gold answer's evidence set,
  label as `insufficient_context_for_answer`.
- Exception: if the model still answers correctly despite missing evidence
  (i.e., it inferred the missing constraint from context), mark as
  `answered_despite_insufficient_context` and flag for manual review.

### 7.6 Scale-Up Plan

| Stage | Variants to Audit | Method |
|---|---|---|
| Hard smoke v2 | 6 samples × 4 conditions = 24 | Manual audit (all) |
| Pilot | 12 samples × 5 conditions = 60 | Manual audit (20%); spot-check for oracle |
| Formal | 40 samples × 5 conditions = 200 | Rule-based for oracle; LLM-assisted for LLM conditions; manual audit 10% |

---

## 8. Incorrect Inference Contamination Analysis

`incorrect_inference` assistant behavior (where the assistant makes a nearly-correct
but incomplete inference) introduces a prior-agent-error confound: if the answer
model fails, it may be because the assistant's error misled it, not because the
compression strategy lost information.

### 8.1 Formal Analysis Rules

- Samples with `assistant_behavior: incorrect_inference` are flagged in the
  parsed output with `assistant_inference_error_present: true`.
- In the main formal analysis, errors on these samples are separated into two
  categories:
  - **Compression failure**: the compressed artifact lacks required evidence
    that was present in the original dialogue.
  - **Assistant contamination failure**: the compressed artifact preserves the
    assistant's incorrect inference, and the answer model adopts it.
- The main accuracy table should report both "all samples" and "excluding
  `incorrect_inference` samples" so the reader can assess the confound.

### 8.2 Formal Distribution Recommendation

- Hard smoke v2: up to 2/6 samples with `incorrect_inference` (as per the
  calibration table in §5.2).
- Pilot: up to 2/12 samples with `incorrect_inference`.
- Formal: target ≤4/40 samples (10%) with `incorrect_inference`. This keeps
  the confound visible but not dominant.
- If `incorrect_inference` proves too confounding in pilot, reclassify those
  samples as supplementary and replace them in the formal set.
