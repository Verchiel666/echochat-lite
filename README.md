# echochat-lite

[![npm](https://img.shields.io/npm/v/echochat-lite.svg)](https://www.npmjs.com/package/echochat-lite)
[![license](https://img.shields.io/npm/l/echochat-lite.svg)](./LICENSE)

EchoChat（NeurIPS 2026，[arXiv:2610.04826](https://arxiv.org/abs/2610.04826)）**推理期**实现的零依赖 Node 工具：
级联门控奖励 + 步骤级信用分配（SDCA）+ 根因定位。

写它是因为论文里有两类东西：**必须训练才能用的**（AAA、RL 训练），和**推理期就能用的**（结构化三段轨迹、
门控评分、步骤级定位）。本机没有 GPU 也没有可用 LLM API，所以只落地后者——不需要训练，不需要 GPU，
输入是一组候选回应 + 三阶段轨迹 + judge 打分，输出是"哪个候选赢、为什么、该重做哪一步"。

## 安装

```bash
npm i echochat-lite            # 或用 npx 直接跑
npx echochat-lite --help
```

## 用法

```bash
# 1) 方程一致性自检（25 项，全是手算值）
node dev/eq-test.mjs

# 2) 信用分配保真度审计（2 万组 × 4 rollout，约 1 秒）
node dev/credit-audit.mjs

# 3) toy 训练对照（会饱和，见下文"结论与边界"）
node dev/sim-toy.mjs

# 4) 给自己的候选回应打分
node echochat.mjs cases/modal-conflict.json --tau 0.6 --json out/result.json
```

案例文件格式：

```json
{"scenario": "...", "user": "...", "config": {"alpha": [0.4, 0.3, 0.3]},
 "rollouts": [{"id": "A",
               "scores": {"s": [感知, 推理, 策略], "content": 0-5, "audio": 0|1, "format": true},
               "trace": {"perception": "", "mental_state": "", "strategy": "", "response": ""}}]}
```

`s` 三段是 LLM-as-judge 给的 0–5 分（评分口径见 [`skill/SKILL.md`](./skill/SKILL.md)），
本工具只吃分数、不调用任何模型——所以它跟哪个 judge 都能配。

## 输出长什么样

```
id           [感知 推理 策略]  R_reas  门控  R_resp  R_total | A(4段)              | 最弱环节
A-温暖空          2   4   4 |  0.160    01   0.860   0.524 | -1.40 -1.36 -1.29 -0.00 | 感知
B-依据对但空        4   3   3 |  0.680    11   0.720   0.728 |  0.70  0.39  0.16 -1.21 | 心智状态推理/支持策略
C-依据对且贴        4   4   5 |  0.860    11   1.000   0.930 |  0.70  0.97  1.13  1.21 | 感知/心智状态推理
```

A 是"共情模仿"的典型：响应奖励最高（0.860），但因为感知没过门控，级联奖励被清零，总分垫底。
返工指令是分阶段的——A 只该重做感知，改回应无效。

## 实测（本机，2026-11-07）

| 检查 | 结果 |
|---|---|
| 方程一致性 | 25/25 通过（含 α 归一校验、τ 闭区间、σ=0 回退、逐 token 按段展开） |
| 信用分配保真度（20000 组 × 4 rollout） | 错误感知被正向强化的比例 **12.47% → 2.10%**（广播优势 → 门控+SDCA） |
| 上游失守时的下游信用 | 推理段均值 **-1.034**（上游错）vs **+0.563**（上游对），差 1.597 |
| toy 训练对照 | 四种配置都在 ~50 步内到贝叶斯最优，**无差异** |

## 结论与边界

- **能验证的**：门控 + SDCA 作为奖励函数，确实会惩罚"依据不对但话很暖"的轨迹——这是可数的。
- **验证不了的**：论文 Table 2/6/7/8 的收益（感知 8.23 / 推理 8.43 / 回应 9.26 / 级联错误 0.21）来自
  Kimi-Audio + LoRA 的 20k 步真实 RL 训练，**不要把论文数字当成本工具的实测**。
- **toy 会饱和**：2 动作表格策略在这个规模梯度信号太足，四种配置无差异。这个负面结果留着，因为它说明
  "小规模 toy 给不了论文背书"。
- **踩到的坑**：不能用 SDCA 的 A_k 取 argmin 找根因——门控会把上游失守时的下游分值清零，A₂/A₃ 一起变负，
  argmin 会指到"心智状态推理"。根因要用**未加门**的原始分（工具里已处理）。

## DSH 技能版

[`skill/SKILL.md`](./skill/SKILL.md) 是把这套流程用于实际对话的操作手册（评分口径、三条硬规则、返工顺序），
装到 `$DSH_HOME/skills/empathy-cascade/` 后，DSH 会在需要给共情回应时自动用上。

## 出处

论文 <https://arxiv.org/abs/2610.04826>（HTML 全文可读）　项目页 <https://github.com/dingdongwang/EchoChat>（本机网络不通，未验证）

MIT License.
