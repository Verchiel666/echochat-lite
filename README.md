# echochat-lite

EchoChat（NeurIPS 2026, arXiv:2610.04826）**推理期可用的机制实现**，跑在 Android + Node 上。
不是训练复现（本机没有 GPU、没有 LLM API key），而是把论文里能脱离训练使用的那部分落成工具：

- 级联门控奖励（Eq.3-4）：感知不过阈值，下游两项**不计分**，堵死"表面漂亮"的补偿路径
- 响应级奖励（Eq.5）+ 轨迹总奖励（Eq.6）
- SDCA 步骤级信用分配（Eq.7-8）：四段 span 各自算组内相对优势，替代 GRPO 的单一广播标量
- 段边界解析失败时的轨迹级回退通道
- 根因定位：告诉你"该重做哪一步"

## 用法

```bash
node dev/eq-test.mjs                      # 25 项方程一致性断言（手算值）
node dev/credit-audit.mjs                 # 信用分配保真度审计（2 万组）
node dev/sim-toy.mjs                      # toy 训练对照（会饱和，见 README 结论）
node echochat.mjs cases/modal-conflict.json [--tau 0.6] [--json out/x.json]
```

案例文件格式：

```json
{"scenario":"...","user":"...","config":{"alpha":[0.4,0.3,0.3]},
 "rollouts":[{"id":"A","scores":{"s":[感知,推理,策略],"content":0-5,"audio":0|1,"format":true},
              "trace":{"perception":"","mental_state":"","strategy":"","response":""}}]}
```

## 实测（本机，2026-11-07）

| 检查 | 结果 |
|---|---|
| 方程一致性 | 25/25 通过（含 α 归一校验、τ 闭区间、σ=0 回退） |
| 信用分配保真度（2 万组 × 4 rollout） | 错误感知被正向强化的比例 **12.47% → 2.10%**（广播优势 → 门控+SDCA） |
| 上游失守时的下游信用 | 推理段均值 **-1.034**（上游错）vs **+0.563**（上游对），差 1.597 |
| toy 训练对照 | 四种配置都在 ~50 步内到贝叶斯最优，**无差异** → 该规模复现不出论文收益 |

## 结论与边界

- **能验证的**：门控 + SDCA 作为奖励函数，确实会惩罚"依据不对但话很暖"的轨迹——这是可数的。
- **验证不了的**：论文 Table 2/6/8 的收益（感知 8.23 / 推理 8.43 / 回应 9.26 / 级联错误 0.21）来自
  Kimi-Audio + LoRA 的 20k 步真实 RL 训练，本机不具备，**不要把论文数字当成本工具的实测**。
- **踩到的坑**：不能用 SDCA 的 A_k 取 argmin 找根因——门控会把上游失守时的下游分值清零，
  A_2/A_3 一起变负，argmin 会指到"心智状态推理"。根因要用**未加门**的原始分。

## 出处

论文 https://arxiv.org/abs/2610.04826（HTML 全文可读）；项目页 https://github.com/dingdongwang/EchoChat（未验证可达）。
