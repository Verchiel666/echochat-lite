#!/usr/bin/env node
// 推理期用法（不需要训练）：给一组候选回应 + 三阶段推理轨迹 + LLM-as-judge 打分，
// 输出级联门控奖励、SDCA 步骤级信用、根因定位与返工目标。
import fs from 'node:fs';
import { reasoningReward, responseReward, totalReward, sdcaAdvantages, diagnose } from './lib/rewards.mjs';

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith('-'));
if (!file) { console.error('usage: node echochat.mjs case.json [--tau 0.6] [--json out.json]'); process.exit(2); }
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? Number(args[i + 1]) : def; };
const tau = opt('--tau', 0.6);
const caseObj = JSON.parse(fs.readFileSync(file, 'utf8'));
const { scenario, user, rollouts } = caseObj;
if (!Array.isArray(rollouts) || rollouts.length < 2) { console.error('need >=2 rollouts'); process.exit(2); }

const o = { tau, alpha: caseObj.config?.alpha, omegaC: caseObj.config?.omegaC, omegaA: caseObj.config?.omegaA };
const list = rollouts.map((r) => ({ id: r.id, s: r.scores.s, resp: r.scores }));
const { rows } = sdcaAdvantages(list, o);
const diag = diagnose(list, o);

const table = rollouts.map((r, i) => {
  const rr = reasoningReward(r.scores.s, o);
  const rp = responseReward(r.scores, o);
  return {
    id: r.id,
    Rreas: rr.Rreas, rresp: rp.rresp, Rtotal: totalReward({ Rreas: rr.Rreas, rresp: rp.rresp, rfmt: rp.rfmt }),
    gates: rr.g, A: rows[i].A, spans: rows[i].spans,
    rootCause: diag[i].rootCause, repair: diag[i].repair, response: r.trace?.response ?? '',
  };
});
const best = [...table].sort((a, b) => b.Rtotal - a.Rtotal)[0];

const L = (s, n = 30) => String(s).replace(/\s+/g, ' ').slice(0, n);
console.log(`# 场景: ${scenario}`);
console.log(`# 用户: ${L(user, 120)}`);
console.log(`# τ=${tau}  α=[默认]  候选数=${table.length}\n`);
console.log('id           [感知 推理 策略]  R_reas  门控  R_resp  R_total | A(4段)              | 最弱环节');
for (const t of table) {
  const s = rollouts.find((r) => r.id === t.id).scores.s;
  console.log(
    `${String(t.id).padEnd(12)} ${s.map((x) => String(x).padStart(3)).join(' ')} |`
    + ` ${t.Rreas.toFixed(3).padStart(6)} ${t.gates.join('').padStart(5)} `
    + ` ${t.rresp.toFixed(3).padStart(6)} ${t.Rtotal.toFixed(3).padStart(7)} | `
    + `${t.A.map((x) => x.toFixed(2).padStart(5)).join(' ')} | ${t.rootCause ?? '全达标'}`);
}
console.log(`\n胜出: ${best.id}  (R_total=${best.Rtotal.toFixed(3)})`);
for (const t of table) console.log(`  ${t.id}: ${t.repair}`);
if (args.includes('--json')) {
  const out = args[args.indexOf('--json') + 1];
  fs.writeFileSync(out, JSON.stringify({ scenario, tau, table }, null, 2));
  console.log(`\n已写 ${out}`);
}
