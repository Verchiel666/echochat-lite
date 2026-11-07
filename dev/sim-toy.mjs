// 机制复现（toy）：三阶段链路 e -> n -> s* 上做 GRPO 式 REINFORCE，
// 对比 {广播优势, 无门控, 无SDCA, 完整 EchoChat} 四组的跨阶段错误传播。
// 声明：这是按论文机理**构造**的小规模仿真，用来验证公式实现的因果行为，不能替代论文的训练结果。
import { reasoningReward, responseReward, totalReward, spanRewards, sdcaAdvantages } from '../lib/rewards.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const __outDir = (() => { const d = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'out'); fs.mkdirSync(d, { recursive: true }); return d; })();

const rnd = (seed) => () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
function gauss(R) { let u = 0, v = 0; while (!u) u = R(); while (!v) v = R(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

const CFG = { tau: 0.6, flipN: 0.2, qObs: 0.82, judgeNoise: 1.2, G: 4, steps: 4000, lr: 0.25, evalN: 4000 };

const softmax = (z) => { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a,b)=>a+b,0); return e.map(v=>v/s); };
const sample = (p, R) => { let x = R(), k = 0; for (; k < p.length-1; k++) { if (x < p[k]) break; x -= p[k]; } return k; };

function makePolicy() {
  return { p1: [[0.5,0.5],[0.5,0.5]], p2: [[0.5,0.5],[0.5,0.5]], p3: [[0.5,0.5],[0.5,0.5]] }; // 概率参数(softmax 的 logit 用 log p)
}
const logits = (row) => row.map(v => Math.log(Math.max(v, 1e-6)));
const upd = (row, a, A, lr) => { const p = softmax(logits(row)); for (let k=0;k<2;k++) row[k] = Math.max(1e-3, Math.min(1-1e-3, p[k] + lr*A*((k===a?1:0)-p[k]))); };
const renorm = (row) => { const s = row[0]+row[1]; row[0] /= s; row[1] /= s; };

function rollout(pi, R) {
  const e = R() < 0.5 ? 0 : 1;
  const n = R() < CFG.flipN ? 1-e : e;
  const o = R() < CFG.qObs ? e : 1-e;
  const eh = sample(pi.p1[o], R);
  const nh = sample(pi.p2[eh], R);
  const sh = sample(pi.p3[nh], R);
  // 冻结的 LLM-as-judge（带噪声）：s1 对真值；s2/s3 只对"与上游前题是否自洽" + 噪声 —— 这就是"表面漂亮"的来源
  const s1 = eh === e ? 5 : 0;
  const s2 = Math.max(0, Math.min(5, (nh === eh ? 5 : 1) + gauss(R) * CFG.judgeNoise));
  const s3 = Math.max(0, Math.min(5, (sh === nh ? 5 : 1) + gauss(R) * CFG.judgeNoise));
  const content = sh === n ? 5 : 1;
  const audio = R() < 0.5 ? 1 : 0;
  return { e, n, o, eh, nh, sh, s: [s1, s2, s3], resp: { content, audio } };
}

function evalPolicy(pi, R, nEval = 2000) {
  let cp = 0, cr = 0, cs = 0, cascade = 0, corrP = 0;
  for (let i = 0; i < nEval; i++) {
    const e = R() < 0.5 ? 0 : 1, n = R() < CFG.flipN ? 1-e : e, o = R() < CFG.qObs ? e : 1-e;
    const eh = sample(softmax(logits(pi.p1[o])), R), nh = sample(softmax(logits(pi.p2[eh])), R), sh = sample(softmax(logits(pi.p3[nh])), R);
    if (eh === e) cp++; if (nh === n) cr++; if (sh === n) cs++;
    if (eh === e) { corrP++; if (sh !== n) cascade++; }
  }
  return { accP: cp/nEval, accR: cr/nEval, accResp: cs/nEval, cascade: corrP ? cascade/corrP : 0 };
}

function run(variant, seed, curve = false) {
  const R = rnd(seed), pi = makePolicy();
  const curvePts = [50, 100, 200, 400, 800, 1600, 4000];
  const out = [];
  for (let t = 0; t < CFG.steps; t++) {
    if (curve && curvePts.includes(t)) { const m = evalPolicy(pi, rnd(seed + t)); m.step = t; out.push(m); }
    const grp = Array.from({ length: CFG.G }, () => rollout(pi, R));
    const gating = variant !== 'broadcast' && variant !== 'noGating';
    const sdca = variant === 'sdca' || variant === 'noGating';
    const o = { tau: CFG.tau };
    if (sdca) {
      const list = grp.map((r) => ({ id: 'x', s: r.s, resp: r.resp }));
      let A;
      if (gating) { A = sdcaAdvantages(list, o).rows; }
      else { // 无门控：tau 设为 0 -> 门恒开
        A = sdcaAdvantages(list, { ...o, tau: 0 }).rows;
      }
      grp.forEach((r, i) => {
        upd(pi.p1[r.o], r.eh, A[i].A[0], CFG.lr);
        upd(pi.p2[r.eh], r.nh, A[i].A[1], CFG.lr);
        upd(pi.p3[r.nh], r.sh, A[i].A[2], CFG.lr);
      });
    } else {
      const Rs = grp.map((r) => {
        const rr = reasoningReward(r.s, { ...o, tau: gating ? CFG.tau : 0 });
        const rp = responseReward(r.resp, o);
        return totalReward({ Rreas: rr.Rreas, rresp: rp.rresp, rfmt: rp.rfmt });
      });
      const mu = Rs.reduce((a,b)=>a+b,0)/Rs.length;
      const sd = Math.max(1e-3, Math.sqrt(Rs.reduce((a,b)=>a+(b-mu)**2,0)/Rs.length));
      grp.forEach((r, i) => {
        const A = (Rs[i]-mu)/(sd+1e-3);
        upd(pi.p1[r.o], r.eh, A, CFG.lr);
        upd(pi.p2[r.eh], r.nh, A, CFG.lr);
        upd(pi.p3[r.nh], r.sh, A, CFG.lr);
      });
    }
    [pi.p1,pi.p2,pi.p3].forEach(rows => rows.forEach(renorm));
  }
  if (curve) { const m = evalPolicy(pi, rnd(seed + 99999)); m.step = CFG.steps; out.push(m); return { curve: out }; }
  // 评测（无 judge 噪声的影响：指标只看对真值）
  let cp=0, cr=0, cs=0, cascade=0, corrP=0;
  const nEval = CFG.evalN;
  for (let i = 0; i < nEval; i++) {
    const e = R() < 0.5 ? 0 : 1;
    const n = R() < CFG.flipN ? 1-e : e;
    const o = R() < CFG.qObs ? e : 1-e;
    const eh = sample(softmax(logits(pi.p1[o])), R);
    const nh = sample(softmax(logits(pi.p2[eh])), R);
    const sh = sample(softmax(logits(pi.p3[nh])), R);
    if (eh===e) cp++; if (nh===n) cr++; if (sh===n) cs++;
    if (eh===e) { corrP++; if (sh!==n) cascade++; }   // 感知对了但最终仍错 = 级联失败
  }
  return { accP: cp/nEval, accR: cr/nEval, accResp: cs/nEval, P_R_given_P: 0, cascade: corrP ? cascade/corrP : 0 };
}

const variants = [ ['broadcast','广播优势(无门控/无SDCA)'], ['noGating','SDCA 但无门控'], ['noSDCA','门控 但广播优势'], ['sdca','门控+SDCA(完整)'] ];
const seeds = [7, 42, 1234];
const rows = [];
for (const [v, label] of variants) {
  const res = seeds.map(s => run(v, s));
  const mean = (k) => res.reduce((a,b)=>a+b[k],0)/res.length;
  const std = (k) => { const m = mean(k); return Math.sqrt(res.reduce((a,b)=>a+(b[k]-m)**2,0)/res.length); };
  const r = { variant: v, label, accP: mean('accP'), accR: mean('accR'), accResp: mean('accResp'), cascade: mean('cascade'), sd: { accP: std('accP'), cascade: std('cascade') } };
  rows.push(r);
  console.log(`${label.padEnd(26)} 感知=${(r.accP*100).toFixed(1)}%  推理=${(r.accR*100).toFixed(1)}%  回应=${(r.accResp*100).toFixed(1)}%  级联失败率=${(r.cascade*100).toFixed(1)}%`);
}
fs.writeFileSync(path.join(__outDir, 'sim-toy.json'), JSON.stringify({ cfg: CFG, seeds, rows }, null, 2));

console.log('\n[学习曲线] 每组 3 seed 平均；感知=ê==e，回应=ŝ==n');
const curves = {};
for (const [v, label] of variants) {
  const cs = seeds.map(sd => run(v, sd, true).curve);
  const pts = cs[0].map((_, i) => {
    const at = cs.map(c => c[i]);
    return { step: at[0].step, accP: at.reduce((a,b)=>a+b.accP,0)/at.length, accResp: at.reduce((a,b)=>a+b.accResp,0)/at.length };
  });
  curves[v] = pts;
  console.log(label.padEnd(26) + pts.map(p => `@${p.step}:P${(p.accP*100).toFixed(1)}/R${(p.accResp*100).toFixed(1)}`).join('  '));
}
fs.writeFileSync(path.join(__outDir, 'sim-curves.json'), JSON.stringify({ cfg: CFG, curves }, null, 2));
