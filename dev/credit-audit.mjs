// 信用分配保真度审计（确定性、可复现）：
// 在"下游表面自洽但上游感知错"的 rollout 组上，比较 GRPO 式广播优势 vs 门控+SDCA
// 是否会**正向强化**一个错误感知。这是奖励函数的性质，不是训练效果的声明。
import { reasoningReward, responseReward, totalReward, sdcaAdvantages } from '../lib/rewards.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const __outDir = (() => { const d = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'out'); fs.mkdirSync(d, { recursive: true }); return d; })();

const R = (s) => () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
const rr = R(20261107);
function gauss() { let u = 0, v = 0; while (!u) u = rr(); while (!v) v = rr(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
const clip = (x) => Math.max(0, Math.min(5, x));

const CFG = { tau: 0.6, G: 4, groups: 20000, judgeNoise: 1.2, pPercWrong: 0.35 };

function sampleGroup() {
  const g = [];
  for (let i = 0; i < CFG.G; i++) {
    const percOk = rr() >= CFG.pPercWrong;
    // 感知对的样本 s1 满 5 分；错的样本 s1∈[0,2]（模仿"听到了但没听准"）
    const s1 = percOk ? 5 : clip(rr() * 2);
    // 下游：与(错误)前题自洽 => 5 分左右；judge 噪声让下游分数有方差（"表面漂亮"的来源）
    const s2 = clip(5 + gauss() * CFG.judgeNoise);
    const s3 = clip(5 + gauss() * CFG.judgeNoise);
    const content = percOk ? clip(4 + gauss() * 0.7) : clip(3.5 + gauss() * 0.9);
    g.push({ id: i, percOk, s: [s1, s2, s3], resp: { content, audio: rr() < 0.6 ? 1 : 0 } });
  }
  return g;
}

const stat = (v) => ({ n: v.length, pos: v.filter(x => x > 0).length / v.length, mean: v.reduce((a,b)=>a+b,0)/v.length });

let bc = { wrong: [], right: [] }, sd = { wrong: [], right: [], wrongS2: [], rightS2: [], wrongS3: [], rightS3: [] };
for (let k = 0; k < CFG.groups; k++) {
  const g = sampleGroup();
  // (1) 经典 GRPO：广播优势（门控关闭）
  const Rs = g.map(r => { const a = reasoningReward(r.s, { tau: 0 }), b = responseReward(r.resp, {});
    return totalReward({ Rreas: a.Rreas, rresp: b.rresp, rfmt: b.rfmt }); });
  const mu = Rs.reduce((a,b)=>a+b,0)/Rs.length;
  const sdv = Math.max(1e-3, Math.sqrt(Rs.reduce((a,b)=>a+(b-mu)**2,0)/Rs.length));
  g.forEach((r, i) => { const A = (Rs[i]-mu)/(sdv+1e-3); (r.percOk ? bc.right : bc.wrong).push(A); });
  // (2) 门控 + SDCA
  const adv = sdcaAdvantages(g, { tau: CFG.tau }).rows;
  g.forEach((r, i) => {
    if (r.percOk) { sd.right.push(adv[i].A[0]); sd.rightS2.push(adv[i].A[1]); sd.rightS3.push(adv[i].A[2]); }
    else { sd.wrong.push(adv[i].A[0]); sd.wrongS2.push(adv[i].A[1]); sd.wrongS3.push(adv[i].A[2]); }
  });
}

const pct = (x) => (x*100).toFixed(2) + '%';
const out = [];
console.log(`组数=${CFG.groups} 每组 rollout 数=${CFG.G} τ=${CFG.tau}\n`);
console.log('【经典 GRPO / 广播优势】感知段 token 收到的优势');
const b1 = stat(bc.wrong), b2 = stat(bc.right);
console.log(`  感知错(${b1.n})  正向比例=${pct(b1.pos)}  均值=${b1.mean.toFixed(3)}`);
console.log(`  感知对(${b2.n})  正向比例=${pct(b2.pos)}  均值=${b2.mean.toFixed(3)}`);
out.push({ scheme:'broadcast', wrongPos:b1.pos, rightPos:b2.pos, wrongMean:b1.mean, rightMean:b2.mean });
console.log('\n【门控 + SDCA】各段 token 收到的优势');
const d1 = stat(sd.wrong), d2 = stat(sd.right);
console.log(`  感知段: 感知错正向比例=${pct(d1.pos)} (均值 ${d1.mean.toFixed(3)}) | 感知对正向比例=${pct(d2.pos)} (均值 ${d2.mean.toFixed(3)})`);
const s2w = stat(sd.wrongS2), s2r = stat(sd.rightS2), s3w = stat(sd.wrongS3), s3r = stat(sd.rightS3);
console.log(`  推理段: 上游错 均值=${s2w.mean.toFixed(3)} | 上游对 均值=${s2r.mean.toFixed(3)}  (差 ${(s2r.mean-s2w.mean).toFixed(3)})`);
console.log(`  策略段: 上游错 均值=${s3w.mean.toFixed(3)} | 上游对 均值=${s3r.mean.toFixed(3)}  (差 ${(s3r.mean-s3w.mean).toFixed(3)})`);
out.push({ scheme:'gated+sdca', wrongPos:d1.pos, rightPos:d2.pos, wrongMean:d1.mean, rightMean:d2.mean,
           span2:{wrong:s2w.mean,right:s2r.mean}, span3:{wrong:s3w.mean,right:s3r.mean} });
console.log(`\n关键对比：错误感知被"正向强化"的比例  ${pct(b1.pos)} (广播) -> ${pct(d1.pos)} (门控+SDCA)`);
console.log(`          对错误感知的平均信用         ${b1.mean.toFixed(3)} -> ${d1.mean.toFixed(3)}`);
fs.writeFileSync(path.join(__outDir, 'credit-audit.json'), JSON.stringify({ cfg: CFG, out }, null, 2));
