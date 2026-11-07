// 方程一致性测试：每个断言都是手算值，按论文 Eq.(3)-(8)
import { reasoningReward as RR, responseReward as RP, totalReward as TR,
         spanRewards as SR, sdcaAdvantages as SDCA, fallbackAdvantage, diagnose, norm } from '../lib/rewards.mjs';

let pass = 0, fail = 0;
const eq = (name, got, want, tol = 1e-9) => {
  const ok = (typeof got === 'string' || typeof want === 'string')
    ? got === want
    : (Array.isArray(want) ? JSON.stringify(got.map(v=>+v.toFixed(12))) === JSON.stringify(want.map(v=>+v.toFixed(12))) : Math.abs(got - want) <= tol);
  const fmt=(v)=>typeof v==="number"?v.toFixed(6):JSON.stringify(v);
  if (ok) { pass++; console.log(`  ok   ${name} = ${fmt(got)}`); }
  else { fail++; console.log(`  FAIL ${name} got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
};

console.log('[Eq.3+4] 级联门控推理奖励  alpha=[.4,.3,.3] tau=0.6');
let r = RR([4, 3, 4]);
eq('s=[4,3,4] R_reas', r.Rreas, 0.74);            // .4*.8 + .3*.6*1 + .3*.8*1
eq('gates', r.g, [1, 1]);
r = RR([4, 3, 4], { tau: 0.7 });                   // 感知 0.8>=0.7 过，推理 0.6<0.7 不过
eq('tau=0.7 s=[4,3,4] R_reas', r.Rreas, 0.5);      // .32 + .18*1 + .24*1*0
eq('tau=0.7 gates', r.g, [1, 0]);
r = RR([2, 5, 5]);
eq('感知失守 s=[2,5,5] R_reas', r.Rreas, 0.16);     // 只剩 .4*.4，下游被门控清零（抑制补偿）
eq('感知失守 gates', r.g, [0, 1]); // g2 独立判定，但被 g1 乘掉 -> 不计分
eq('边界 tau 恰好命中 (5*.6=3)', RR([3, 0, 0], { tau: 0.6 }).g[0], 1); // >= 是闭区间
eq('alpha 归一校验抛错', (() => { try { RR([1,1,1], { alpha: [0.5,0.5,0.5] }); return 0; } catch { return 1; } })(), 1);
eq('out-of-range 抛错', (() => { try { norm(5.1); return 0; } catch { return 1; } })(), 1);

console.log('[Eq.5] 响应级奖励  wc=.7 wa=.3');
let p = RP({ content: 4, audio: 1 });
eq('content=4 audio=1 r_resp', p.rresp, 0.86);      // .7*.8+.3*1
p = RP({ content: 4, audio: 0 });
eq('content=4 audio=0 r_resp', p.rresp, 0.56);
eq('format 缺失 -> 0', RP({ content: 3, format: false }).rfmt, 0);

console.log('[Eq.6] 轨迹总奖励  w=[.5,.4,.1]');
eq('R_total', TR({ Rreas: 0.74, rresp: 0.86, rfmt: 1 }), 0.814); // .37+.344+.1

console.log('[Eq.7] 四段分解 spanRewards');
let s = SR({ s: [4, 3, 4], resp: { content: 4, audio: 1 } }, { omegaResp: 0.6, omegaFmt: 0.4 });
eq('spans [r1,r2,r3,rR]', s.spans, [0.32, 0.18, 0.24, 0.916]); // rR=.6*.86+.4*1

console.log('[Eq.8] SDCA 组内相对优势');
const group = [
  { id: 'A', s: [4, 4, 4], resp: { content: 4, audio: 1 } },
  { id: 'B', s: [2, 5, 5], resp: { content: 4, audio: 1 } }, // 感知失守但下游表面漂亮
  { id: 'C', s: [4, 4, 5], resp: { content: 5, audio: 1 } },
];
const d = SDCA(group, { omegaResp: 0.6, omegaFmt: 0.4 });
const spans = d.rows.map(x => x.spans[0]);
const mu1 = spans.reduce((a, b) => a + b, 0) / 3, sd1 = Math.sqrt(spans.reduce((a, b) => a + (b - mu1) ** 2, 0) / 3);
eq('span1 均值', mu1, (0.32 + 0.16 + 0.32) / 3);
eq('A_1(A)', d.rows[0].A[0], (0.32 - mu1) / (sd1 + 1e-3));
eq('B 的感知优势为负', d.rows[1].A[0] < 0, true);
eq('B 的"漂亮下游"在轨迹级仍是低分', TR({
  Rreas: RR(group[1].s, {}).Rreas, rresp: RP(group[1].resp, {}).rresp, rfmt: 1 }) < TR({
  Rreas: RR(group[0].s, {}).Rreas, rresp: RP(group[0].resp, {}).rresp, rfmt: 1 }), true);
const tok = d.perToken([2, 3, 2, 4])[1];
eq('B 逐 token 优势按段展开', tok, [d.rows[1].A[0], d.rows[1].A[0], d.rows[1].A[1], d.rows[1].A[1], d.rows[1].A[1], d.rows[1].A[2], d.rows[1].A[2], d.rows[1].A[3], d.rows[1].A[3], d.rows[1].A[3], d.rows[1].A[3]]);
const fb = fallbackAdvantage(group);
eq('回退通道 σ=0 时优势为 0', fallbackAdvantage([{ s: [4,4,4], resp:{content:4,audio:1} },{ s: [4,4,4], resp:{content:4,audio:1} }]).every(x=>x===0), true);
eq('回退通道与 SDCA 同号', fb[1] < fb[2], true);

console.log('[诊断] 定位最弱环节');
const dg = diagnose(group);
eq('B 的根因 = 感知（不是 A 最小的那段）', dg[1].rootCause, '感知');
eq('B 的 A 最小段（门控副作用，不可当根因）', dg[1].weakestByAdvantage, '心智状态推理');
eq('A 全段过门控且并列最低', dg[0].rootCause, '感知/心智状态推理/支持策略');
eq('B 带门控说明', dg[1].notes.length > 0, true);

console.log(`\n${fail === 0 ? 'PASS' : 'FAIL'}  ${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
