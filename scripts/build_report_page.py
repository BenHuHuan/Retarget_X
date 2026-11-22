"""Build the interactive dark-theme HTML benchmark report at docs/index.html."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark" / "tiangong3_bones_seed_nonhoi"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    hht = json.loads((BENCH / "hhtools_120_nonhoi_summary.json").read_text(encoding="utf-8"))
    dtmr = json.loads((BENCH / "vanilla_dtmr_18_summary.json").read_text(encoding="utf-8"))
    rx = json.loads((BENCH / "retarget_x_v2_8_summary.json").read_text(encoding="utf-8"))

    hht_map = {r["filename"]: r for r in hht}
    dtmr_map = {r["filename"]: r for r in dtmr}

    paired = []
    for r in rx:
        fn = r["clip"]
        h = hht_map.get(fn, {})
        d = dtmr_map.get(fn, {})
        a = r["stage4_audit"]
        c2 = r["stage2_conditioning"]
        dm = d["mpjpe_world_mm"]
        rm = a["mpjpe_world_mm"]
        db = d["base_pos_err_mm"]
        rb = a["base_pos_err_mm"]
        paired.append({
            "clip": fn,
            "difficulty": r["stage3_mpc"]["difficulty"],
            "hht_surv": round(h.get("pd_survival_ratio", 0.0) * 100.0, 1),
            "hht_pen": round(c2["raw_floor_pen_p95_mm"], 1),
            "cond_pen": round(c2["cond_floor_pen_p95_mm"], 2),
            "out_pen": round(a["floor_pen_p95_mm"], 1),
            "dtmr_mpjpe": round(dm, 1),
            "dtmr_base": round(db, 1),
            "dtmr_ori": round(d["orient_err_deg"], 1),
            "rx_mpjpe": round(rm, 1),
            "rx_relcom": round(a["mpjpe_relcom_mm"], 1),
            "rx_base": round(rb, 1),
            "rx_ori": round(a["orient_err_deg"], 1),
            "rx_stretch": round(a["time_stretch_ratio"], 2),
            "mpjpe_delta": round((rm - dm) / dm * 100.0, 1),
            "base_delta": round((rb - db) / db * 100.0, 1),
            "acc_drop": round((c2["cond_joint_acc_p95_rad_s2"] - c2["raw_joint_acc_p95_rad_s2"]) / max(c2["raw_joint_acc_p95_rad_s2"], 1e-6) * 100.0, 1),
        })

    hht_compact = [
        {
            "filename": r["filename"],
            "bucket": r["bucket"],
            "category": r["category"],
            "frames": r["frames_50hz"],
            "sec": r["hht_seconds"],
            "pen95": r["kin_floor_pen_p95_mm"],
            "step99": r["kin_joint_step_p99_deg"],
            "surv": round(r["pd_survival_ratio"] * 100.0, 1),
            "fell": r["pd_fell"],
        }
        for r in hht
    ]

    dtmr_compact = [
        {
            "filename": r["filename"],
            "bucket": r["bucket"],
            "mpjpe": round(r["mpjpe_world_mm"], 1),
            "relcom": round(r["mpjpe_relcom_mm"], 1),
            "base": round(r["base_pos_err_mm"], 1),
            "ori": round(r["orient_err_deg"], 1),
            "stretch": round(r["time_stretch_ratio"], 2),
            "sec": round(r["dtmr_seconds"], 1),
        }
        for r in dtmr
    ]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Retarget_X — Tiangong3 Kinodynamic Retargeting Benchmark Report</title>
<style>
  :root {{
    --bg: #0b0f17;
    --surface: #121826;
    --surface2: #182132;
    --border: rgba(148, 163, 184, 0.16);
    --text: #e2e8f0;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --emerald: #34d399;
    --violet: #a78bfa;
    --amber: #fbbf24;
    --rose: #fb7185;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: radial-gradient(circle at 15% -10%, rgba(56,189,248,0.14), transparent 40%),
                radial-gradient(circle at 85% 0%, rgba(167,139,250,0.14), transparent 42%),
                var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}
  .container {{ max-width: 1220px; margin: 0 auto; padding: 36px 24px 80px; }}
  .hero {{
    text-align: center;
    padding: 48px 28px 42px;
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(24,33,50,0.85), rgba(15,23,42,0.92));
    border: 1px solid var(--border);
    box-shadow: 0 24px 60px rgba(0,0,0,0.45);
    margin-bottom: 32px;
  }}
  .pill {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--cyan);
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    margin-bottom: 16px;
  }}
  h1 {{
    font-size: clamp(32px, 5vw, 52px);
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
  }}
  .subtitle {{
    max-width: 780px;
    margin: 0 auto 24px;
    color: var(--muted);
    font-size: 16px;
  }}
  .hero-actions {{ display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }}
  .btn {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 10px 20px; border-radius: 12px; font-weight: 600; font-size: 14px;
    text-decoration: none; transition: transform .15s ease, opacity .15s ease;
  }}
  .btn:hover {{ transform: translateY(-1px); opacity: 0.94; }}
  .btn-primary {{ background: linear-gradient(90deg, #0284c7, #6366f1); color: #fff; }}
  .btn-ghost {{ background: rgba(255,255,255,0.05); color: var(--text); border: 1px solid var(--border); }}

  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 36px;
  }}
  .kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
  }}
  .kpi::after {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent, var(--cyan)), transparent);
  }}
  .kpi-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }}
  .kpi-val {{ font-size: 30px; font-weight: 800; margin: 6px 0 4px; color: var(--accent, var(--cyan)); }}
  .kpi-sub {{ font-size: 12.5px; color: var(--muted); }}

  section {{ margin-bottom: 42px; }}
  h2 {{
    font-size: 22px; font-weight: 700; margin-bottom: 16px;
    display: flex; align-items: center; gap: 10px;
  }}
  h2 span.dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--cyan); display: inline-block; }}

  .pipeline {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
  }}
  .stage-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
  }}
  .stage-num {{
    font-size: 11px; font-weight: 700; color: var(--violet);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
  }}
  .stage-card h3 {{ font-size: 16px; margin-bottom: 8px; color: #f8fafc; }}
  .stage-card p {{ font-size: 13.5px; color: var(--muted); }}

  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 20px;
  }}
  .chart-card img {{ width: 100%; height: auto; border-radius: 12px; display: block; }}
  .chart-caption {{ font-size: 13px; color: var(--muted); margin-top: 10px; }}

  .tabs {{ display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }}
  .tab-btn {{
    background: var(--surface); color: var(--muted);
    border: 1px solid var(--border); padding: 9px 16px; border-radius: 10px;
    cursor: pointer; font-weight: 600; font-size: 13.5px;
  }}
  .tab-btn.active {{
    background: rgba(56, 189, 248, 0.14); color: var(--cyan);
    border-color: rgba(56, 189, 248, 0.45);
  }}
  .controls {{ display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }}
  .controls input, .controls select {{
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 8px 12px; font-size: 13.5px; outline: none;
  }}
  .table-wrap {{
    overflow-x: auto;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  th, td {{ padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  th {{ background: var(--surface2); color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
  tr:hover td {{ background: rgba(56, 189, 248, 0.04); }}
  .good {{ color: var(--emerald); font-weight: 700; }}
  .bad {{ color: var(--rose); font-weight: 600; }}
  .tag {{
    display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11.5px; font-weight: 600;
  }}
  .tag-high {{ background: rgba(251,113,133,0.14); color: var(--rose); }}
  .tag-medium {{ background: rgba(251,191,36,0.14); color: var(--amber); }}
  .tag-low {{ background: rgba(52,211,153,0.14); color: var(--emerald); }}
  .hidden {{ display: none; }}
  footer {{ text-align: center; color: var(--muted); font-size: 13px; margin-top: 48px; padding-top: 24px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
  <div class="hero">
    <div class="pill">Tiangong3 (29-DOF · 65.8 kg) · BONES-SEED Non-HOI Benchmark</div>
    <h1>Retarget_X</h1>
    <p class="subtitle">
      融合 <strong>hhtools</strong> 极速跨本体运动学重定向与 <strong>DTMR</strong> MuJoCo Warp 物理时序优化的 4 阶段统一引擎。
      在天工3非 HOI 动作集上实现 <strong>100% 物理不倒</strong>、参考穿地归零 <strong>0.00 mm</strong>，并较 Vanilla DTMR 降低 <strong>15.9% MPJPE</strong> 与 <strong>18.1% 骨盆误差</strong>。
    </p>
    <div class="hero-actions">
      <a class="btn btn-primary" href="https://github.com/BenHuHuan/Retarget_X" target="_blank">GitHub Repository</a>
      <a class="btn btn-ghost" href="#charts">查看对比图表</a>
      <a class="btn btn-ghost" href="#explorer">交互式数据浏览器</a>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi" style="--accent: var(--emerald)">
      <div class="kpi-label">Physical No-Fall Rate</div>
      <div class="kpi-val">0% → 100%</div>
      <div class="kpi-sub">vs hhtools direct PD (1.7% on 120 clips)</div>
    </div>
    <div class="kpi" style="--accent: var(--cyan)">
      <div class="kpi-label">World MPJPE vs DTMR</div>
      <div class="kpi-val">-15.9%</div>
      <div class="kpi-sub">68.8 mm → 57.9 mm (High Jump -60.9%)</div>
    </div>
    <div class="kpi" style="--accent: var(--violet)">
      <div class="kpi-label">Pelvis Base Error</div>
      <div class="kpi-val">-18.1%</div>
      <div class="kpi-sub">52.1 mm → 42.6 mm mean across 8 clips</div>
    </div>
    <div class="kpi" style="--accent: var(--emerald)">
      <div class="kpi-label">Stage 2 Floor Penetration</div>
      <div class="kpi-val">0.00 mm</div>
      <div class="kpi-sub">Reduced from 25.0 mm P95 (-100%)</div>
    </div>
    <div class="kpi" style="--accent: var(--amber)">
      <div class="kpi-label">MPC Speedup (L4 GPU)</div>
      <div class="kpi-val">4.8×</div>
      <div class="kpi-sub">4.8 min vs 19.9 min (beats 4096 on Egypt Dance)</div>
    </div>
  </div>

  <section>
    <h2><span class="dot"></span>4-Stage Kinodynamic Architecture</h2>
    <div class="pipeline">
      <div class="stage-card">
        <div class="stage-num">Stage 1 · Kinematic IK</div>
        <h3>Enhanced hhtools</h3>
        <p>采用精确 SDK 脚底站立高度比 (<code>1.2586</code>) 彻底消除 <code>1.3m</code> 身高回退穿地 Bug，支持单/多姿态 (<code>T90/T60/T90_elbow45</code>) 整段择优。</p>
      </div>
      <div class="stage-card">
        <div class="stage-num">Stage 2 · Core Bridge</div>
        <h3>Kinodynamic Conditioning</h3>
        <p>零相位 Savitzky-Golay 平滑（加速度毛刺降 12%–37%）+ 精确脚底胶囊体接触投影（穿地归零 <code>0.00 mm</code> 且保留腾空）+ 动力学难度画像。</p>
      </div>
      <div class="stage-card">
        <div class="stage-num">Stage 3 · Physics MPC</div>
        <h3>Difficulty-Adaptive DTMR</h3>
        <p>按 <code>low / medium / high</code> 难度自动分配 MuJoCo Warp 采样预算、前瞻时域与时间相位弹性，并基于消融结论关闭 <code>w_foot_slip</code>。</p>
      </div>
      <div class="stage-card">
        <div class="stage-num">Stage 4 · Audit & Export</div>
        <h3>Unified RL / Sim Delivery</h3>
        <p>输出运动学+动力学双维度质检报告，并导出标准 50 Hz <code>.npz</code>，可直接接入 xMimic / StableMimic / IsaacLab 训练。</p>
      </div>
    </div>
  </section>

  <section id="charts">
    <h2><span class="dot"></span>Visual Benchmark Results</h2>
    <div class="chart-card">
      <img src="assets/benchmark_overview.png" alt="Vanilla DTMR vs Retarget_X MPJPE and Base Position Error" />
      <div class="chart-caption">图 1：8 条高代表性非 HOI BONES-SEED 动作上 Vanilla DTMR 与 Retarget_X 的全身世界坐标 MPJPE（左）与骨盆位置误差（右）对比。</div>
    </div>
    <div class="chart-card">
      <img src="assets/stage2_and_baseline.png" alt="Stage 2 Conditioning and 120-clip hhtools Baseline" />
      <div class="chart-caption">图 2：Stage 2 调理前后参考穿地归零效果（左），以及 120 条非 HOI BONES-SEED（28,445 帧 @ 50Hz）按类别统计的 hhtools 直接 PD 存活率对比（右）。</div>
    </div>
    <div class="chart-card">
      <img src="assets/ablation_tradeoff.png" alt="Ablation Study on Foot-Slip Weight and Runtime" />
      <div class="chart-caption">图 3：消融实验——关闭 <code>w_foot_slip</code> 并匹配天工3踝关节阻尼比后，Egypt Dance 从 84.4 mm 降至 26.9 mm（4.8 分钟超越 20 分钟 Vanilla 4096 的 28.0 mm），同时 High Jump 维持 -60.9% 巨大降幅。</div>
    </div>
  </section>

  <section id="explorer">
    <h2><span class="dot"></span>Interactive Benchmark Explorer</h2>
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('paired', this)">三方同场精测 (8 Clips)</button>
      <button class="tab-btn" onclick="switchTab('dtmr18', this)">Vanilla DTMR 基线 (18 Clips)</button>
      <button class="tab-btn" onclick="switchTab('hht120', this)">hhtools 非 HOI 全量基线 (120 Clips)</button>
    </div>

    <div id="tab-paired">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Clip</th><th>Difficulty</th><th>hhtools PD Surv</th><th>Ref FloorPen95</th>
              <th>Acc Drop</th><th>Vanilla DTMR (MPJPE / Base / Ori)</th>
              <th>Retarget_X (MPJPE / Base / Ori)</th><th>MPJPE Δ</th><th>Base Δ</th><th>Stretch</th>
            </tr>
          </thead>
          <tbody id="tbody-paired"></tbody>
        </table>
      </div>
    </div>

    <div id="tab-dtmr18" class="hidden">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Clip</th><th>Bucket</th><th>MPJPE World</th><th>MPJPE RelCoM</th>
              <th>Base Pos Err</th><th>Orient Err</th><th>Time Stretch</th><th>L4 Time (s)</th>
            </tr>
          </thead>
          <tbody id="tbody-dtmr18"></tbody>
        </table>
      </div>
    </div>

    <div id="tab-hht120" class="hidden">
      <div class="controls">
        <input id="search-hht" type="text" placeholder="搜索动作名..." oninput="renderHht120()" />
        <select id="bucket-hht" onchange="renderHht120()">
          <option value="">全部类别 (120)</option>
        </select>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Clip</th><th>Bucket</th><th>Category</th><th>Frames (50Hz)</th>
              <th>hht Time (s)</th><th>FloorPen95 (mm)</th><th>JointStep99 (°)</th><th>Direct PD Surv (%)</th>
            </tr>
          </thead>
          <tbody id="tbody-hht120"></tbody>
        </table>
      </div>
    </div>
  </section>

  <footer>
    Retarget_X · Tiangong3 Kinodynamic & Dense Temporal Motion Retargeting · Motion Data by Bones Studio (BONES-SEED Non-HOI)
  </footer>
</div>

<script>
const PAIRED = {json.dumps(paired, ensure_ascii=False)};
const DTMR18 = {json.dumps(dtmr_compact, ensure_ascii=False)};
const HHT120 = {json.dumps(hht_compact, ensure_ascii=False)};

function switchTab(id, btn) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  ['paired', 'dtmr18', 'hht120'].forEach(t => {{
    document.getElementById('tab-' + t).classList.toggle('hidden', t !== id);
  }});
}}

function renderPaired() {{
  const tb = document.getElementById('tbody-paired');
  tb.innerHTML = PAIRED.map(r => `
    <tr>
      <td><code>${{r.clip}}</code></td>
      <td><span class="tag tag-${{r.difficulty}}">${{r.difficulty}}</span></td>
      <td class="bad">${{r.hht_surv.toFixed(1)}}%</td>
      <td>${{r.hht_pen.toFixed(1)}} → <span class="good">${{r.cond_pen.toFixed(2)}} mm</span></td>
      <td class="good">${{r.acc_drop.toFixed(1)}}%</td>
      <td>${{r.dtmr_mpjpe}} mm / ${{r.dtmr_base}} mm / ${{r.dtmr_ori}}°</td>
      <td><strong>${{r.rx_mpjpe}} mm / ${{r.rx_base}} mm / ${{r.rx_ori}}°</strong></td>
      <td class="${{r.mpjpe_delta <= 0 ? 'good' : 'bad'}}">${{r.mpjpe_delta > 0 ? '+' : ''}}${{r.mpjpe_delta.toFixed(1)}}%</td>
      <td class="${{r.base_delta <= 0 ? 'good' : 'bad'}}">${{r.base_delta > 0 ? '+' : ''}}${{r.base_delta.toFixed(1)}}%</td>
      <td>${{r.rx_stretch.toFixed(2)}}×</td>
    </tr>`).join('');
}}

function renderDtmr18() {{
  const tb = document.getElementById('tbody-dtmr18');
  tb.innerHTML = DTMR18.map(r => `
    <tr>
      <td><code>${{r.filename}}</code></td>
      <td>${{r.bucket}}</td>
      <td>${{r.mpjpe}} mm</td>
      <td>${{r.relcom}} mm</td>
      <td>${{r.base}} mm</td>
      <td>${{r.ori}}°</td>
      <td>${{r.stretch.toFixed(2)}}×</td>
      <td>${{r.sec}} s</td>
    </tr>`).join('');
}}

function initBuckets() {{
  const sel = document.getElementById('bucket-hht');
  const buckets = [...new Set(HHT120.map(r => r.bucket))].sort();
  buckets.forEach(b => {{
    const cnt = HHT120.filter(r => r.bucket === b).length;
    const opt = document.createElement('option');
    opt.value = b; opt.textContent = `${{b}} (${{cnt}})`;
    sel.appendChild(opt);
  }});
}}

function renderHht120() {{
  const q = document.getElementById('search-hht').value.toLowerCase();
  const b = document.getElementById('bucket-hht').value;
  const rows = HHT120.filter(r => (!b || r.bucket === b) && (!q || r.filename.toLowerCase().includes(q)));
  document.getElementById('tbody-hht120').innerHTML = rows.map(r => `
    <tr>
      <td><code>${{r.filename}}</code></td>
      <td>${{r.bucket}}</td>
      <td>${{r.category}}</td>
      <td>${{r.frames}}</td>
      <td>${{r.sec.toFixed(1)}} s</td>
      <td>${{r.pen95.toFixed(1)}} mm</td>
      <td>${{r.step99.toFixed(1)}}°</td>
      <td class="${{r.fell ? 'bad' : 'good'}}">${{r.surv.toFixed(1)}}% ${{r.fell ? '(fell)' : '(ok)'}}</td>
    </tr>`).join('');
}}

renderPaired();
renderDtmr18();
initBuckets();
renderHht120();
</script>
</body>
</html>
"""
    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Built interactive report -> {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
