import { useEffect, useState } from "react";
import {
  PieChart, Pie, Cell, Tooltip, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";

const SEGMENT_PROFILES = [
  { subject:"Avg Spend",     "Budget Shoppers":30, "Premium Buyers":95, "Occasional Browsers":45, "Loyal High-Spenders":88 },
  { subject:"Visit Freq",    "Budget Shoppers":55, "Premium Buyers":70, "Occasional Browsers":20, "Loyal High-Spenders":90 },
  { subject:"Session Len",   "Budget Shoppers":40, "Premium Buyers":65, "Occasional Browsers":35, "Loyal High-Spenders":80 },
  { subject:"Pages Viewed",  "Budget Shoppers":50, "Premium Buyers":75, "Occasional Browsers":30, "Loyal High-Spenders":85 },
  { subject:"Loyalty",       "Budget Shoppers":35, "Premium Buyers":60, "Occasional Browsers":15, "Loyal High-Spenders":95 },
];

const TRAITS = {
  "Budget Shoppers":     ["Price sensitive",      "Low basket value",   "High visit frequency",  "Promo-driven"],
  "Premium Buyers":      ["High basket value",    "Brand loyal",        "Low price sensitivity", "Quality focused"],
  "Occasional Browsers": ["Low visit frequency",  "Window shopping",    "Low conversion",        "Deal hunters"],
  "Loyal High-Spenders": ["Repeat purchases",     "High LTV",          "Cross-sell ready",      "VIP candidates"],
};

export default function Segments() {
  const [segs,    setSegs]    = useState([]);
  const [active,  setActive]  = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/segments")
      .then(r => r.json())
      .then(d => { setSegs(d); setActive(d[0]?.name || null); })
      .catch(() => {});
  }, []);

  const pieData = segs.map(s => ({ name: s.name, value: s.count, color: s.color }));
  const radarKeys = segs.map(s => s.name);
  const COLORS    = segs.map(s => s.color);

  const activeSeg = segs.find(s => s.name === active);

  return (
    <div>
      <div className="page-header">
        <h1>Customer Segments</h1>
        <p>K-Means clustering with k=4 – click a segment to explore</p>
      </div>

      {segs.length === 0 ? (
        <div className="card" style={{textAlign:"center", padding:"60px 20px", color:"var(--muted)"}}>
          Run ml_pipeline.py to generate segment data
        </div>
      ) : (
        <>
          {/* Segment tiles */}
          <div className="seg-grid" style={{marginBottom:24}}>
            {segs.map(s => (
              <button key={s.name}
                onClick={() => setActive(s.name)}
                style={{
                  all:"unset", cursor:"pointer",
                  border:`2px solid ${active===s.name ? s.color : "rgba(255,255,255,0.07)"}`,
                  borderRadius:12, padding:20,
                  background: active===s.name ? `${s.color}18` : "var(--bg-800)",
                  transition:"all 0.2s",
                }}>
                <div className="seg-dot" style={{background:s.color, boxShadow:`0 0 8px ${s.color}66`}} />
                <div className="seg-name">{s.name}</div>
                <div className="seg-count">{s.count.toLocaleString()} customers</div>
                <div style={{marginTop:10, display:"flex", flexWrap:"wrap", gap:4}}>
                  {(TRAITS[s.name]||[]).map(t => (
                    <span key={t} style={{
                      fontSize:10, padding:"2px 7px", borderRadius:999,
                      background:`${s.color}22`, color:s.color, fontWeight:500,
                    }}>{t}</span>
                  ))}
                </div>
              </button>
            ))}
          </div>

          {/* Charts row */}
          <div className="grid-2">
            {/* Pie */}
            <div className="card">
              <div className="card-title">Segment Distribution</div>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%" cy="50%"
                    innerRadius={65} outerRadius={105}
                    paddingAngle={3}
                    dataKey="value"
                    label={({ name, percent }) => `${(percent*100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color}
                        stroke={active===entry.name ? "#fff" : "transparent"}
                        strokeWidth={2}
                        style={{cursor:"pointer"}}
                        onClick={() => setActive(entry.name)}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{background:"#1c2030", border:"1px solid rgba(255,255,255,0.1)", borderRadius:8}}
                    formatter={(v, n) => [v.toLocaleString(), n]}
                  />
                  <Legend formatter={(v, e) => <span style={{color:e.color, fontSize:12}}>{v}</span>} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Radar */}
            <div className="card">
              <div className="card-title">Segment Behaviour Radar</div>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={SEGMENT_PROFILES}>
                  <PolarGrid stroke="rgba(255,255,255,0.08)" />
                  <PolarAngleAxis dataKey="subject" tick={{fill:"#94a3b8", fontSize:11}} />
                  <PolarRadiusAxis domain={[0,100]} tick={false} axisLine={false} />
                  {radarKeys.map((k, i) => (
                    <Radar key={k} name={k} dataKey={k}
                      stroke={COLORS[i]} fill={COLORS[i]} fillOpacity={0.15}
                      strokeWidth={active===k ? 2.5 : 1}
                      dot={active===k}
                    />
                  ))}
                  <Legend formatter={(v,e) => <span style={{color:e.color, fontSize:12}}>{v}</span>} />
                  <Tooltip
                    contentStyle={{background:"#1c2030", border:"1px solid rgba(255,255,255,0.1)", borderRadius:8}}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Active segment detail */}
          {activeSeg && (
            <div className="card" style={{marginTop:20, borderColor: activeSeg.color}}>
              <div className="card-title">Segment Detail — {activeSeg.name}</div>
              <div style={{display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(180px,1fr))", gap:16, marginTop:8}}>
                <StatBox label="Customers"    value={activeSeg.count.toLocaleString()} color={activeSeg.color} />
                <StatBox label="Share"        value={`${((activeSeg.count/(segs.reduce((a,s)=>a+s.count,0)))*100).toFixed(1)}%`} color={activeSeg.color} />
                <StatBox label="Cluster ID"   value={`#${activeSeg.id}`} color={activeSeg.color} />
                <StatBox label="Key Traits"   value={TRAITS[activeSeg.name]?.[0] || "—"} color={activeSeg.color} />
              </div>
              <div style={{marginTop:16}}>
                <div style={{fontSize:12, color:"var(--muted)", marginBottom:8}}>Recommended Actions</div>
                <div style={{display:"flex", gap:8, flexWrap:"wrap"}}>
                  {getActions(activeSeg.name).map(a => (
                    <span key={a} style={{
                      fontSize:12, padding:"5px 12px", borderRadius:999,
                      background:`${activeSeg.color}20`, color:activeSeg.color,
                      border:`1px solid ${activeSeg.color}44`,
                    }}>{a}</span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function StatBox({ label, value, color }) {
  return (
    <div style={{
      background:"var(--bg-700)", borderRadius:10,
      padding:"14px 16px", border:`1px solid ${color}33`,
    }}>
      <div style={{fontSize:11, color:"var(--muted)", textTransform:"uppercase", letterSpacing:".05em"}}>{label}</div>
      <div style={{fontFamily:"Space Mono,monospace", fontSize:20, fontWeight:700, color, marginTop:4}}>{value}</div>
    </div>
  );
}

function getActions(name) {
  const map = {
    "Budget Shoppers":      ["Send discount coupons",  "Flash sale alerts",      "Bundle offers"],
    "Premium Buyers":       ["Early access products",  "Loyalty rewards",        "Premium support"],
    "Occasional Browsers":  ["Re-engagement emails",   "Wishlist reminders",     "Exit-intent popups"],
    "Loyal High-Spenders":  ["VIP programme invite",   "Exclusive new arrivals", "Referral bonuses"],
  };
  return map[name] || [];
}
