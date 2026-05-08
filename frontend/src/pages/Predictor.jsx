import { useState } from "react";

const COUNTRIES = ["UK","Germany","France","Netherlands","Australia","Japan","Sweden","Norway"];

const DEFAULTS = {
  Quantity: 5, UnitPrice: 25.99, NumProducts: 3,
  DayOfWeek: 2, Month: 11, Hour: 14,
  IsReturning: 1, SessionLength: 18, PagesViewed: 7, Country: "UK",
};

export default function Predictor() {
  const [form,    setForm]    = useState(DEFAULTS);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");

  const handle = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const predict = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          Quantity:      parseFloat(form.Quantity),
          UnitPrice:     parseFloat(form.UnitPrice),
          NumProducts:   parseInt(form.NumProducts),
          DayOfWeek:     parseInt(form.DayOfWeek),
          Month:         parseInt(form.Month),
          Hour:          parseInt(form.Hour),
          IsReturning:   parseInt(form.IsReturning),
          SessionLength: parseFloat(form.SessionLength),
          PagesViewed:   parseInt(form.PagesViewed),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Server error");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1>Customer Predictor</h1>
        <p>Enter customer session data to get real-time ML predictions</p>
      </div>

      <div className="predictor-layout">
        {/* ── Form ── */}
        <div className="card">
          <div className="section-title">Customer Input Data</div>
          {error && <div className="error-banner">⚠ {error}</div>}

          <div className="form-grid">
            <Field label="Quantity"       type="number" value={form.Quantity}      onChange={v => handle("Quantity", v)}      min={1}   max={999}  step="1" />
            <Field label="Unit Price (₹)" type="number" value={form.UnitPrice}     onChange={v => handle("UnitPrice", v)}     min={0.01} max={9999} step="0.01" />
            <Field label="Num Products"   type="number" value={form.NumProducts}   onChange={v => handle("NumProducts", v)}   min={1}   max={200}  step="1" />
            <Field label="Day of Week"    type="number" value={form.DayOfWeek}     onChange={v => handle("DayOfWeek", v)}     min={0}   max={6}    step="1" hint="0=Mon…6=Sun" />
            <Field label="Month"          type="number" value={form.Month}         onChange={v => handle("Month", v)}         min={1}   max={12}   step="1" />
            <Field label="Hour (24h)"     type="number" value={form.Hour}          onChange={v => handle("Hour", v)}          min={0}   max={23}   step="1" />
            <Field label="Session Length (min)" type="number" value={form.SessionLength} onChange={v => handle("SessionLength",v)} min={1} max={300} step="0.5"/>
            <Field label="Pages Viewed"   type="number" value={form.PagesViewed}   onChange={v => handle("PagesViewed", v)}   min={1}   max={200}  step="1" />

            {/* Is Returning */}
            <div className="form-group">
              <label>Returning Customer</label>
              <select value={form.IsReturning} onChange={e => handle("IsReturning", e.target.value)}>
                <option value={1}>Yes</option>
                <option value={0}>No</option>
              </select>
            </div>

            {/* Country */}
            <div className="form-group">
              <label>Country</label>
              <select value={form.Country} onChange={e => handle("Country", e.target.value)}>
                {COUNTRIES.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>

          <button className="btn-predict" onClick={predict} disabled={loading}>
            {loading ? <><span className="spinner"/>Predicting…</> : "🔮 Run Prediction"}
          </button>
        </div>

        {/* ── Results ── */}
        <div className="result-block">
          {!result && !loading && (
            <div className="card" style={{textAlign:"center", padding:"60px 20px"}}>
              <div style={{fontSize:48, marginBottom:12}}>🎯</div>
              <div style={{color:"var(--muted)", fontSize:14}}>Fill in customer data and click Run Prediction</div>
            </div>
          )}

          {loading && (
            <div className="card" style={{textAlign:"center", padding:"60px 20px"}}>
              <div style={{fontSize:48, marginBottom:12}}>⚙️</div>
              <div style={{color:"var(--muted)", fontSize:14}}>Running ML models…</div>
            </div>
          )}

          {result && (
            <>
              {/* Purchase Decision */}
              <div className={`result-card ${result.will_purchase ? "green" : "red"}`}>
                <div className="rc-label">Purchase Prediction</div>
                <div className="rc-value">
                  {result.will_purchase ? "✅ Will Purchase" : "❌ Won't Purchase"}
                </div>
                <div className="prob-bar-wrap" style={{marginTop:12}}>
                  <div style={{display:"flex", justifyContent:"space-between", fontSize:11, color:"var(--muted)", marginBottom:5}}>
                    <span>Purchase Probability</span>
                    <span>{result.purchase_probability}%</span>
                  </div>
                  <div className="prob-bar-bg">
                    <div className="prob-bar-fill" style={{width:`${result.purchase_probability}%`}} />
                  </div>
                </div>
              </div>

              {/* Sales Prediction */}
              <div className="result-card blue">
                <div className="rc-label">Predicted Sales</div>
                <div className="rc-value">₹ {result.predicted_sales.toLocaleString("en-IN", {minimumFractionDigits:2})}</div>
                <div style={{fontSize:12, color:"var(--muted)", marginTop:4}}>XGBoost Regressor estimate</div>
              </div>

              {/* Cluster */}
              <div className="result-card orange">
                <div className="rc-label">Customer Segment</div>
                <div className="rc-value" style={{fontSize:20, marginTop:8}}>
                  <span style={{display:"inline-block", width:12, height:12,
                    borderRadius:"50%", background: result.cluster_color,
                    marginRight:8, verticalAlign:"middle"}} />
                  {result.cluster_name}
                </div>
                <div style={{fontSize:12, color:"var(--muted)", marginTop:4}}>K-Means cluster #{result.cluster_id}</div>
              </div>

              {/* Raw JSON */}
              <details style={{marginTop:4}}>
                <summary style={{cursor:"pointer", color:"var(--muted)", fontSize:12, userSelect:"none"}}>
                  View raw response JSON
                </summary>
                <pre style={{
                  marginTop:10, background:"var(--bg-700)",
                  border:"1px solid var(--border)", borderRadius:8,
                  padding:"12px 14px", fontSize:11,
                  color:"#94a3b8", overflowX:"auto"
                }}>
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({ label, type, value, onChange, min, max, step, hint }) {
  return (
    <div className="form-group">
      <label>{label}{hint && <span style={{color:"var(--muted)",marginLeft:4,fontSize:10}}>({hint})</span>}</label>
      <input
        type={type}
        value={value}
        min={min} max={max} step={step}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  );
}
