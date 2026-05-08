import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

export default function Dashboard({ metrics }) {
  const [trend, setTrend] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5000/api/sales-trend")
      .then(r => r.json())
      .then(setTrend)
      .catch(() => {});
  }, []);

  const clf = metrics?.classification;
  const reg = metrics?.regression;

  const clfBars = clf ? [
    { name:"Accuracy",  value: clf.accuracy  * 100 },
    { name:"Precision", value: clf.precision * 100 },
    { name:"Recall",    value: clf.recall    * 100 },
    { name:"F1 Score",  value: clf.f1        * 100 },
  ] : [];

  return (
    <div>
      <div className="page-header">
        <h1>Analytics Dashboard</h1>
        <p>Overview of model performance and sales trends</p>
      </div>

      {/* Metric tiles */}
      <div className="metric-grid">
        <Tile label="Classifier Accuracy" value={clf ? `${(clf.accuracy*100).toFixed(1)}%` : "–"} sub="XGBoost + Bayesian" />
        <Tile label="F1 Score"            value={clf ? clf.f1.toFixed(3) : "–"}                   sub="Classification" />
        <Tile label="Regression R²"       value={reg ? reg.r2.toFixed(3) : "–"}                   sub="Sales Prediction" />
        <Tile label="RMSE"                value={reg ? `₹${reg.rmse}` : "–"}                       sub="Avg Error" />
        <Tile label="Precision"           value={clf ? `${(clf.precision*100).toFixed(1)}%` : "–"} sub="Purchase Detection" />
        <Tile label="MAE"                 value={reg ? `₹${reg.mae}` : "–"}                        sub="Mean Abs Error" />
      </div>

      {/* Charts row */}
      <div className="grid-2" style={{marginBottom: 20}}>
        <div className="card">
          <div className="card-title">Monthly Avg Sales Trend</div>
          {trend.length > 0 ? (
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="month" tick={{fill:"#64748b", fontSize:11}} />
                <YAxis tick={{fill:"#64748b", fontSize:11}} />
                <Tooltip
                  contentStyle={{background:"#1c2030", border:"1px solid rgba(255,255,255,0.1)", borderRadius:8}}
                  labelStyle={{color:"#e2e8f0"}} itemStyle={{color:"#5b8cff"}}
                />
                <Line type="monotone" dataKey="sales" stroke="#5b8cff"
                  strokeWidth={2.5} dot={{ r:3, fill:"#5b8cff" }} activeDot={{ r:5 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <Placeholder />
          )}
        </div>

        <div className="card">
          <div className="card-title">Classifier Metrics</div>
          {clfBars.length > 0 ? (
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={clfBars} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" domain={[0,100]} tick={{fill:"#64748b", fontSize:11}}
                  tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="name" tick={{fill:"#e2e8f0", fontSize:12}} width={70}/>
                <Tooltip
                  contentStyle={{background:"#1c2030", border:"1px solid rgba(255,255,255,0.1)", borderRadius:8}}
                  formatter={v => [`${v.toFixed(1)}%`]}
                />
                <Bar dataKey="value" radius={[0,4,4,0]}>
                  {clfBars.map((_, i) => (
                    <rect key={i} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Placeholder />
          )}
        </div>
      </div>

      {/* Model param cards */}
      <div className="grid-2">
        <div className="card">
          <div className="card-title">Best Classifier Params (Bayesian Opt)</div>
          <ParamTable params={clf?.best_params} />
        </div>
        <div className="card">
          <div className="card-title">Best Regressor Params (Bayesian Opt)</div>
          <ParamTable params={reg?.best_params} />
        </div>
      </div>
    </div>
  );
}

function Tile({ label, value, sub }) {
  return (
    <div className="metric-tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}

function ParamTable({ params }) {
  if (!params) return <Placeholder />;
  return (
    <table style={{width:"100%", borderCollapse:"collapse", fontSize:13}}>
      <tbody>
        {Object.entries(params).map(([k,v]) => (
          <tr key={k} style={{borderBottom:"1px solid rgba(255,255,255,0.05)"}}>
            <td style={{padding:"7px 0", color:"#94a3b8", fontFamily:"Space Mono, monospace", fontSize:11}}>{k}</td>
            <td style={{padding:"7px 0", textAlign:"right", color:"#a78bfa", fontFamily:"Space Mono, monospace"}}>{String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Placeholder() {
  return (
    <div style={{height:150, display:"flex", alignItems:"center", justifyContent:"center",
      color:"#475569", fontSize:13}}>
      Run ml_pipeline.py to load data
    </div>
  );
}
