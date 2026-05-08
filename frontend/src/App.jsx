import { useState, useEffect } from "react";
import Dashboard   from "./pages/Dashboard";
import Predictor   from "./pages/Predictor";
import Segments    from "./pages/Segments";
import "./App.css";

const NAV = [
  { id:"dashboard",  label:"📊 Dashboard"   },
  { id:"predictor",  label:"🔮 Predictor"   },
  { id:"segments",   label:"👥 Segments"    },
];

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/metrics")
      .then(r => r.json())
      .then(setMetrics)
      .catch(() => setMetrics("error"));
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">⚡</span>
          <div>
            <div className="brand-title">CustomerIQ</div>
            <div className="brand-sub">ML Analytics</div>
          </div>
        </div>
        <nav>
          {NAV.map(n => (
            <button
              key={n.id}
              className={`nav-btn ${page === n.id ? "active" : ""}`}
              onClick={() => setPage(n.id)}
            >{n.label}</button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="status-dot" />
          <span>Models {metrics && metrics !== "error" ? "Loaded ✓" : "Offline"}</span>
        </div>
      </aside>

      <main className="main-content">
        {page === "dashboard" && <Dashboard metrics={metrics} />}
        {page === "predictor" && <Predictor />}
        {page === "segments"  && <Segments />}
      </main>
    </div>
  );
}
