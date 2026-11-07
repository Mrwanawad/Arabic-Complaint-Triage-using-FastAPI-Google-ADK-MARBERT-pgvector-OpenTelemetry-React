import Plotly from "plotly.js-basic-dist-min";
import { useEffect, useRef, useState } from "react";
import { Dashboard, getDashboard } from "./api";

const SERIES_COLORS = ["#2a6fbb", "#c8553d", "#3a8f6a", "#8a63b8", "#7a7f87"];
const HOUR_MS = 3_600_000;

function themeColors() {
  const styles = getComputedStyle(document.documentElement);
  return {
    text: styles.getPropertyValue("--text").trim(),
    grid: styles.getPropertyValue("--border").trim(),
  };
}

function VolumeChart({ data }: { data: Dashboard }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const categories = [...new Set(data.by_hour.map((bucket) => bucket.category))].sort();
    const traces = categories.map((category, index) => {
      const buckets = data.by_hour.filter((bucket) => bucket.category === category);
      return {
        type: "bar",
        name: category,
        x: buckets.map((bucket) => bucket.hour),
        y: buckets.map((bucket) => bucket.count),
        marker: { color: SERIES_COLORS[index % SERIES_COLORS.length] },
        width: HOUR_MS * 0.8,
      };
    });
    // Without an explicit range, a single hour of data makes the axis zoom into milliseconds.
    const times = data.by_hour.map((bucket) => new Date(bucket.hour).getTime());
    const range = [Math.min(...times) - HOUR_MS, Math.max(...times) + HOUR_MS].map((time) =>
      new Date(time).toISOString(),
    );
    const colors = themeColors();
    Plotly.react(
      node,
      traces,
      {
        barmode: "stack",
        height: 320,
        margin: { t: 16, r: 16, b: 48, l: 48 },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: colors.text },
        xaxis: {
          type: "date",
          range,
          tickformat: "%b %d %H:%M",
          gridcolor: colors.grid,
          title: { text: "Hour (UTC)" },
        },
        yaxis: { gridcolor: colors.grid, title: { text: "Complaints" }, rangemode: "tozero" },
        legend: { orientation: "h", y: 1.12 },
      },
      { displayModeBar: false, responsive: true },
    );
    return () => Plotly.purge(node);
  }, [data]);

  return <div ref={ref} role="img" aria-label="Complaint volume per hour, stacked by category" />;
}

function Breakdown({ title, values }: { title: string; values: Record<string, number> }) {
  const total = Object.values(values).reduce((sum, value) => sum + value, 0);
  return (
    <div className="panel">
      <h3>{title}</h3>
      <ul className="breakdown">
        {Object.entries(values)
          .sort(([, a], [, b]) => b - a)
          .map(([name, value]) => (
            <li key={name}>
              <span>{name}</span>
              <span>{value} · {total ? Math.round((value / total) * 100) : 0}%</span>
            </li>
          ))}
      </ul>
    </div>
  );
}

export function DashboardView({ apiKey }: { apiKey: string }) {
  const [hours, setHours] = useState(24);
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!apiKey) return;
    let current = true;
    setError(null);
    getDashboard(apiKey, hours)
      .then((result) => current && setData(result))
      .catch((reason: Error) => {
        if (!current) return;
        setData(null);
        setError(reason.message);
      });
    return () => {
      current = false;
    };
  }, [apiKey, hours]);

  if (!apiKey) return <p className="muted">Enter the admin API key to open the dashboard.</p>;
  return (
    <section>
      <div className="row between">
        <h2>Dashboard</h2>
        <label>
          Window
          <select value={hours} onChange={(event) => setHours(Number(event.target.value))}>
            <option value={24}>24 hours</option>
            <option value={168}>7 days</option>
            <option value={720}>30 days</option>
          </select>
        </label>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {data && (
        <>
          <div className="stats">
            <div className="panel"><span className="muted">Tickets</span><strong>{data.total}</strong></div>
            <div className="panel"><span className="muted">Pending review</span><strong>{data.pending_review}</strong></div>
            <div className="panel"><span className="muted">Override rate</span><strong>{Math.round(data.override_rate * 100)}%</strong></div>
          </div>
          <div className="panel">
            <h3>Volume by hour</h3>
            {data.by_hour.length ? <VolumeChart data={data} /> : <p className="muted">No tickets in this window.</p>}
          </div>
          <div className="stats">
            <Breakdown title="Category" values={data.by_category} />
            <Breakdown title="Urgency" values={data.by_urgency} />
            <Breakdown title="Sentiment" values={data.by_sentiment} />
            <Breakdown title="Decided by" values={data.by_decider} />
          </div>
        </>
      )}
    </section>
  );
}
