import { useEffect, useState } from "react";
import { ClassifyView } from "./ClassifyView";
import { DashboardView } from "./DashboardView";
import { ReviewView } from "./ReviewView";

type Tab = "classify" | "review" | "dashboard";

const TABS: { id: Tab; label: string }[] = [
  { id: "classify", label: "Classify" },
  { id: "review", label: "Review queue" },
  { id: "dashboard", label: "Dashboard" },
];

function readStoredKey(): string {
  try {
    return sessionStorage.getItem("triage-api-key") ?? "";
  } catch {
    return "";
  }
}

export function App() {
  const [tab, setTab] = useState<Tab>("classify");
  const [apiKey, setApiKey] = useState(readStoredKey);

  useEffect(() => {
    try {
      sessionStorage.setItem("triage-api-key", apiKey);
    } catch {
      // Storage can be blocked; the key then lasts only for this page view.
    }
  }, [apiKey]);

  return (
    <div className="app">
      <header className="topbar">
        <h1>Complaint Triage</h1>
        <nav aria-label="Sections">
          {TABS.map((item) => (
            <button
              key={item.id}
              className={tab === item.id ? "tab active" : "tab"}
              aria-current={tab === item.id ? "page" : undefined}
              onClick={() => setTab(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <label className="key-field">
          {tab === "classify" ? "API key" : "Admin API key"}
          <input
            type="password"
            autoComplete="off"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
          />
        </label>
      </header>
      <main>
        {tab === "classify" && <ClassifyView apiKey={apiKey} />}
        {tab === "review" && <ReviewView apiKey={apiKey} />}
        {tab === "dashboard" && <DashboardView apiKey={apiKey} />}
      </main>
    </div>
  );
}
