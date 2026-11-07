import { FormEvent, useEffect, useState } from "react";
import { classify, getModels, Triage } from "./api";

const SAMPLES = [
  { label: "Egyptian Arabic", text: "النت واقع من امبارح ومفيش خدمة خالص والشغل واقف" },
  { label: "Franco-Arabic", text: "el net wa2e3 men embare7 w mafeesh ay service" },
  { label: "Mixed", text: "الـ invoice بتاعة الشهر ده فيها 450 جنيه زيادة" },
  { label: "English", text: "My internet speed is very slow since yesterday" },
];

const percent = (value: number) => `${Math.round(value * 100)}%`;

export function ClassifyView() {
  const [text, setText] = useState(SAMPLES[0].text);
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [result, setResult] = useState<Triage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getModels()
      .then(({ models: available }) => {
        setModels(available);
        setModel(available.includes("encoder") ? "encoder" : available[0] ?? "");
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await classify(text, model));
    } catch (reason) {
      setResult(null);
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="grid-two">
      <form className="panel" onSubmit={submit}>
        <h2>Complaint</h2>
        <div className="samples" role="group" aria-label="Sample messages">
          {SAMPLES.map((sample) => (
            <button type="button" key={sample.label} onClick={() => setText(sample.text)}>
              {sample.label}
            </button>
          ))}
        </div>
        <label htmlFor="complaint">Message text</label>
        <textarea
          id="complaint"
          dir="auto"
          rows={6}
          maxLength={5000}
          required
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <div className="row">
          <label>
            Model
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {models.map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
          <button className="primary" disabled={busy || !model}>
            {busy ? "Classifying..." : "Classify"}
          </button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </form>

      <div className="panel" aria-live="polite">
        <h2>Result</h2>
        {!result && <p className="muted">Submit a message to see its triage decision.</p>}
        {result && (
          <>
            <p className="queue">
              <span className="muted">Routed to</span> <strong>{result.target_queue}</strong>
              {result.review_reason && <span className="badge warn">{result.review_reason}</span>}
            </p>
            <dl className="facts">
              <dt>Category</dt>
              <dd>{result.category} · {percent(result.category_confidence)}</dd>
              <dt>Urgency</dt>
              <dd>{result.urgency} · {percent(result.urgency_confidence)}</dd>
              <dt>Sentiment</dt>
              <dd>{result.sentiment} · {percent(result.sentiment_confidence)}</dd>
              <dt>Decided by</dt>
              <dd>{result.decided_by}</dd>
              <dt>Script</dt>
              <dd>{result.script}</dd>
              <dt>Normalized</dt>
              <dd dir="auto">{result.normalized_text}</dd>
              <dt>Model</dt>
              <dd>{result.model_version}</dd>
              <dt>Latency</dt>
              <dd>{result.latency_ms.toFixed(1)} ms</dd>
            </dl>
          </>
        )}
      </div>
    </section>
  );
}
