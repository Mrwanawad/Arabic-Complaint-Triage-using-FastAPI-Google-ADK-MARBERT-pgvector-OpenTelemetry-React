import { useCallback, useEffect, useState } from "react";
import {
  CATEGORIES,
  Category,
  getPending,
  ReviewItem,
  Sentiment,
  SENTIMENTS,
  submitReview,
  Urgency,
  URGENCIES,
} from "./api";

function ReviewCard({
  item,
  apiKey,
  onDone,
}: {
  item: ReviewItem;
  apiKey: string;
  onDone: () => void;
}) {
  const [category, setCategory] = useState<Category>(item.category);
  const [urgency, setUrgency] = useState<Urgency>(item.urgency);
  const [sentiment, setSentiment] = useState<Sentiment>(item.sentiment);
  const [reviewer, setReviewer] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    try {
      await submitReview(item.ticket_id, { category, urgency, sentiment, reviewer }, apiKey);
      onDone();
    } catch (reason) {
      setError((reason as Error).message);
    }
  }

  return (
    <article className="panel review-card">
      <p dir="auto" className="message">{item.masked_text}</p>
      <p className="muted">
        {item.review_reason} · suggested {item.category} / {item.urgency} ·{" "}
        {Math.round(item.category_confidence * 100)}% category confidence
      </p>
      <div className="row wrap">
        <label>
          Category
          <select value={category} onChange={(e) => setCategory(e.target.value as Category)}>
            {CATEGORIES.map((value) => <option key={value}>{value}</option>)}
          </select>
        </label>
        <label>
          Urgency
          <select value={urgency} onChange={(e) => setUrgency(e.target.value as Urgency)}>
            {URGENCIES.map((value) => <option key={value}>{value}</option>)}
          </select>
        </label>
        <label>
          Sentiment
          <select value={sentiment} onChange={(e) => setSentiment(e.target.value as Sentiment)}>
            {SENTIMENTS.map((value) => <option key={value}>{value}</option>)}
          </select>
        </label>
        <label>
          Reviewer
          <input value={reviewer} maxLength={64} onChange={(e) => setReviewer(e.target.value)} />
        </label>
        <button className="primary" disabled={!reviewer.trim()} onClick={submit}>
          Save decision
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </article>
  );
}

export function ReviewView({ apiKey }: { apiKey: string }) {
  const [items, setItems] = useState<ReviewItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!apiKey) return;
    setError(null);
    try {
      setItems(await getPending(apiKey));
    } catch (reason) {
      setItems(null);
      setError((reason as Error).message);
    }
  }, [apiKey]);

  useEffect(() => {
    load();
  }, [load]);

  if (!apiKey) return <p className="muted">Enter the admin API key to open the review queue.</p>;
  return (
    <section>
      <div className="row between">
        <h2>Review queue {items ? `(${items.length})` : ""}</h2>
        <button onClick={load}>Refresh</button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {items?.length === 0 && <p className="muted">No tickets are waiting for review.</p>}
      {items?.map((item) => (
        <ReviewCard key={item.ticket_id} item={item} apiKey={apiKey} onDone={load} />
      ))}
    </section>
  );
}
