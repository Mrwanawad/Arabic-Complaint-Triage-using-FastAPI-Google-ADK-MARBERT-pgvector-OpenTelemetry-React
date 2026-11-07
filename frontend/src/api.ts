export type Category = "billing" | "outage" | "speed" | "installation" | "other";
export type Urgency = "high" | "normal";
export type Sentiment = "negative" | "neutral" | "positive";

export const CATEGORIES: Category[] = ["billing", "outage", "speed", "installation", "other"];
export const URGENCIES: Urgency[] = ["high", "normal"];
export const SENTIMENTS: Sentiment[] = ["negative", "neutral", "positive"];

export interface Triage {
  ticket_id: string;
  category: Category;
  category_confidence: number;
  urgency: Urgency;
  urgency_confidence: number;
  sentiment: Sentiment;
  sentiment_confidence: number;
  normalized_text: string;
  script: string;
  decided_by: string;
  target_queue: string;
  status: string;
  review_reason: string | null;
  model_version: string;
  latency_ms: number;
}

export interface ReviewItem extends Triage {
  masked_text: string;
}

export interface HourBucket {
  hour: string;
  category: string;
  count: number;
  negative: number;
}

export interface Dashboard {
  total: number;
  by_category: Record<string, number>;
  by_urgency: Record<string, number>;
  by_sentiment: Record<string, number>;
  by_decider: Record<string, number>;
  by_hour: HourBucket[];
  pending_review: number;
  override_rate: number;
}

export interface Decision {
  category: Category;
  urgency: Urgency;
  sentiment: Sentiment;
  reviewer: string;
}

async function request<T>(path: string, init: RequestInit = {}, apiKey?: string): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (apiKey) headers.set("X-API-Key", apiKey);
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const classify = (text: string, model: string) =>
  request<Triage>(`/classify?model=${encodeURIComponent(model)}`, {
    method: "POST",
    body: JSON.stringify({ text, channel: "api" }),
  });

export const getModels = () => request<{ models: string[] }>("/health");

export const getPending = (apiKey: string) => request<ReviewItem[]>("/review", {}, apiKey);

export const submitReview = (id: string, decision: Decision, apiKey: string) =>
  request<Triage>(`/review/${id}`, { method: "POST", body: JSON.stringify(decision) }, apiKey);

export const getDashboard = (apiKey: string, hours: number) =>
  request<Dashboard>(`/dashboard?hours=${hours}`, {}, apiKey);
