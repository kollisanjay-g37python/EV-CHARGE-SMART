// frontend/src/services/api.js
// Centralised API service layer.
// All fetch calls to FastAPI backend go through here.

const DEFAULT_TIMEOUT_MS = 10000;

async function apiFetch(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError") throw new Error("Request timed out");
    throw err;
  }
}

// ─── Stations ─────────────────────────────────────────────────────────────────

export async function fetchStations(base, { lat, lng, status, connector, limit = 20 } = {}) {
  const params = new URLSearchParams({ limit });
  if (lat != null) params.set("lat", lat);
  if (lng != null) params.set("lng", lng);
  if (status)     params.set("status", status);
  if (connector)  params.set("connector", connector);
  return apiFetch(`${base}/stations?${params}`);
}

export async function fetchStation(base, id) {
  return apiFetch(`${base}/stations/${id}`);
}

// ─── Prediction ───────────────────────────────────────────────────────────────

export async function fetchPrediction(base, payload) {
  return apiFetch(`${base}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchBatchPredictions(base, items) {
  return apiFetch(`${base}/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(items),
  });
}

// ─── Recommendations ──────────────────────────────────────────────────────────

export async function fetchRecommendations(base, payload) {
  return apiFetch(`${base}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

// ─── Queue Analysis ───────────────────────────────────────────────────────────

export async function fetchQueueAnalysis(base, stationId, { arrival_rate = 8, service_rate = 3 } = {}) {
  const params = new URLSearchParams({ arrival_rate, service_rate });
  return apiFetch(`${base}/queue/${stationId}?${params}`);
}

// ─── Forecast ─────────────────────────────────────────────────────────────────

export async function fetchForecast(base, stationId) {
  return apiFetch(`${base}/forecast/${stationId}`);
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

export async function fetchMetrics(base) {
  return apiFetch(`${base}/metrics`);
}

// ─── Capacity Plan ────────────────────────────────────────────────────────────

export async function fetchCapacityPlan(base, { arrival_rate, service_rate, target_wait_min }) {
  const params = new URLSearchParams({ arrival_rate, service_rate, target_wait_min });
  return apiFetch(`${base}/capacity-plan?${params}`);
}
