// frontend/src/hooks/useStations.js
// Custom React hooks for stations, predictions, and real-time polling.

import { useState, useEffect, useCallback, useRef } from "react";
import {
  fetchStations, fetchPrediction, fetchBatchPredictions,
  fetchRecommendations, fetchForecast, fetchQueueAnalysis,
} from "../services/api";

// ─── useStations ──────────────────────────────────────────────────────────────
/**
 * Fetches and auto-refreshes station list.
 * @param {string} apiBase   - API base URL
 * @param {object} options   - { lat, lng, status, connector, limit, refreshMs }
 */
export function useStations(apiBase, options = {}) {
  const { lat, lng, status, connector, limit = 20, refreshMs = 60000 } = options;
  const [stations, setStations]   = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await fetchStations(apiBase, { lat, lng, status, connector, limit });
      setStations(data.stations || []);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBase, lat, lng, status, connector, limit]);

  useEffect(() => { load(); }, [load]);

  // Auto-refresh
  useEffect(() => {
    if (!refreshMs) return;
    const interval = setInterval(load, refreshMs);
    return () => clearInterval(interval);
  }, [load, refreshMs]);

  return { stations, loading, error, lastUpdated, refresh: load };
}

// ─── usePrediction ────────────────────────────────────────────────────────────
/**
 * Runs a single-station prediction on demand.
 */
export function usePrediction(apiBase) {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const predict = useCallback(async (payload) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPrediction(apiBase, payload);
      setResult(data);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, predict, reset };
}

// ─── useBatchPredictions ──────────────────────────────────────────────────────
/**
 * Runs batch predictions for multiple stations.
 */
export function useBatchPredictions(apiBase) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const predict = useCallback(async (items) => {
    if (!items || items.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchBatchPredictions(apiBase, items);
      setResults(data.predictions || []);
      return data.predictions;
    } catch (err) {
      setError(err.message);
      return [];
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  return { results, loading, error, predict };
}

// ─── useRecommendations ───────────────────────────────────────────────────────
/**
 * Fetches station recommendations for a user location.
 */
export function useRecommendations(apiBase, payload, enabled = true) {
  const [recs, setRecs]       = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const payloadRef = useRef(null);
  const payloadStr = JSON.stringify(payload);

  useEffect(() => {
    if (!enabled || !payload?.user_lat) return;
    if (payloadStr === payloadRef.current) return;
    payloadRef.current = payloadStr;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchRecommendations(apiBase, payload);
        setRecs(data.recommendations || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [apiBase, payloadStr, enabled]);

  return { recs, loading, error };
}

// ─── useForecast ──────────────────────────────────────────────────────────────
/**
 * Fetches 12-hour forecast for a station.
 */
export function useForecast(apiBase, stationId) {
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  useEffect(() => {
    if (!stationId) return;
    const load = async () => {
      setLoading(true);
      try {
        const data = await fetchForecast(apiBase, stationId);
        setForecast(data.forecast || []);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [apiBase, stationId]);

  return { forecast, loading, error };
}

// ─── useQueueAnalysis ─────────────────────────────────────────────────────────
/**
 * Fetches M/M/c queue analysis for a station.
 */
export function useQueueAnalysis(apiBase, stationId, params = {}) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    if (!stationId) return;
    const load = async () => {
      setLoading(true);
      try {
        const result = await fetchQueueAnalysis(apiBase, stationId, params);
        setData(result);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [apiBase, stationId, params.arrival_rate, params.service_rate]);

  return { data, loading, error };
}

// ─── useNetworkStats ──────────────────────────────────────────────────────────
/**
 * Derives network-level stats from a stations array.
 */
export function useNetworkStats(stations) {
  return {
    totalStations: stations.length,
    totalPorts:    stations.reduce((a, s) => a + (s.num_ports || 0), 0),
    totalAvail:    stations.reduce((a, s) => a + (s.available_ports || 0), 0),
    totalQueue:    stations.reduce((a, s) => a + (s.queue_size || 0), 0),
    avgWait:       stations.length
      ? stations.reduce((a, s) => a + (s.wait_time_minutes || s.predicted_wait_min || 0), 0) / stations.length
      : 0,
    operational:   stations.filter((s) => s.status?.includes("Operational")).length,
    utilisation:   stations.length
      ? stations.reduce((a, s) => {
          const u = (s.num_ports - s.available_ports) / Math.max(s.num_ports, 1);
          return a + u;
        }, 0) / stations.length
      : 0,
  };
}

// ─── useLocalTime ─────────────────────────────────────────────────────────────
/**
 * Returns a live-updating clock object.
 */
export function useLocalTime(intervalMs = 1000) {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), intervalMs);
    return () => clearInterval(t);
  }, [intervalMs]);
  return {
    time,
    hour: time.getHours(),
    day:  time.getDay() === 0 ? 6 : time.getDay() - 1, // Mon=0
    timeStr: time.toLocaleTimeString(),
    dateStr: time.toLocaleDateString(),
  };
}

// ─── useGeolocation ───────────────────────────────────────────────────────────
/**
 * Requests browser geolocation and returns current position.
 */
export function useGeolocation() {
  const [location, setLocation] = useState(null);
  const [error, setError]       = useState(null);
  const [loading, setLoading]   = useState(false);

  const request = useCallback(() => {
    if (!navigator.geolocation) {
      setError("Geolocation not supported");
      return;
    }
    setLoading(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        setLoading(false);
      },
      (err) => {
        setError(err.message);
        setLoading(false);
      },
      { timeout: 10000, maximumAge: 60000 }
    );
  }, []);

  return { location, error, loading, request };
}
