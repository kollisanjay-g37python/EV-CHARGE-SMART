// frontend/src/hooks/useWebSocket.js
import { useState, useEffect, useRef, useCallback } from "react";

const WS_BASE = (process.env.REACT_APP_API_URL || "http://localhost:8000/api/v1")
  .replace("http://", "ws://")
  .replace("https://", "wss://")
  .replace("/api/v1", "");

/**
 * useWebSocket — generic hook for a single WebSocket connection.
 * Handles reconnection with exponential back-off.
 *
 * @param {string}   path       — WS path, e.g. "/ws/live"
 * @param {boolean}  enabled    — connect only when true
 * @returns {{ lastMessage, readyState, reconnect }}
 */
export function useWebSocket(path, enabled = true) {
  const [lastMessage,  setLastMessage]  = useState(null);
  const [readyState,   setReadyState]   = useState(WebSocket.CLOSED);
  const wsRef     = useRef(null);
  const retryRef  = useRef(0);
  const timerRef  = useRef(null);

  const connect = useCallback(() => {
    if (!enabled) return;
    const url = `${WS_BASE}${path}`;
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setReadyState(WebSocket.OPEN);
        retryRef.current = 0;
      };

      ws.onmessage = (evt) => {
        try {
          setLastMessage(JSON.parse(evt.data));
        } catch {
          setLastMessage(evt.data);
        }
      };

      ws.onerror = () => {
        setReadyState(WebSocket.CLOSED);
      };

      ws.onclose = () => {
        setReadyState(WebSocket.CLOSED);
        // Exponential back-off: 2s, 4s, 8s … max 30s
        const delay = Math.min(2000 * 2 ** retryRef.current, 30000);
        retryRef.current += 1;
        timerRef.current = setTimeout(connect, delay);
      };
    } catch {
      // WebSocket not available (e.g. test environment) — ignore silently
    }
  }, [path, enabled]);

  useEffect(() => {
    if (!enabled) return;
    connect();
    return () => {
      clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, [connect, enabled]);

  const reconnect = useCallback(() => {
    clearTimeout(timerRef.current);
    wsRef.current?.close();
    retryRef.current = 0;
    connect();
  }, [connect]);

  return { lastMessage, readyState, reconnect };
}

/**
 * useLiveStations — subscribes to /ws/live and merges updates
 * into an existing stations array.
 *
 * @param {Array}   baseStations  — stations from REST API
 * @param {boolean} enabled
 */
export function useLiveStations(baseStations = [], enabled = true) {
  const [stations, setStations] = useState(baseStations);
  const { lastMessage } = useWebSocket("/ws/live", enabled);

  // Seed from REST data
  useEffect(() => {
    if (baseStations.length > 0) setStations(baseStations);
  }, [baseStations]);

  // Merge WS updates
  useEffect(() => {
    if (!lastMessage?.stations) return;
    setStations((prev) => {
      const map = new Map(prev.map((s) => [s.station_id, s]));
      lastMessage.stations.forEach((s) => map.set(s.station_id, { ...map.get(s.station_id), ...s }));
      return Array.from(map.values());
    });
  }, [lastMessage]);

  return stations;
}

/**
 * useAlerts — subscribes to /ws/alerts and accumulates alert events.
 * @param {boolean} enabled
 */
export function useAlerts(enabled = true) {
  const [alerts, setAlerts] = useState([]);
  const { lastMessage } = useWebSocket("/ws/alerts", enabled);

  useEffect(() => {
    if (!lastMessage?.alerts) return;
    setAlerts((prev) => [
      ...lastMessage.alerts.map((a) => ({ ...a, seenAt: Date.now() })),
      ...prev.slice(0, 49),   // keep last 50
    ]);
  }, [lastMessage]);

  const dismiss = useCallback((stationId) => {
    setAlerts((prev) => prev.filter((a) => a.station_id !== stationId));
  }, []);

  return { alerts, dismiss };
}
