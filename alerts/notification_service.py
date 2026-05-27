"""
alerts/notification_service.py
Sends email, webhook, and in-app alerts when:
  - Predicted wait time exceeds threshold
  - Station utilisation is critical (>90%)
  - A station goes offline
  - Queue length spikes above normal
Supports: SMTP email · Slack webhook · Generic HTTP webhook
"""

import logging
import smtplib
import json
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    ALERT_UTILIZATION_THRESHOLD, ALERT_WAIT_THRESHOLD_MIN,
    SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER,
)

logger = logging.getLogger(__name__)


# ─── Alert Types ──────────────────────────────────────────────────────────────

class Alert:
    """Represents a triggered alert."""

    SEVERITY_INFO = "INFO"
    SEVERITY_WARNING = "WARNING"
    SEVERITY_CRITICAL = "CRITICAL"

    def __init__(
        self,
        station_id: int,
        station_name: str,
        alert_type: str,
        message: str,
        severity: str,
        data: Optional[Dict] = None,
    ):
        self.station_id = station_id
        self.station_name = station_name
        self.alert_type = alert_type
        self.message = message
        self.severity = severity
        self.data = data or {}
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "alert_type": self.alert_type,
            "message": self.message,
            "severity": self.severity,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_emoji_str(self) -> str:
        icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(self.severity, "📢")
        return f"{icon} [{self.severity}] {self.station_name}: {self.message}"


# ─── Alert Rules ──────────────────────────────────────────────────────────────

class AlertRuleEngine:
    """Evaluates predictions against alert thresholds."""

    def __init__(
        self,
        wait_threshold_min: float = ALERT_WAIT_THRESHOLD_MIN,
        util_threshold: float = ALERT_UTILIZATION_THRESHOLD,
    ):
        self.wait_threshold = wait_threshold_min
        self.util_threshold = util_threshold

    def evaluate(self, station: Dict, prediction: Dict) -> List[Alert]:
        """Return list of triggered alerts for one station."""
        alerts = []
        sid = station.get("station_id", 0)
        name = station.get("name", f"Station {sid}")
        wait = prediction.get("predicted_wait_min", 0)
        util = prediction.get("utilization_pct", 0) / 100
        status = station.get("status", "Unknown")

        # Wait-time alert
        if wait >= self.wait_threshold:
            severity = Alert.SEVERITY_CRITICAL if wait >= self.wait_threshold * 2 else Alert.SEVERITY_WARNING
            alerts.append(Alert(
                station_id=sid, station_name=name,
                alert_type="HIGH_WAIT_TIME",
                message=f"Predicted wait time {wait:.1f} min exceeds threshold ({self.wait_threshold} min)",
                severity=severity,
                data={"predicted_wait_min": wait, "threshold_min": self.wait_threshold},
            ))

        # Utilisation alert
        if util >= self.util_threshold:
            alerts.append(Alert(
                station_id=sid, station_name=name,
                alert_type="HIGH_UTILIZATION",
                message=f"Station utilisation {util * 100:.1f}% exceeds threshold ({self.util_threshold * 100:.0f}%)",
                severity=Alert.SEVERITY_WARNING,
                data={"utilization_pct": util * 100},
            ))

        # Offline alert
        if "Offline" in str(status):
            alerts.append(Alert(
                station_id=sid, station_name=name,
                alert_type="STATION_OFFLINE",
                message="Station is offline or unreachable",
                severity=Alert.SEVERITY_CRITICAL,
                data={"status": status},
            ))

        return alerts

    def evaluate_network(
        self,
        stations: List[Dict],
        predictions: List[Dict],
    ) -> List[Alert]:
        """Evaluate all stations and aggregate alerts."""
        all_alerts = []
        pred_lookup = {p.get("station_id"): p for p in predictions}
        for station in stations:
            sid = station.get("station_id")
            pred = pred_lookup.get(sid, {})
            all_alerts.extend(self.evaluate(station, pred))
        return all_alerts


# ─── Notification Channels ────────────────────────────────────────────────────

class EmailNotifier:
    """SMTP email notifications."""

    def __init__(self, recipient_email: str):
        self.recipient = recipient_email

    def send(self, alerts: List[Alert]) -> bool:
        if not alerts or not SMTP_USER:
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"⚡ EV ChargeSmart: {len(alerts)} Alert(s) Triggered"
            msg["From"] = SMTP_USER
            msg["To"] = self.recipient

            body_lines = [f"<p>{a.to_emoji_str()}</p>" for a in alerts]
            html = f"""
            <html><body style="font-family:sans-serif;">
            <h2>⚡ EV ChargeSmart Alerts — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</h2>
            {''.join(body_lines)}
            <hr><p style="color:gray;">EV ChargeSmart Monitoring System</p>
            </body></html>
            """
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, self.recipient, msg.as_string())
            logger.info(f"Email alert sent to {self.recipient} ({len(alerts)} alerts)")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class SlackNotifier:
    """Slack incoming webhook notifications."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, alerts: List[Alert]) -> bool:
        if not alerts or not self.webhook_url:
            return False
        try:
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"⚡ EV ChargeSmart: {len(alerts)} Alert(s)",
                    },
                }
            ]
            for a in alerts[:10]:   # Slack limits
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{a.severity}* | {a.station_name}\n{a.message}",
                    },
                })
            payload = {"blocks": blocks}
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info(f"Slack alert sent ({len(alerts)} alerts)")
            return True
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False


class WebhookNotifier:
    """Generic HTTP webhook (for custom integrations)."""

    def __init__(self, url: str, headers: Optional[Dict] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, alerts: List[Alert]) -> bool:
        if not alerts:
            return True
        try:
            payload = {
                "source": "ev_chargesmart",
                "timestamp": datetime.utcnow().isoformat(),
                "alerts": [a.to_dict() for a in alerts],
                "count": len(alerts),
            }
            resp = requests.post(self.url, json=payload, headers=self.headers, timeout=10)
            resp.raise_for_status()
            logger.info(f"Webhook sent: {len(alerts)} alerts to {self.url}")
            return True
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False


# ─── Notification Service ─────────────────────────────────────────────────────

class NotificationService:
    """
    Master notification orchestrator.
    Evaluates rules, deduplicates alerts, and dispatches through all channels.
    """

    def __init__(
        self,
        email_recipient: Optional[str] = None,
        slack_webhook: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ):
        self.rule_engine = AlertRuleEngine()
        self.channels = []
        self._alert_history: List[Alert] = []
        self._cooldown: Dict[str, float] = {}
        self._cooldown_seconds = 600   # 10 min per alert type per station

        if email_recipient:
            self.channels.append(EmailNotifier(email_recipient))
        if slack_webhook:
            self.channels.append(SlackNotifier(slack_webhook))
        if webhook_url:
            self.channels.append(WebhookNotifier(webhook_url))

    def _is_in_cooldown(self, alert: Alert) -> bool:
        import time
        key = f"{alert.station_id}:{alert.alert_type}"
        last = self._cooldown.get(key, 0)
        return (time.time() - last) < self._cooldown_seconds

    def _mark_cooldown(self, alert: Alert) -> None:
        import time
        key = f"{alert.station_id}:{alert.alert_type}"
        self._cooldown[key] = time.time()

    def process(
        self,
        stations: List[Dict],
        predictions: List[Dict],
    ) -> List[Alert]:
        """Evaluate rules and dispatch non-cooldown alerts."""
        raw_alerts = self.rule_engine.evaluate_network(stations, predictions)

        # Filter cooldown
        fresh_alerts = [a for a in raw_alerts if not self._is_in_cooldown(a)]
        for a in fresh_alerts:
            self._mark_cooldown(a)

        self._alert_history.extend(fresh_alerts)
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]

        if fresh_alerts:
            logger.warning(f"Dispatching {len(fresh_alerts)} alert(s)")
            for channel in self.channels:
                channel.send(fresh_alerts)
            for a in fresh_alerts:
                logger.warning(a.to_emoji_str())
        else:
            logger.debug("No new alerts to dispatch")

        return fresh_alerts

    def get_history(self, last_n: int = 50) -> List[Dict]:
        return [a.to_dict() for a in self._alert_history[-last_n:]]


if __name__ == "__main__":
    # Demo
    service = NotificationService()
    stations = [
        {"station_id": 1, "name": "Downtown Hub", "status": "Operational"},
        {"station_id": 2, "name": "Airport Terminal", "status": "Offline"},
    ]
    predictions = [
        {"station_id": 1, "predicted_wait_min": 55, "utilization_pct": 96},
        {"station_id": 2, "predicted_wait_min": 0, "utilization_pct": 0},
    ]
    alerts = service.process(stations, predictions)
    for a in alerts:
        print(a.to_emoji_str())
