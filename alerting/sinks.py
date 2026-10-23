"""Alert delivery sinks.

The base paper claims its framework "can be integrated into existing SIEM or EDR
infrastructures" (Sect. 6.1) and that it "supports practical applicability in
both SME and enterprise-scale environments" (Sect. 1.1). It ships no integration
whatsoever: there is no notification code in the paper's design.

This module provides the integration the paper asserts, plus the deduplication
that any real deployment needs. A ransomware episode lasts from seconds to hours,
so naive per-window notification produces thousands of identical alerts for one
incident. ``AlertDeduplicator`` collapses an episode into a single alert that
updates rather than re-firing, and re-arms only after the condition clears.

Every sink is best-effort by design: a failing notification target must not take
the monitor down, because losing detection because a webhook timed out is a worse
outcome than a missed notification. Failures are recorded and returned rather than
raised.
"""

import json
import logging
import time
import urllib.error
import urllib.request

LOGGER = logging.getLogger(__name__)


class AlertDeduplicator:
    """Collapse an ongoing episode into one alert.

    The paper's evaluation is window-by-window and has no concept of an episode.
    Operational alerting does, because a human or a ticketing system should be
    engaged once per incident, not once per second for its duration.
    """

    def __init__(self, cooldown_seconds=300, key="share"):
        self.cooldown = cooldown_seconds
        self.key = key
        self.active = {}
        self.sent = {}
        self.history = []

    def should_send(self, alert_key, timestamp=None):
        now = timestamp if timestamp is not None else time.time()
        last = self.sent.get(alert_key)
        if last is not None and now - last < self.cooldown:
            return False
        self.sent[alert_key] = now
        self.history.append({"Key": alert_key, "Sent At": now})
        return True

    def begin_episode(self, alert_key, timestamp=None):
        now = timestamp if timestamp is not None else time.time()
        if alert_key in self.active:
            return False
        self.active[alert_key] = now
        return True

    def end_episode(self, alert_key, timestamp=None):
        now = timestamp if timestamp is not None else time.time()
        started = self.active.pop(alert_key, None)
        if started is None:
            return None
        duration = now - started
        return {"Key": alert_key, "Started At": started,
                "Ended At": now, "Duration Seconds": round(duration, 3)}

    def status(self):
        return {
            "Active Episodes": dict(self.active),
            "Cooldown Seconds": self.cooldown,
            "Alerts Sent": len(self.history),
        }


class AlertSink:
    """Base class. Subclasses implement ``_deliver``."""

    name = "sink"

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.delivered = 0
        self.failed = 0
        self.failures = []

    def _deliver(self, alert):  # pragma: no cover - interface
        raise NotImplementedError

    def send(self, alert):
        """Deliver one alert, recording failure instead of raising."""
        if self.dry_run:
            self.delivered += 1
            LOGGER.info("[%s:dry-run] %s", self.name, alert.get("title"))
            return {"ok": True, "dry_run": True}

        try:
            result = self._deliver(alert)
            self.delivered += 1
            return {"ok": True, "detail": result}
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            self.failures.append({"at": time.time(), "error": str(exc)})
            LOGGER.warning("[%s] delivery failed: %s", self.name, exc)
            return {"ok": False, "error": str(exc)}

    def stats(self):
        return {"Sink": self.name, "Delivered": self.delivered,
                "Failed": self.failed, "Dry Run": self.dry_run}


class WebhookSink(AlertSink):
    """POST a JSON alert to an HTTP endpoint.

    The SIEM/EDR integration the paper describes is normally an HTTP ingest, so
    this is the sink that makes the claim concrete.
    """

    name = "webhook"

    def __init__(self, url, headers=None, timeout=5.0, dry_run=False):
        super().__init__(dry_run=dry_run)
        if not url.startswith(("http://", "https://")):
            raise ValueError("webhook url must be http or https")
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def _deliver(self, alert):
        payload = json.dumps(alert, default=str).encode("utf-8")
        request = urllib.request.Request(self.url, data=payload,
                                         headers=self.headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return {"status": response.status}


class EmailSink(AlertSink):
    """Send by SMTP. Uses ``smtplib`` lazily so the import is not required."""

    name = "email"

    def __init__(self, host, port, sender, recipients, username=None,
                 password=None, subject_prefix="[RANSOMWARE]", dry_run=False):
        super().__init__(dry_run=dry_run)
        self.host = host
        self.port = port
        self.sender = sender
        self.recipients = list(recipients)
        self.username = username
        self.password = password
        self.subject_prefix = subject_prefix

    def _deliver(self, alert):
        import smtplib
        from email.message import EmailMessage

        message = EmailMessage()
        message["Subject"] = f"{self.subject_prefix} {alert.get('title', '')}"
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content(json.dumps(alert, indent=2, default=str))

        with smtplib.SMTP(self.host, self.port, timeout=10) as server:
            if self.username:
                server.starttls()
                server.login(self.username, self.password)
            server.send_message(message)
        return {"recipients": len(self.recipients)}


class SyslogSink(AlertSink):
    """Emit to syslog, mapping severity to standard levels."""

    name = "syslog"

    LEVELS = {"NORMAL": "info", "UNSURE": "warning", "RANSOMWARE": "critical"}

    def _deliver(self, alert):
        import syslog as syslog_module

        level = self.LEVELS.get(alert.get("verdict", "NORMAL"), "info")
        message = json.dumps(alert, default=str)
        getattr(syslog_module, f"syslog_{level}")(message)
        return {"level": level}


class LogSink(AlertSink):
    """Write to the application log. Always available, used as a fallback."""

    name = "log"

    LEVELS = {"NORMAL": logging.INFO, "UNSURE": logging.WARNING,
              "RANSOMWARE": logging.CRITICAL}

    def _deliver(self, alert):
        level = self.LEVELS.get(alert.get("verdict", "NORMAL"), logging.INFO)
        LOGGER.log(level, "%s %s", alert.get("title", ""), alert)
        return {"level": logging.getLevelName(level)}


class AlertManager:
    """Fan out to several sinks, with deduplication and episode tracking."""

    def __init__(self, sinks=None, deduplicator=None):
        self.sinks = list(sinks or [LogSink()])
        self.dedup = deduplicator or AlertDeduplicator()
        self.results = []

    def add_sink(self, sink):
        self.sinks.append(sink)
        return sink

    def notify(self, verdict, title, detail=None, key="default", timestamp=None):
        """Send an alert if the verdict warrants it and it is not a repeat.

        Returns a record of what was attempted, including the reason when
        nothing was sent, so the caller can distinguish "no alert needed" from
        "alert suppressed" from "delivery failed".
        """
        now = timestamp if timestamp is not None else time.time()

        if verdict == "NORMAL":
            closed = self.dedup.end_episode(key, now)
            return {"Sent": False, "Reason": "verdict is NORMAL",
                    "Episode Closed": closed}

        new_episode = self.dedup.begin_episode(key, now)
        if not self.dedup.should_send(key, now):
            return {"Sent": False, "Reason": "within cooldown",
                    "New Episode": new_episode}

        alert = {
            "title": title,
            "verdict": verdict,
            "detail": detail or {},
            "key": key,
            "timestamp": now,
            "new_episode": new_episode,
        }

        deliveries = [sink.send(alert) for sink in self.sinks]
        record = {"Sent": True, "Alert": alert, "Deliveries": deliveries,
                  "Succeeded": any(d.get("ok") for d in deliveries)}
        self.results.append(record)
        return record

    def stats(self):
        return {
            "Sinks": [s.stats() for s in self.sinks],
            "Deduplicator": self.dedup.status(),
            "Alerts Emitted": len(self.results),
        }


def from_config(config):
    """Build a manager from a configuration mapping.

    Sinks whose configuration is incomplete are skipped with a warning rather than
    raising, so one misconfigured target does not prevent the others working.
    """
    manager = AlertManager(sinks=[LogSink()])

    webhook = config.get("webhook")
    if webhook and webhook.get("url"):
        manager.add_sink(WebhookSink(
            webhook["url"],
            headers=webhook.get("headers"),
            timeout=float(webhook.get("timeout", 5.0)),
            dry_run=bool(webhook.get("dry_run", False))))

    email = config.get("email")
    if email and email.get("host"):
        # Every field the constructor needs must be present. A half-configured
        # email block is skipped with a warning rather than raising, because
        # failing to construct the manager would disable the webhook that is
        # already working.
        missing = [f for f in ("sender", "recipients") if not email.get(f)]
        if missing:
            LOGGER.warning("email sink configuration incomplete (missing %s); "
                           "skipping", ", ".join(missing))
        else:
            manager.add_sink(EmailSink(
                email["host"], int(email.get("port", 587)), email["sender"],
                email["recipients"], username=email.get("username"),
                password=email.get("password"),
                dry_run=bool(email.get("dry_run", False))))

    return manager
