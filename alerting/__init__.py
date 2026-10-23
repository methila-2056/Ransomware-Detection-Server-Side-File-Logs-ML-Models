"""Alert delivery sinks.

The base paper claims integration with SIEM and EDR infrastructure (Sect. 6.1)
without providing any. This package supplies it.
"""

from alerting.sinks import (  # noqa: F401
    AlertDeduplicator,
    AlertManager,
    AlertSink,
    EmailSink,
    LogSink,
    SyslogSink,
    WebhookSink,
    from_config,
)

__all__ = [
    "AlertDeduplicator",
    "AlertManager",
    "AlertSink",
    "EmailSink",
    "LogSink",
    "SyslogSink",
    "WebhookSink",
    "from_config",
]
