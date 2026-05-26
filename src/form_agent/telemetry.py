"""Application Insights / OpenTelemetry wiring.

Call ``setup_telemetry()`` once at process startup (done by ``FormAgent.__init__``).
No-op when ``APPLICATIONINSIGHTS_CONNECTION_STRING`` is not set, so tests and
local runs without Azure are unaffected.
"""
from __future__ import annotations

import logging

from opentelemetry import trace

logger = logging.getLogger(__name__)

# Module-level tracer — always safe to import; spans become no-ops when no
# exporter is configured.
tracer: trace.Tracer = trace.get_tracer("form_agent")

_configured = False


def setup_telemetry(connection_string: str | None = None) -> None:
    """Configure Azure Monitor exporter if a connection string is available.

    Idempotent — safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    from .config import CONFIG

    cs = connection_string or CONFIG.appinsights_connection_string
    if not cs:
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=cs)
        logger.info("Azure Monitor telemetry configured.")
    except Exception as exc:  # pragma: no cover — only fails when SDK missing/misconfigured
        logger.warning("Failed to configure Azure Monitor telemetry: %s", exc)
        return

    _configured = True
