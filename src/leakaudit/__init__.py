"""LeakAudit: leakage and shortcut audit for labelled text corpora."""
__version__ = "0.2.0"

from .audit import AuditConfig, ContentAwareThresholds, Thresholds, audit, content_aware_certificate  # noqa: E402
from .report import to_markdown  # noqa: E402

__all__ = ["audit", "AuditConfig", "Thresholds", "ContentAwareThresholds", "content_aware_certificate", "to_markdown", "__version__"]
