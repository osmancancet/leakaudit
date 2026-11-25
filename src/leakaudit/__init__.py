"""LeakAudit: leakage and shortcut audit for labelled text corpora."""
__version__ = "0.1.0"

from .audit import AuditConfig, Thresholds, audit  # noqa: E402
from .report import to_markdown  # noqa: E402

__all__ = ["audit", "AuditConfig", "Thresholds", "to_markdown", "__version__"]
