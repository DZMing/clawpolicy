"""Public package wrapper for ClawPolicy.

The clawpolicy package provides the stable public API for external integrations.
The lib module contains internal implementation details and should not be imported directly.
"""

from lib.api import ConfirmationAPI, create_api
from lib.md_to_policy import MarkdownToPolicyConverter
from lib.policy_models import PolicyEvent, Playbook, Rule
from lib.policy_store import PolicyStore
from lib.policy_to_md import PolicyToMarkdownExporter
from lib import __version__

__all__ = [
    "__version__",
    "ConfirmationAPI",
    "PolicyEvent",
    "PolicyStore",
    "Playbook",
    "Rule",
    "MarkdownToPolicyConverter",
    "PolicyToMarkdownExporter",
    "create_api",
]
