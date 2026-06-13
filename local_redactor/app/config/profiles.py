"""Rule profiles — named bundles of words, enabled patterns, and GLiNER labels.

Profiles are presets the user can pick and then customise. They never contain
document text — only rule configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Profile:
    name: str
    description: str
    gliner_labels: List[str] = field(default_factory=list)
    #: Regex pattern names to enable (see detectors/regex_de.py).
    regex_patterns: List[str] = field(default_factory=list)
    include_opt_in_regex: bool = False
    suggested_words: List[str] = field(default_factory=list)


_BASE_GOV_IDS = ["iban", "steuer_id", "sozialversicherung", "egk", "bic", "phone_de", "email"]
_BASE_LABELS = ["person", "address", "phone_number", "email", "date_of_birth"]


PRESETS: Dict[str, Profile] = {
    "default_de": Profile(
        name="Default German Privacy",
        description="Names, addresses, contact details, common German identifiers.",
        gliner_labels=_BASE_LABELS + ["bank_account", "government_id"],
        regex_patterns=_BASE_GOV_IDS + ["plz", "date_numeric", "kfz"],
    ),
    "medical": Profile(
        name="Medical",
        description="Patient identity plus health insurance IDs and diagnoses context.",
        gliner_labels=_BASE_LABELS + ["health_insurance_id", "medical_condition", "government_id"],
        regex_patterns=_BASE_GOV_IDS + ["date_numeric"],
    ),
    "insurance": Profile(
        name="Insurance",
        description="Policyholder identity, policy/claim numbers, bank details.",
        gliner_labels=_BASE_LABELS + ["bank_account", "company", "government_id"],
        regex_patterns=_BASE_GOV_IDS + ["credit_card", "date_numeric"],
    ),
    "legal": Profile(
        name="Legal",
        description="Parties, case references, addresses; conservative on dates.",
        gliner_labels=_BASE_LABELS + ["company", "government_id"],
        regex_patterns=_BASE_GOV_IDS + ["date_numeric", "kfz"],
    ),
    "hr": Profile(
        name="HR",
        description="Employee identity, tax/social-security IDs, bank details.",
        gliner_labels=_BASE_LABELS + ["bank_account", "government_id", "company"],
        regex_patterns=_BASE_GOV_IDS + ["steuernummer", "date_numeric"],
        include_opt_in_regex=True,
    ),
    "university": Profile(
        name="University",
        description="Student identity, matriculation numbers, contact details.",
        gliner_labels=_BASE_LABELS + ["government_id"],
        regex_patterns=["email", "phone_de", "date_numeric"],
    ),
    "custom": Profile(
        name="Custom",
        description="Empty starting point — add your own words, patterns, labels.",
        gliner_labels=[],
        regex_patterns=[],
    ),
}


def get_preset(key: str) -> Profile:
    return PRESETS[key]
