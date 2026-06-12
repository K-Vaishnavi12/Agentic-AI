"""Role-based access control — each stakeholder sees only allowed resources."""
from enum import Enum
from typing import Set


class Role(str, Enum):
    CHAIRMAN = "chairman"
    INVESTOR = "investor"
    SHAREHOLDER = "shareholder"
    LEGAL = "legal"
    FINANCE = "finance"


# Which agent outputs / resources each role can access
ROLE_RESOURCES: dict[Role, Set[str]] = {
    Role.CHAIRMAN: {
        "executive_briefing",
        "stock_agent",
        "legal_agent",
        "startup_agent",
        "risk_agent",
        "shareholder_agent",
    },
    Role.INVESTOR: {"stock_agent", "risk_agent", "alerts"},
    Role.SHAREHOLDER: {"shareholder_agent", "dividends", "equity", "board_summary"},
    Role.LEGAL: {"legal_agent", "compliance", "regulatory"},
    Role.FINANCE: {"risk_agent", "stock_agent", "financial_summary"},
}


def get_allowed_resources(role: Role) -> Set[str]:
    return ROLE_RESOURCES.get(role, set()).copy()


def check_access(role: Role, resource: str, strict: bool = True) -> bool:
    allowed = get_allowed_resources(role)
    if not strict:
        return True
    return resource in allowed
