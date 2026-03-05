"""
Intelligent Search API – minimal main.
Exposes GET /ask-ai (delegates to Intelligent search.run_agentic_flow) plus /, /health, /alerts.
Run: python main.py  or  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

_main_dir = Path(__file__).resolve().parent
if str(_main_dir) not in sys.path:
    sys.path.insert(0, str(_main_dir))
try:
    os.chdir(_main_dir)
except Exception:
    pass

for env_path in [_main_dir / ".env", Path.cwd() / ".env", _main_dir.parent / ".env"]:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        break
load_dotenv(override=False)

_logger = logging.getLogger("main")

app = FastAPI(title="Intelligent Search API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Legacy fallback when Intelligent search.py is not loadable (nav-only)
# Kept in sync with Intelligent search.py ROUTE_MAP
# ---------------------------------------------------------------------------
ROUTE_MAP = {
    "profit": "/financial-reports/profit-loss",
    "loss": "/financial-reports/profit-loss",
    "ledger": "/ledger",
    "ledgers": "/ledgers",
    "gl posting": "/gl-postings",
    "journal": "/journal",
    "chart of accounts": "/chart-of-accounts",
    "invoice": "/ar/invoice",
    "balance sheet": "/financial-reports/balance-sheet",
    "trial balance": "/financial-reports/trial-balance",
    "bank reconciliation": "/bank/reconciliation",
    "employee": "/hr/employees",
    "payroll": "/hr/payroll",
    "attendance": "/hr/attendance",
    "timesheet": "/hr/timesheets",
    "leave": "/hr/leaves",
    "organization": "/hr/organization",
    "check-in": "/front-office/checkin",
    "checkin": "/front-office/checkin",
    "reservation": "/front-office/reservations",
    "walkin": "/front-office/reservations/walkins",
    "ota": "/front-office/reservations/ota",
    "group booking": "/front-office/reservations/group-booking",
    "room": "/hotel/rooms",
    "room type": "/hotel/room-type",
    "material": "/materials/MaterialMasterManager",
    "stock": "/materials/MaterialMasterManager",
    "inventory": "/materials/MaterialMasterManager",
    "add stock": "/materials/MaterialMasterManager",
    "reorder": "/materials/POManager",
    "vendor": "/ap/vendors/list",
    "purchase order": "/materials/POManager",
    "purchase req": "/materials/PRManager",
    "restaurant": "/restaurant/restaurants",
    "restaurant order": "/restaurant/restaurants",
    "restaurant table": "/restaurant/restaurants",
    "asset": "/assets",
    "work order": "/maintenance/workorder",
    "company": "/company",
    "branch": "/hotel/branches",
    "guest": "/front-office/reservations",
    "folio": "/financials/guest-folio-invoice",
    "guest billing": "/financials/guest-billing",
    "housekeeping": "/housekeeping/dashboard",
    "checkout": "/front-office/checkout",
    "ar customer": "/ar/customers",
    "bank": "/bank/master",
    "cost center": "/cost-center",
    "general ledger": "/ledger",
    "gl": "/ledger",
    "extend stay": "/front-office/reservations",
    "front desk": "/front-office/reservations",
    "cleaning supplies": "/materials/MaterialMasterManager",
    "running low": "/materials/MaterialMasterManager",
    "salary slip": "/hr/payroll",
    "generate salary": "/hr/payroll",
    "empty room": "/hotel/rooms",
    "vacant": "/hotel/rooms",
    "new vendor": "/ap/vendors/list",
    "register vendor": "/ap/vendors/list",
    "move employee": "/hr/employees",
    "transfer employee": "/hr/employees",
    "department": "/hr/employees",
    "outstanding": "/ar/invoice",
    "suppliers": "/ap/vendors/list",
    "autopricing": "/auto-pricing",
    "auto pricing": "/auto-pricing",
    "bill": "/restaurant/restaurants",
    "bill amount": "/restaurant/restaurants",
    "bill amount of room": "/financials/guest-billing",
    "room bill": "/financials/guest-billing",
    "pay for employee": "/hr/payroll",
    "pay employee": "/hr/payroll",
    "i need to pay": "/hr/payroll",
}

_NAVIGATION_PHRASES = (
    "go to ", "open ", "take me to ", "navigate to ", "show me the ",
    "show ", "take me ", "want to go ", "go to the ", "open the ",
    "extend stay", "cleaning supplies", "salary slip", "generate the salary", "empty room",
    "new vendor", "register vendor", "move employee", "transfer employee", "outstanding amount",
    "pay for employee", "pay employee", "i need to pay",
)

QUESTION_STARTS = (
    "how many", "how much", "what is", "what are", "list ", "show me all",
    "count ", "total ", "which ", "who ", "when ", "why ", "give me ",
    "number of", "amount of", "sum of", "average ",
    "do we ", "do i ", "should we ", "can we ", "need to ", "have we ",
    "are we ", "is there ", "do we need", "need we ", "add stock", "reorder ",
)


def _get_navigate_route(query: str) -> Optional[str]:
    q = (query or "").lower().strip()
    if not q:
        return None
    q = q.replace("&", " and ").replace("p&l", "profit loss").replace("p and l", "profit loss")
    for key in sorted(ROUTE_MAP.keys(), key=len, reverse=True):
        if key in q:
            return ROUTE_MAP[key]
    return None


def _is_navigation_only(query: str) -> bool:
    if not query or not query.strip():
        return False
    q = (query or "").lower().strip()
    q = q.replace("&", " and ").replace("p&l", "profit loss")
    if _get_navigate_route(query) is None:
        return False
    for start in QUESTION_STARTS:
        if q.startswith(start) or f" {start}" in q:
            return False
    if len(q.split()) <= 5:
        return True
    for phrase in _NAVIGATION_PHRASES:
        if phrase in q:
            return True
    return False


def _friendly_page_name(route: str) -> str:
    if not route:
        return "page"
    part = route.strip("/").replace("-", " ").split("/")[-1]
    return part.title() if part else "page"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "Intelligent Search API",
        "version": "1.0.0",
        "endpoints": {
            "/ask-ai": "GET - Natural language query (query=...&session_id=...)",
            "/health": "GET - Health check",
            "/alerts": "GET - Alerts (placeholder)",
            "/docs": "GET - API documentation",
        },
    }


@app.get("/ask-ai")
async def ask_ai(
    query: str = Query(..., description="Natural language query or navigation"),
    session_id: Optional[str] = Query(None, description="Optional session for conversation memory"),
) -> Dict[str, Any]:
    """
    Multi-agent ERP: Navigation (0 LLM), quick SQL agents, Pure RAG vector search,
    then Planner READ/WRITE with LLM. Powered by Intelligent search.py.
    """
    try:
        import importlib.util
        path = _main_dir / "Intelligent search.py"
        spec = importlib.util.spec_from_file_location("intelligent_search", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = await mod.run_agentic_flow(session_id or "default", query)
        if result.get("type") == "NAVIGATE" and result.get("route"):
            result["navigateTo"] = result["route"]
        if result.get("navigateTo") is None and result.get("route"):
            result["navigateTo"] = result["route"]
        return result
    except (ImportError, FileNotFoundError):
        pass
    except Exception as e:
        _logger.exception("Agentic ERP error")
        nav = _get_navigate_route(query)
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            out = {
                "type": "ERROR",
                "content": "API quota reached. Please try again in a minute.",
                "thought_process": None,
                "sql_query": None,
                "action_taken": None,
                "next_steps": [],
            }
            if nav:
                out["navigateTo"] = nav
            return out
        out = {
            "type": "ERROR",
            "content": f"AI Error: {err_str}",
            "thought_process": None,
            "sql_query": None,
            "action_taken": None,
            "next_steps": [],
        }
        if nav:
            out["navigateTo"] = nav
        return out

    # Fallback when Intelligent search.py not loadable: navigation-only
    nav_route = _get_navigate_route(query)
    if nav_route is not None and _is_navigation_only(query):
        return {
            "type": "NAVIGATE",
            "route": nav_route,
            "navigateTo": nav_route,
            "content": "Navigating...",
            "thought_process": "Navigation intent; bypassing LLM (0 tokens).",
            "sql_query": None,
            "action_taken": "NAVIGATE",
            "next_steps": [],
            "tableData": None,
            "source": "Intelligent Search",
        }
    return {
        "type": "ERROR",
        "content": "Intelligent search module is required. Ensure 'Intelligent search.py' is present and dependencies (langchain_google_genai, sqlalchemy, rapidfuzz, etc.) are installed.",
        "thought_process": None,
        "sql_query": None,
        "action_taken": None,
        "next_steps": [],
    }


@app.get("/alerts")
async def get_alerts():
    return {"alerts": []}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
