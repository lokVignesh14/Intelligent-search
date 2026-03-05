"""
Intelligent Search – Multi-Agent ERP (NLP + Pure RAG + LLM).
Understands the question and replies from input:
  1. Navigation agent: semantic_router → NAVIGATE (0 LLM).
  2. Quick SQL agents (NLP-style triggers, 0 LLM):
     - Stock/reorder, room bill, table bill.
  3. Pure RAG – Vector search: "luxury stay" etc. → semantic search over room descriptions
     (no "luxury" column in DB; embeddings match by meaning).
  4. Planner + Read/Write (LLM/RAG): READ/WRITE → schema + LLM.
- GET /ask-ai?query=...&session_id=...  → main.py (run_agentic_flow)
- Run: python main.py  or  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import re
import json
import asyncio
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from rapidfuzz import process

# Ensure this directory is on path when run as script
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Optional: LangChain for planner + SQL
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
    from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

# Optional: Vector search (Pure RAG) – sentence_transformers + FAISS
_VECTOR_DEPS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    _VECTOR_DEPS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    np = None
    faiss = None

# ---------------------------------------------------------------------------
# ROUTE_MAP – semantic router (navigation). Only used when query is nav-only.
# Data questions (stock, bill, etc.) are answered by quick agents or planner/LLM.
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

# Long queries with these + a route → navigate (no LLM). Data questions use QUESTION_STARTS so we answer instead.
ACTION_NAVIGATION_PHRASES = (
    "extend stay", "cleaning supplies", "salary slip", "generate the salary", "empty room",
    "new vendor", "register vendor", "move employee", "transfer employee", "outstanding amount",
    "pay for employee", "pay employee", "i need to pay",
)

NAVIGATION_PHRASES = (
    "go to ", "open ", "take me to ", "navigate to ", "show me the ",
    "show ", "take me ", "want to go ", "go to the ", "open the ",
)
# NLP-style: phrases that indicate a real question (answer with data, don't nav-only)
QUESTION_STARTS = (
    "how many", "how much", "what is", "what are", "list ", "show me all",
    "count ", "total ", "which ", "who ", "when ", "why ", "give me ",
    "number of", "amount of", "sum of", "average ",
    "do we ", "do i ", "should we ", "can we ", "need to ", "have we ",
    "are we ", "is there ", "do we need", "need we ", "add stock", "reorder ",
)

POSTGRES_QUOTE_RULE = (
    " CRITICAL: PostgreSQL with case-sensitive identifiers. "
    'Wrap every table and column name in double quotes (e.g. "Employee"). '
    "Never use unquoted identifiers."
)

PERSONA = (
    "You are a Senior ERP System Architect. Your role is to ensure the highest "
    "reliability and security in SQL generation. Always use double-quoted identifiers "
    "for PostgreSQL. Prefer SELECT for reads; for writes, verify data exists before UPDATE/DELETE."
)

_conversation_memory: Dict[str, List[Dict[str, str]]] = {}
_MEMORY_MAX_TURNS = 10

_engine: Optional[Engine] = None
_sql_agent = None
_llm = None
_db = None

# Vector search (Pure RAG): room index
_embedding_model = None
_room_index_faiss = None
_room_metadata: List[Dict[str, Any]] = []  # list of {id, roomNumber, room_type, basePrice, text}


def _norm(q: str) -> str:
    if not q:
        return ""
    q = (q or "").lower().strip()
    return q.replace("&", " and ").replace("p&l", "profit loss").replace("p and l", "profit loss")


def _get_route_exact(query: str) -> Optional[str]:
    q = _norm(query)
    if not q:
        return None
    for key in sorted(ROUTE_MAP.keys(), key=len, reverse=True):
        if key in q:
            return ROUTE_MAP[key]
    return None


def _fuzzy_route(query: str, score_cutoff: int = 75) -> Optional[Tuple[str, str]]:
    q = _norm(query)
    if not q or not q.strip():
        return None
    result = process.extractOne(q, list(ROUTE_MAP.keys()), score_cutoff=score_cutoff)
    if result is None:
        return None
    key, _, _ = result
    return (key, ROUTE_MAP[key])


def _is_navigation_only(query: str) -> bool:
    if not query or not query.strip():
        return False
    q = _norm(query)
    if _get_route_exact(query) is None and _fuzzy_route(query) is None:
        return False
    for start in QUESTION_STARTS:
        if q.startswith(start) or f" {start}" in q:
            return False
    if len(q.split()) <= 5:
        return True
    for phrase in NAVIGATION_PHRASES:
        if phrase in q:
            return True
    # Long queries that match an action phrase (10 Advanced ERP intents) still navigate without API
    for phrase in ACTION_NAVIGATION_PHRASES:
        if phrase in q:
            return True
    return False


def semantic_router(query: str) -> Optional[Dict[str, Any]]:
    if not query or not query.strip():
        return None
    route = _get_route_exact(query)
    if route is None:
        fuzzy = _fuzzy_route(query)
        if fuzzy is not None:
            _, route = fuzzy
    if route is None or not _is_navigation_only(query):
        return None
    return {
        "type": "NAVIGATE",
        "route": route,
        "content": "Navigating...",
        "thought_process": "Navigation intent detected; bypassing LLM (0 tokens).",
        "sql_query": None,
        "action_taken": "NAVIGATE",
        "next_steps": [],
        "tableData": None,
    }


def get_memory(session_id: str) -> List[Dict[str, str]]:
    if session_id not in _conversation_memory:
        _conversation_memory[session_id] = []
    return _conversation_memory[session_id]


def add_to_memory(session_id: str, role: str, content: str) -> None:
    mem = get_memory(session_id)
    mem.append({"role": role, "content": content})
    if len(mem) > _MEMORY_MAX_TURNS * 2:
        _conversation_memory[session_id] = mem[-_MEMORY_MAX_TURNS * 2:]


def build_context_for_agent(session_id: str) -> str:
    mem = get_memory(session_id)
    if not mem:
        return ""
    lines = [f"{m['role'].upper()}: {m['content']}" for m in mem[-_MEMORY_MAX_TURNS * 2:]]
    return "\n".join(lines)


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        db_url = os.getenv("DB_URL")
        if not db_url:
            raise ValueError("DB_URL is missing in .env")
        _engine = create_engine(db_url)
    return _engine


def _get_llm():
    global _llm
    if _llm is None:
        if not _DEPS_AVAILABLE:
            raise ValueError("langchain_google_genai not available")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing in .env")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        _llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )
    return _llm


def _get_db():
    global _db
    if _db is None:
        if not _DEPS_AVAILABLE:
            raise ValueError("langchain_community not available")
        _db = SQLDatabase(_get_engine())
    return _db


async def _invoke_with_429_retry(invoke_fn, *args, max_retries: int = 1, **kwargs) -> Any:
    """Call invoke_fn; on 429 do not retry (saves quota). max_retries=1 to fail fast."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(invoke_fn, *args, **kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            raise
    raise last_err


def _planner_classify(query: str, context: str) -> str:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", PERSONA + "\n\nYou classify user intent for an ERP. Reply with exactly one word: READ or WRITE.\n"
         "READ: analytical, list, show, count, report, fetch data (SELECT).\n"
         "WRITE: update, insert, delete, cancel, mark, set, change, create, book (INSERT/UPDATE/DELETE)."),
        ("human", "Previous context:\n{context}\n\nCurrent query: {query}\n\nReply only: READ or WRITE"),
    ])
    chain = prompt | llm
    ctx = context or "(no prior context)"
    out = chain.invoke({"context": ctx, "query": query})
    text = out.content.strip().upper() if hasattr(out, "content") else str(out).strip().upper()
    return "WRITE" if "WRITE" in text else "READ"


def _read_tool_execute(sql: str, engine: Engine) -> Tuple[Optional[List[Dict]], Optional[str]]:
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return None, "Only SELECT is allowed in ReadTool"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            keys = result.keys()
            rows = [dict(zip(keys, row)) for row in result.fetchall()]
            return rows, None
    except Exception as e:
        return None, str(e)


def _get_table_schema_hint(engine: Engine) -> str:
    from sqlalchemy import inspect
    insp = inspect(engine)
    tables = insp.get_table_names()
    parts = []
    for t in tables[:30]:
        cols = [c["name"] for c in insp.get_columns(t)]
        cols_str = ", ".join(f'"{c}"' for c in cols)
        parts.append(f'"{t}" ({cols_str})')
    return "\n".join(parts)


def read_tool_with_retry(query: str, session_id: str, max_retries: int = 3) -> Dict[str, Any]:
    engine = _get_engine()
    context = build_context_for_agent(session_id)
    llm = _get_llm()
    schema_hint = _get_table_schema_hint(engine)
    postgres_rule = POSTGRES_QUOTE_RULE
    thought = []
    sql_used = None
    for attempt in range(max_retries):
        prompt = (
            f"{PERSONA}\n{postgres_rule}\n\nRelevant schema (use double-quoted names):\n{schema_hint}\n\n"
            + (f"Previous attempt failed. Generate a corrected SELECT only. Schema reminder:\n{schema_hint}\n\n" if attempt > 0 and schema_hint else "")
            + f"Context:\n{context}\n\nUser question: {query}\n\nReply with only one valid PostgreSQL SELECT statement, no explanation."
        )
        out = llm.invoke(prompt)
        sql_used = out.content.strip() if hasattr(out, "content") else str(out).strip()
        for sep in ("```sql", "```"):
            if sep in sql_used:
                sql_used = sql_used.split(sep)[1] if len(sql_used.split(sep)) > 1 else sql_used
        sql_used = sql_used.strip()
        rows, err = _read_tool_execute(sql_used, engine)
        if err is None:
            thought.append(f"Attempt {attempt + 1}: SQL succeeded.")
            return {"success": True, "rows": rows, "sql_query": sql_used, "thought_process": " ".join(thought), "action_taken": "DB_READ"}
        thought.append(f"Attempt {attempt + 1} failed: {err}")
    return {
        "success": False,
        "rows": None,
        "sql_query": sql_used,
        "thought_process": " ".join(thought),
        "action_taken": "DB_READ",
        "error": thought[-1] if thought else "Max retries exceeded",
    }


def _write_tool_verify_exists(engine: Engine, table: str, where_clause: str) -> bool:
    try:
        sql = f'SELECT 1 FROM "{table}" WHERE {where_clause} LIMIT 1'
        with engine.connect() as conn:
            r = conn.execute(text(sql))
            return r.fetchone() is not None
    except Exception:
        return False


def _parse_table_and_where(sql: str) -> Tuple[Optional[str], Optional[str]]:
    raw = sql.strip()
    sql_upper = raw.upper()
    if sql_upper.startswith("UPDATE"):
        parts = raw.split(None, 2)
        if len(parts) >= 2:
            table = parts[1].strip('"')
            rest = parts[2] if len(parts) > 2 else ""
            if "WHERE" in rest.upper():
                return table, rest.split("WHERE", 1)[1].strip()
    elif sql_upper.startswith("DELETE"):
        rest = raw.split(None, 1)[-1] if len(raw.split()) > 1 else ""
        if "FROM" in rest.upper():
            from_part = rest.split("FROM", 1)[1].strip()
            table = from_part.split()[0].strip('"')
            if "WHERE" in from_part.upper():
                return table, from_part.split("WHERE", 1)[1].strip()
    return None, None


def write_tool_execute(sql: str, engine: Engine, verify_exists: bool = True) -> Tuple[bool, Optional[str]]:
    raw_sql = sql.strip()
    sql_upper = raw_sql.upper()
    if not any(sql_upper.startswith(x) for x in ("INSERT", "UPDATE", "DELETE")):
        return False, "Only INSERT, UPDATE, DELETE allowed"
    if verify_exists and (sql_upper.startswith("UPDATE") or sql_upper.startswith("DELETE")):
        table, where = _parse_table_and_where(raw_sql)
        if table and where and not _write_tool_verify_exists(engine, table, where):
            return False, "No matching row found; aborting write for safety."
    try:
        with engine.begin() as conn:
            conn.execute(text(raw_sql))
        return True, None
    except Exception as e:
        return False, str(e)


def _get_sql_agent():
    global _sql_agent
    if _sql_agent is None:
        if not _DEPS_AVAILABLE:
            raise ValueError("langchain_community not available")
        db = _get_db()
        llm = _get_llm()
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        full_prefix = SQL_PREFIX + POSTGRES_QUOTE_RULE
        _sql_agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="tool-calling",
            handle_parsing_errors=True,
            prefix=full_prefix,
        )
    return _sql_agent


def _agent_output_to_str(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if hasattr(raw, "content"):
        return _agent_output_to_str(raw.content)
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "content"):
                parts.append(_agent_output_to_str(item.content))
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else ""
    return str(raw)


def _parse_table_json(text_out: str) -> Tuple[str, Optional[List]]:
    table_data = None
    table_match = re.search(r"TABLE_JSON:\s*(\[.*\])", text_out, re.DOTALL)
    if table_match:
        try:
            table_data = json.loads(table_match.group(1))
        except Exception:
            pass
    return text_out.split("TABLE_JSON")[0].strip(), table_data


# Stock/reorder intent triggers (NLP-style keyword match)
_STOCK_REORDER_TRIGGERS = (
    "stock", "reorder", "add stock", "need to add", "running low", "inventory",
    "low stock", "need to reorder", "do we need to add stock", "stock level",
    "which need reorder", "material stock", "reorder level",
)


def _is_stock_reorder_query(query: str) -> bool:
    q = _norm(query)
    return any(t in q for t in _STOCK_REORDER_TRIGGERS)


def _try_quick_stock_reorder(query: str) -> Optional[Dict[str, Any]]:
    """Answer 'do we need to add stock?' / 'show reorder' with low-stock list (0 LLM)."""
    if not _is_stock_reorder_query(query):
        return None
    try:
        engine = _get_engine()
    except Exception:
        return None
    sql = text("""
        SELECT "materialCode" AS "materialCode",
               SUM("remainingQty") AS "totalQty"
        FROM "StockBatch"
        WHERE "isActive" = TRUE
        GROUP BY "materialCode"
        ORDER BY "totalQty" ASC
        LIMIT 20
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
    except Exception:
        return None
    if not rows:
        return {
            "type": "AI_RESPONSE",
            "content": "No active stock records found. Open Materials to manage stock and reorder.",
            "tableData": [],
            "thought_process": "Quick answer (0 tokens): StockBatch query; no rows.",
            "sql_query": None,
            "action_taken": "DB_READ",
            "next_steps": _suggest_next_steps("DB_READ", query, "/materials/MaterialMasterManager"),
            "source": "Agentic ERP",
            "navigateTo": "/materials/MaterialMasterManager",
        }
    keys = ["materialCode", "totalQty"]
    table_data = [dict(zip(keys, r)) for r in rows]
    low = [r for r in rows if float(r[1] or 0) < 50]
    if low:
        content = f"Found {len(rows)} material(s). {len(low)} have low stock (below 50); consider reordering. Open Materials to create purchase orders."
    else:
        content = f"Found {len(rows)} material(s) with current stock. Lowest first. Open Materials for reorder and purchase orders."
    return {
        "type": "AI_RESPONSE",
        "content": content,
        "tableData": table_data,
        "thought_process": "Quick answer (0 tokens): low-stock list from StockBatch.",
        "sql_query": None,
        "action_taken": "DB_READ",
        "next_steps": _suggest_next_steps("DB_READ", query, "/materials/MaterialMasterManager"),
        "source": "Agentic ERP",
        "navigateTo": "/materials/MaterialMasterManager",
    }


def _try_quick_room_bill(query: str) -> Optional[Dict[str, Any]]:
    """Answer 'bill amount of room 205' / 'room 205 bill' with GuestFolio+FolioEntry (0 LLM calls)."""
    q = _norm(query)
    if "bill" not in q or "room" not in q:
        return None
    match = re.search(r"room\s+(\w+)", q, re.IGNORECASE)
    if not match:
        return None
    room_num = match.group(1).strip()
    try:
        engine = _get_engine()
    except Exception:
        return None
    # GuestFolio.roomNumber, FolioEntry.type='CHARGE'|'PAYMENT', amount. Balance = charges - payments.
    sql = text("""
        SELECT gf."id", gf."guestName", gf."roomNumber",
               SUM(CASE WHEN fe."type" = 'CHARGE' THEN fe."amount" ELSE 0 END) AS charges,
               SUM(CASE WHEN fe."type" = 'PAYMENT' THEN fe."amount" ELSE 0 END) AS payments,
               SUM(CASE WHEN fe."type" = 'CHARGE' THEN fe."amount" ELSE 0 END)
                 - SUM(CASE WHEN fe."type" = 'PAYMENT' THEN fe."amount" ELSE 0 END) AS balance
        FROM "GuestFolio" gf
        LEFT JOIN "FolioEntry" fe ON fe."folioId" = gf."id"
        WHERE gf."roomNumber" = :rn
        GROUP BY gf."id", gf."guestName", gf."roomNumber"
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"rn": room_num}).fetchall()
    except Exception:
        return None
    if not rows:
        return {
            "type": "AI_RESPONSE",
            "content": f"No folio found for room {room_num}. Check room number or open Guest Billing.",
            "tableData": [],
            "thought_process": "Quick answer (0 tokens): direct SQL for room folio.",
            "sql_query": None,
            "action_taken": "DB_READ",
            "next_steps": _suggest_next_steps("DB_READ", query, "/financials/guest-billing"),
            "source": "Agentic ERP",
            "navigateTo": "/financials/guest-billing",
        }
    keys = ["id", "guestName", "roomNumber", "charges", "payments", "balance"]
    table_data = [dict(zip(keys, r)) for r in rows]
    row0 = rows[0]
    balance = float(row0[5] or 0)
    content = f"Room {room_num} bill balance: {balance}. Guest: {row0[1] or 'N/A'}."
    if len(rows) > 1:
        total_balance = sum(float(r[5] or 0) for r in rows)
        content = f"Room {room_num}: {len(rows)} folio(s), total balance: {total_balance}."
    return {
        "type": "AI_RESPONSE",
        "content": content,
        "tableData": table_data,
        "thought_process": "Quick answer (0 tokens): direct SQL GuestFolio+FolioEntry by room.",
        "sql_query": None,
        "action_taken": "DB_READ",
        "next_steps": _suggest_next_steps("DB_READ", query, "/financials/guest-billing"),
        "source": "Agentic ERP",
        "navigateTo": "/financials/guest-billing",
    }


def _try_quick_table_bill(query: str) -> Optional[Dict[str, Any]]:
    """Answer 'bill amount of table 01' / 'bill for table T1' with direct SQL (0 LLM calls). Avoids 429."""
    q = _norm(query)
    if "bill" not in q or "table" not in q:
        return None
    match = re.search(r"table\s+(\w+)", q, re.IGNORECASE)
    if not match:
        return None
    table_num = match.group(1).strip()
    try:
        engine = _get_engine()
    except Exception:
        return None
    sql = text(
        'SELECT t."tableNumber", o."totalAmount", o."orderNumber", o."paymentStatus" '
        'FROM "RestaurantOrder" o JOIN "RestaurantTable" t ON o."tableId" = t.id '
        'WHERE t."tableNumber" = :tn ORDER BY o."updatedAt" DESC LIMIT 5'
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"tn": table_num}).fetchall()
    except Exception:
        return None
    if not rows:
        return {
            "type": "AI_RESPONSE",
            "content": f"No orders found for table {table_num}. Check the table number or open Restaurant.",
            "tableData": [],
            "thought_process": "Quick answer (0 tokens): direct SQL for table bill.",
            "sql_query": None,
            "action_taken": "DB_READ",
            "next_steps": _suggest_next_steps("DB_READ", query, "/restaurant/restaurants"),
            "source": "Agentic ERP",
            "navigateTo": "/restaurant/restaurants",
        }
    keys = ["tableNumber", "totalAmount", "orderNumber", "paymentStatus"]
    table_data = [dict(zip(keys, r)) for r in rows]
    total = sum(float(r[1] or 0) for r in rows)
    if len(rows) == 1:
        content = f"Table {table_num} bill amount: {rows[0][1]}."
    else:
        content = f"Table {table_num}: {len(rows)} order(s), total amount: {total}."
    return {
        "type": "AI_RESPONSE",
        "content": content,
        "tableData": table_data,
        "thought_process": "Quick answer (0 tokens): direct SQL for RestaurantOrder by table.",
        "sql_query": None,
        "action_taken": "DB_READ",
        "next_steps": _suggest_next_steps("DB_READ", query, "/restaurant/restaurants"),
        "source": "Agentic ERP",
        "navigateTo": "/restaurant/restaurants",
    }


# ---------------------------------------------------------------------------
# Pure RAG: Vector search for rooms (semantic – "luxury stay" matches by meaning)
# ---------------------------------------------------------------------------
ROOM_SEARCH_TRIGGERS = (
    "luxury", "stay", "room", "suite", "budget", "view", "sea view", "deluxe", "premium",
    "ac", "air conditioned", "family", "couple", "honeymoon", "ocean", "pool", "spa",
    "comfortable", "cheap", "expensive", "best room", "recommend", "available room",
)
VECTOR_INDEX_TOP_K = 10
VECTOR_EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, good for semantic similarity


def _get_embedding_model():
    global _embedding_model
    if not _VECTOR_DEPS_AVAILABLE or SentenceTransformer is None:
        raise ValueError("sentence_transformers not available")
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(VECTOR_EMBED_MODEL)
    return _embedding_model


def _build_room_vector_index(engine: Engine) -> Tuple[Any, List[Dict[str, Any]]]:
    """Build FAISS index from Room + RoomType. Searchable text = room_type + price + room_number.
    Semantic search: 'luxury stay' matches Deluxe/Premium by embedding similarity (no 'luxury' column in DB).
    Optional: add rt.description to SELECT and to text_str for richer semantic match when column exists."""
    sql = text("""
        SELECT r.id, r."roomNumber", rt.name AS room_type, rt."basePrice"
        FROM "Room" r
        JOIN "RoomType" rt ON rt.id = r."roomTypeId"
        ORDER BY r."roomNumber"
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        return None, []
    metadata = []
    texts = []
    for r in rows:
        rid, rnum, rtype, price = r[0], r[1], r[2], r[3]
        text_str = f"{rtype} room {rnum} base price {price}. Stay accommodation."
        texts.append(text_str)
        metadata.append({
            "id": str(rid),
            "roomNumber": str(rnum) if rnum is not None else "",
            "room_type": str(rtype) if rtype is not None else "",
            "basePrice": float(price) if price is not None else 0.0,
        })
    model = _get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False)
    vectors = vectors.astype("float32")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, metadata


def _get_room_vector_index() -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """Lazy-build and cache room FAISS index + metadata."""
    global _room_index_faiss, _room_metadata
    if _room_index_faiss is not None:
        return _room_index_faiss, _room_metadata
    if not _VECTOR_DEPS_AVAILABLE:
        return None, []
    try:
        engine = _get_engine()
        _room_index_faiss, _room_metadata = _build_room_vector_index(engine)
        return _room_index_faiss, _room_metadata
    except Exception:
        return None, []


def _is_room_search_query(query: str) -> bool:
    q = _norm(query)
    return any(t in q for t in ROOM_SEARCH_TRIGGERS)


def _try_vector_search_rooms(query: str) -> Optional[Dict[str, Any]]:
    """Pure RAG: embed query, semantic search over room descriptions (no 'luxury' column in DB needed)."""
    if not _is_room_search_query(query):
        return None
    index, metadata = _get_room_vector_index()
    if index is None or not metadata:
        return None
    try:
        model = _get_embedding_model()
        q_vec = model.encode([query], show_progress_bar=False).astype("float32")
        k = min(VECTOR_INDEX_TOP_K, len(metadata))
        distances, indices = index.search(q_vec, k)
    except Exception:
        return None
    if len(indices) == 0 or len(indices[0]) == 0:
        return {
            "type": "AI_RESPONSE",
            "content": "No rooms found. Open Hotel Rooms to manage room types and availability.",
            "tableData": [],
            "thought_process": "Vector search (Pure RAG): no matches.",
            "sql_query": None,
            "action_taken": "DB_READ",
            "next_steps": _suggest_next_steps("DB_READ", query, "/hotel/rooms"),
            "source": "Agentic ERP",
            "navigateTo": "/hotel/rooms",
        }
    table_data = [metadata[i] for i in indices[0]]
    content = f"Found {len(table_data)} room(s) matching your search (semantic match). Best matches first."
    return {
        "type": "AI_RESPONSE",
        "content": content,
        "tableData": table_data,
        "thought_process": "Pure RAG: vector search over room types (semantic, no keyword column needed).",
        "sql_query": None,
        "action_taken": "DB_READ",
        "next_steps": _suggest_next_steps("DB_READ", query, "/hotel/rooms"),
        "source": "Agentic ERP",
        "navigateTo": "/hotel/rooms",
    }


def _suggest_next_steps(action_taken: str, query: str, route: Optional[str] = None) -> List[str]:
    suggestions = []
    q = _norm(query)
    if action_taken == "NAVIGATE" and route:
        if "payroll" in route or "payroll" in q:
            suggestions.extend(["View payroll report", "Export payroll", "Go to employees"])
        elif "ledger" in route or "ledger" in q:
            suggestions.extend(["View trial balance", "Open journal", "Chart of accounts"])
        elif "employee" in route or "employee" in q:
            suggestions.extend(["Open attendance", "Open payroll", "View organization"])
        elif "material" in route or "stock" in q or "reorder" in q:
            suggestions.extend(["Create purchase order", "View stock levels", "Reorder analysis"])
        else:
            suggestions.extend(["Search again", "View reports", "Back to dashboard"])
    elif action_taken == "DB_READ":
        suggestions.extend(["Drill down", "Export this data", "Show related records"])
    elif action_taken == "DB_UPDATE":
        suggestions.extend(["Confirm change", "View updated record", "Undo"])
    while len(suggestions) < 3:
        suggestions.append("Ask another question")
    return suggestions[:3]


async def run_agentic_flow(session_id: str, query: str) -> Dict[str, Any]:
    session_id = session_id or "default"
    nav = semantic_router(query)
    if nav is not None:
        nav["next_steps"] = _suggest_next_steps("NAVIGATE", query, nav.get("route"))
        return nav

    # Multi-agent: quick SQL agents (0 LLM) — answer by question type (NLP + RAG/LLM)
    QUICK_AGENTS = [_try_quick_stock_reorder, _try_quick_room_bill, _try_quick_table_bill]
    for agent_fn in QUICK_AGENTS:
        try:
            quick = agent_fn(query)
            if quick is not None:
                return quick
        except Exception:
            pass

    # Pure RAG: vector search (semantic) – "luxury stay" matches rooms by meaning, no "luxury" column needed
    try:
        vector_result = _try_vector_search_rooms(query)
        if vector_result is not None:
            return vector_result
    except Exception:
        pass

    if not _DEPS_AVAILABLE:
        return {
            "type": "ERROR",
            "content": "Agentic ERP requires langchain_google_genai and langchain_community.",
            "thought_process": None,
            "sql_query": None,
            "action_taken": None,
            "next_steps": [],
        }

    try:
        _get_engine()
        _get_llm()
    except ValueError as e:
        return {"type": "ERROR", "content": str(e), "thought_process": None, "sql_query": None, "action_taken": None, "next_steps": []}

    context = build_context_for_agent(session_id)
    try:
        intent = await asyncio.to_thread(_planner_classify, query, context)
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
            route = _get_route_exact(query)
            if not route and _fuzzy_route(query):
                route = _fuzzy_route(query)[1]
            return {
                "type": "NAVIGATE",
                "route": route,
                "navigateTo": route,
                "content": "Taking you to the most relevant page. You can try your question again in a minute (API limit).",
                "thought_process": "Planner hit API quota; returning suggested page (0 further tokens).",
                "sql_query": None,
                "action_taken": "NAVIGATE",
                "next_steps": _suggest_next_steps("NAVIGATE", query, route),
                "tableData": None,
            }
        raise

    add_to_memory(session_id, "user", query)

    if intent == "READ":
        try:
            result = await asyncio.to_thread(read_tool_with_retry, query, session_id)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                route = _get_route_exact(query)
                if not route and _fuzzy_route(query):
                    route = _fuzzy_route(query)[1]
                return {
                    "type": "NAVIGATE",
                    "route": route,
                    "navigateTo": route,
                    "content": "Taking you to the most relevant page. Try your question again in a minute (API limit).",
                    "thought_process": None,
                    "sql_query": None,
                    "action_taken": "NAVIGATE",
                    "next_steps": _suggest_next_steps("NAVIGATE", query, route),
                    "tableData": None,
                }
            result = {"success": False, "error": str(e), "thought_process": str(e), "action_taken": "DB_READ"}

        if result.get("success"):
            content = f"Found {len(result.get('rows') or [])} row(s)." if result.get("rows") else "No rows returned."
            add_to_memory(session_id, "assistant", content)
            route = _get_route_exact(query)
            return {
                "type": "AI_RESPONSE",
                "content": content,
                "tableData": result.get("rows"),
                "thought_process": result.get("thought_process"),
                "sql_query": result.get("sql_query"),
                "action_taken": "DB_READ",
                "next_steps": _suggest_next_steps("DB_READ", query, route),
                "source": "Agentic ERP",
                "navigateTo": route,
            }
        try:
            agent = _get_sql_agent()
            response = await _invoke_with_429_retry(agent.invoke, {"input": query})
            raw_output = response.get("output", "")
            text_out = _agent_output_to_str(raw_output)
            clean_text, table_json = _parse_table_json(text_out)
            route = _get_route_exact(query)
            return {
                "type": "AI_RESPONSE",
                "content": clean_text or "Query executed.",
                "tableData": table_json,
                "thought_process": "SQL agent fallback after ReadTool retries.",
                "sql_query": None,
                "action_taken": "DB_READ",
                "next_steps": _suggest_next_steps("DB_READ", query, route),
                "source": "Agentic ERP",
                "navigateTo": route,
            }
        except Exception as e2:
            err_str = str(e2).lower()
            if "429" in err_str or "quota" in err_str:
                return {
                    "type": "ERROR",
                    "content": "API quota reached. Please try again in a minute.",
                    "thought_process": result.get("thought_process"),
                    "sql_query": result.get("sql_query"),
                    "action_taken": "DB_READ",
                    "next_steps": _suggest_next_steps("DB_READ", query),
                    "navigateTo": _get_route_exact(query),
                }
            return {
                "type": "AI_RESPONSE",
                "content": result.get("error", str(e2)),
                "tableData": None,
                "thought_process": result.get("thought_process"),
                "sql_query": result.get("sql_query"),
                "action_taken": "DB_READ",
                "next_steps": _suggest_next_steps("DB_READ", query),
                "source": "Agentic ERP",
            }

    # WRITE
    def _execute_write_sync(q: str, ctx: str, sid: str) -> Dict[str, Any]:
        llm = _get_llm()
        schema_hint = _get_table_schema_hint(_get_engine())
        prompt = (
            f"{PERSONA}\n{POSTGRES_QUOTE_RULE}\n\nSchema:\n{schema_hint}\n\n"
            f"Context:\n{ctx}\n\nUser request: {q}\n\n"
            "Generate a single PostgreSQL INSERT, UPDATE, or DELETE statement. Use double-quoted identifiers. No explanation."
        )
        out = llm.invoke(prompt)
        sql = out.content.strip() if hasattr(out, "content") else str(out).strip()
        for sep in ("```sql", "```"):
            if sep in sql:
                sql = sql.split(sep)[1] if len(sql.split(sep)) > 1 else sql
        sql = sql.strip()
        ok, err = write_tool_execute(sql, _get_engine())
        route = _get_route_exact(q)
        if not route and _fuzzy_route(q):
            route = _fuzzy_route(q)[1]
        if ok:
            add_to_memory(sid, "assistant", "Write executed.")
            return {
                "type": "AI_RESPONSE",
                "content": "Update applied.",
                "tableData": None,
                "thought_process": "ReAct: generated write SQL; executed after validation.",
                "sql_query": sql,
                "action_taken": "DB_UPDATE",
                "next_steps": _suggest_next_steps("DB_UPDATE", q),
                "source": "Agentic ERP",
                "navigateTo": route,
            }
        # Friendly response when write not allowed (e.g. "pay for employee" -> go to Payroll)
        if err and "Only INSERT, UPDATE, DELETE allowed" in err:
            add_to_memory(sid, "assistant", "Taking you to the right screen to complete the action.")
            nav_route = route or "/hr/payroll"
            return {
                "type": "NAVIGATE",
                "route": nav_route,
                "navigateTo": nav_route,
                "content": "Payments and similar actions are done on the app screen. Taking you to Payroll.",
                "tableData": None,
                "thought_process": "Request is better handled on the Payroll screen; redirecting (no DB write from chat).",
                "sql_query": None,
                "action_taken": "NAVIGATE",
                "next_steps": _suggest_next_steps("NAVIGATE", q, nav_route),
                "source": "Agentic ERP",
            }
        add_to_memory(sid, "assistant", f"Write failed: {err}")
        return {
            "type": "AI_RESPONSE",
            "content": f"Update failed: {err}",
            "tableData": None,
            "thought_process": "ReAct: generated write SQL; executed after validation.",
            "sql_query": None,
            "action_taken": "DB_UPDATE",
            "next_steps": _suggest_next_steps("DB_UPDATE", q),
            "source": "Agentic ERP",
            "navigateTo": route,
        }
    try:
        return await asyncio.to_thread(_execute_write_sync, query, context, session_id)
    except Exception as e:
        if "429" in str(e).lower() or "quota" in str(e).lower():
            route = _get_route_exact(query)
            if not route and _fuzzy_route(query):
                route = _fuzzy_route(query)[1]
            return {
                "type": "NAVIGATE",
                "route": route,
                "navigateTo": route,
                "content": "Taking you to the most relevant page. Try again in a minute (API limit).",
                "thought_process": None,
                "sql_query": None,
                "action_taken": "NAVIGATE",
                "next_steps": _suggest_next_steps("NAVIGATE", query, route),
                "tableData": None,
            }
        raise


# --------------- Launcher (only when run as script) ---------------
if __name__ == "__main__":
    os.chdir(_this_dir)
    try:
        from main import app
    except ImportError as e:
        print("Error: Could not import main. Run from AI/AI directory or run: python main.py")
        print(f"Details: {e}")
        sys.exit(1)
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
