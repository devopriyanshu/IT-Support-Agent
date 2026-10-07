import logging
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None
memory = MemorySaver()


def extract_text(content) -> str:
    """Safely extracts clean string text from LLM response content (handles list/dict blocks from Gemini)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(getattr(part, "text", ""))
        return "".join(parts).strip()
    return str(content).strip()


def llm():
    global _llm
    if _llm is None:
        api_key = settings.google_api_key or settings.gemini_api_key
        if api_key:
            _llm = ChatGoogleGenerativeAI(
                model=settings.google_model,
                temperature=0,
                google_api_key=api_key,
            )
        elif settings.openai_api_key:
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0,
                api_key=settings.openai_api_key,
            )
        else:
            raise RuntimeError("GOOGLE_API_KEY is missing")
    return _llm


def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


def contextualize_question(state: AgentState):
    history_messages = state.get("messages", [])[:-1]
    if not history_messages:
        return {"current_query": state["question"]}

    recent = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in history_messages[-4:])
    prompt = f"""Given the conversation history and the latest user question, rephrase the question to be a standalone IT support search query with all necessary subject context (resolving pronouns like 'it', 'that', 'the client', 'step 1').
If the message is a greeting, acknowledgment, thanks, or already completely clear and standalone, return it as-is without changing it.
Return ONLY the query text without quotes or explanation.

Conversation History:
{recent}

Latest Question: {state['question']}
Query:"""
    try:
        resp = llm().invoke(prompt)
        rephrased = extract_text(resp.content).strip().strip('"\'')
        if rephrased.lower().startswith("query:"):
            rephrased = rephrased[len("query:"):].strip()
        changed = rephrased and rephrased.lower() != state["question"].lower()
        trace_msg = f"Contextualized query → {rephrased}" if changed else "Contextualized query → Standalone"
        return {
            "current_query": rephrased or state["question"],
            "trace": add_trace(state, trace_msg),
        }
    except Exception as e:
        logger.warning(f"Contextualization failed: {e}")
        return {"current_query": state["question"]}


def route_question(state: AgentState):
    route_val = "kb"
    query = state.get("current_query") or state["question"]
    try:
        router = llm().with_structured_output(RouteDecision)
        decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {query}
""")
        if isinstance(decision, RouteDecision):
            route_val = decision.route
        elif isinstance(decision, dict):
            route_val = decision.get("route", "kb")
        else:
            route_val = getattr(decision, "route", "kb")
    except Exception as e:
        logger.warning(f"Structured route parsing failed: {e}. Falling back to prompt extraction.")
        res = extract_text(llm().invoke(f"Classify if this question is 'kb' (IT support/policies) or 'direct' (greeting/chat). Question: {query}. Return only 'kb' or 'direct'.").content)
        route_val = "direct" if "direct" in res.lower() else "kb"

    route_val = str(route_val).strip().lower()
    if route_val not in ("kb", "direct"):
        route_val = "direct" if "direct" in route_val else "kb"

    return {"source_used": route_val, "trace": add_trace(state, f"Router → {route_val.upper()}")}


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


def retrieve_kb(state: AgentState):
    query = state.get("current_query") or state["question"]
    docs = get_retriever().invoke(query)
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}


def grade_kb(state: AgentState):
    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in state["kb_docs"])
    query = state.get("current_query") or state["question"]
    grade_val = "weak"
    try:
        grader = llm().with_structured_output(EvidenceGrade)
        grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.
Question: {query}
Private company KB evidence:\n{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak.
""")
        if isinstance(grade, EvidenceGrade):
            grade_val = grade.grade
        elif isinstance(grade, dict):
            grade_val = grade.get("grade", "weak")
        else:
            grade_val = getattr(grade, "grade", "weak")
    except Exception as e:
        logger.warning(f"Structured grade_kb failed: {e}. Falling back to prompt extraction.")
        res = extract_text(llm().invoke(f"Does this evidence answer '{query}'? Evidence:\n{context}\nAnswer ONLY 'good' or 'weak'.").content)
        grade_val = "good" if "good" in res.lower() else "weak"

    grade_val = str(grade_val).strip().lower()
    grade_val = "good" if "good" in grade_val else "weak"
    return {"kb_grade": grade_val, "trace": add_trace(state, f"KB evidence grade → {grade_val.upper()}")}


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


def search_web(state: AgentState):
    query = state.get("current_query") or state["question"]
    result = web_search_tool().invoke({"query": query})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }


def grade_web(state: AgentState):
    query = state.get("current_query") or state["question"]
    grade_val = "weak"
    try:
        grader = llm().with_structured_output(EvidenceGrade)
        grade = grader.invoke(f"""
Question: {query}
Web evidence:\n{state['web_results']}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
""")
        if isinstance(grade, EvidenceGrade):
            grade_val = grade.grade
        elif isinstance(grade, dict):
            grade_val = grade.get("grade", "weak")
        else:
            grade_val = getattr(grade, "grade", "weak")
    except Exception as e:
        logger.warning(f"Structured grade_web failed: {e}. Falling back to prompt extraction.")
        res = extract_text(llm().invoke(f"Does this web evidence answer '{query}'? Evidence:\n{state['web_results']}\nAnswer ONLY 'good' or 'weak'.").content)
        grade_val = "good" if "good" in res.lower() else "weak"

    grade_val = str(grade_val).strip().lower()
    grade_val = "good" if "good" in grade_val else "weak"
    return {"web_grade": grade_val, "trace": add_trace(state, f"Web evidence grade → {grade_val.upper()}")}


def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"


def rewrite_query(state: AgentState):
    query = state.get("current_query") or state["question"]
    resp = llm().invoke(f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {query}
""")
    rewritten = extract_text(resp.content).strip().strip('"\'')
    if rewritten.lower().startswith("rewritten query:"):
        rewritten = rewritten[len("rewritten query:"):].strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }


def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    history_messages = state.get("messages", [])[:-1]
    history_block = ""
    if history_messages:
        recent = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in history_messages[-4:])
        history_block = f"\nRecent Conversation History:\n{recent}\n"

    resp = llm().invoke(f"""
You are an enterprise IT support copilot. Answer ONLY from the private company KB below, taking into account recent conversation history if relevant.
Be concise, actionable, and safe. If steps are present, present them clearly.
Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
{history_block}
Question: {state['question']}

Private KB:
{context}
""")
    answer = extract_text(resp.content)
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title": src.split("/")[-1], "url": "", "type": "private_kb"})
    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "messages": [AIMessage(content=answer)],
        "trace": add_trace(state, "Answer generation → PRIVATE KB"),
    }


def generate_from_web(state: AgentState):
    history_messages = state.get("messages", [])[:-1]
    history_block = ""
    if history_messages:
        recent = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in history_messages[-4:])
        history_block = f"\nRecent Conversation History:\n{recent}\n"

    resp = llm().invoke(f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below, taking into account recent conversation history if relevant.
Clearly say this is external web information and may need IT validation before changing company-managed systems.
{history_block}
Question: {state['question']}

Web evidence:
{state['web_results']}
""")
    answer = extract_text(resp.content)
    return {
        "answer": answer,
        "source_used": "web_search",
        "messages": [AIMessage(content=answer)],
        "trace": add_trace(state, "Answer generation → WEB SEARCH"),
    }


def direct_answer(state: AgentState):
    history_messages = state.get("messages", [])[:-1]
    history_block = ""
    if history_messages:
        recent = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in history_messages[-4:])
        history_block = f"Recent Conversation History:\n{recent}\n\n"

    resp = llm().invoke(f"{history_block}Respond briefly, naturally, and helpfully to: {state['question']}")
    answer = extract_text(resp.content)
    return {
        "answer": answer,
        "source_used": "direct",
        "messages": [AIMessage(content=answer)],
        "trace": add_trace(state, "Direct response → no retrieval"),
    }


def insufficient(state: AgentState):
    msg = "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details."
    return {
        "answer": msg,
        "source_used": "insufficient_evidence",
        "messages": [AIMessage(content=msg)],
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }


def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "contextualize_question": contextualize_question,
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "contextualize_question")
    graph.add_edge("contextualize_question", "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile(checkpointer=memory)


agent_graph = build_graph()


def ask(question: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    initial: AgentState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial, config=config)
