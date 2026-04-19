"""
cars_agent_streamlit.py — Cars Production Agent UI
Run: streamlit run cars_agent_streamlit.py
"""
import streamlit as st
import uuid
import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="🚗 Cars Expert Agent", page_icon="🚗", layout="centered")
st.title("🚗 Cars Expert Agent")
st.caption("Ask me anything about car specs, pricing, safety ratings, EVs, and buying tips!")

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Toyota Camry — Specs & Pricing",
        "text": """The Toyota Camry is one of the best-selling mid-size sedans in the world.
Available trims: LE, SE, XSE, XLE, TRD, and Hybrid. The base LE starts at approximately $27,000 USD.
Engine options: 2.5L 4-cylinder (203 hp) or 3.5L V6 (301 hp). The Hybrid variant uses a 2.5L engine
paired with an electric motor for a combined 208 hp and achieves up to 51 MPG city / 53 MPG highway.
Fuel economy for the standard 4-cylinder: 28 MPG city / 39 MPG highway.
The Camry seats 5 passengers, has a 15.1 cubic foot trunk, and comes standard with Toyota Safety Sense 2.5+.
NHTSA overall safety rating: 5 stars. IIHS Top Safety Pick+.
Warranty: 3 years/36,000 miles basic; 5 years/60,000 miles powertrain.
The Camry is known for exceptional reliability and low cost of ownership."""
    },
    {
        "id": "doc_002",
        "topic": "Tesla Model 3 — Electric Sedan Specs",
        "text": """The Tesla Model 3 is a premium all-electric sedan with three variants:
Standard Range RWD (~$40,240), Long Range AWD (~$47,240), and Performance AWD (~$53,240) USD prices before incentives.
Range: Standard — 272 miles; Long Range — 358 miles; Performance — 315 miles (EPA estimated).
0-60 mph: Standard — 5.8 sec; Long Range — 4.2 sec; Performance — 3.1 sec.
Top speed: up to 162 mph (Performance). Charging: supports Tesla Supercharger V3 (up to 250 kW),
adding ~170 miles in 15 minutes. Home charging on a 240V outlet adds ~37 miles per hour.
Interior: minimalist 15.4-inch touchscreen. Seats 5. Trunk: 15 cu ft + 2.7 cu ft frunk.
Autopilot standard. NHTSA: 5-star overall. IIHS Top Safety Pick+.
Federal tax credit: up to $7,500 (eligibility depends on income and trim)."""
    },
    {
        "id": "doc_003",
        "topic": "Honda CR-V — Compact SUV Specs",
        "text": """The Honda CR-V is a top-selling compact SUV praised for practicality and reliability.
Trims: LX, EX, EX-L, Sport, Sport-L, Sport Touring. Starting price: approximately $31,895 USD (LX FWD).
Engine: 1.5L turbocharged 4-cylinder (190 hp, 179 lb-ft torque) mated to a CVT transmission.
Hybrid option: 2.0L Atkinson-cycle + electric motor (204 hp combined); up to 40 MPG city / 34 MPG highway.
Non-hybrid fuel economy: 28 MPG city / 34 MPG highway (FWD). AWD available on all trims.
Cargo space: 39.3 cu ft behind rear seats; 76.5 cu ft with rear seats folded — one of the largest in its class.
Towing capacity: up to 1,500 lbs (non-hybrid). Seats 5 adults comfortably.
Honda Sensing safety suite standard on all trims. NHTSA: 5-star overall. IIHS Top Safety Pick+."""
    },
    {
        "id": "doc_004",
        "topic": "BMW 3 Series — Luxury Sports Sedan",
        "text": """The BMW 3 Series is the benchmark luxury compact sports sedan. Starting at around $45,000 USD (330i).
Engine options: 2.0L turbocharged inline-4 (255 hp) in 330i; 3.0L inline-6 (382 hp) in M340i;
plug-in hybrid (288 hp combined) in 330e. All-wheel drive (xDrive) available on all variants.
Performance: 330i — 0-60 in 5.6 sec; M340i — 0-60 in 4.2 sec.
Fuel economy: 330i — 26 MPG city / 36 MPG highway. M340i — 25 MPG city / 33 MPG highway.
Infotainment: BMW iDrive 8 with 14.9-inch curved touchscreen, wireless Apple CarPlay/Android Auto.
Seats 5 (tight in rear). Trunk: 17 cubic feet. Warranty: 4 years/50,000 miles (bumper-to-bumper).
Best suited for driving enthusiasts who want a premium feel and sporty dynamics."""
    },
    {
        "id": "doc_005",
        "topic": "Ford F-150 — Full-Size Pickup Truck",
        "text": """The Ford F-150 is America's best-selling vehicle for 46+ consecutive years.
Starting price: approximately $36,080 USD (Regular Cab XL).
Engine choices: 2.7L EcoBoost V6 (325 hp), 3.5L EcoBoost V6 (400 hp), 5.0L V8 (400 hp),
3.5L PowerBoost Hybrid (430 hp, 12,700 lb towing).
Towing capacity: up to 14,000 lbs (3.5L EcoBoost with Max Trailer Tow Package).
Payload: up to 2,455 lbs. Bed lengths: 5.5, 6.5, or 8 feet.
Fuel economy (PowerBoost Hybrid): 25 MPG city / 26 MPG highway.
The F-150 Lightning is the electric variant — up to 320 miles range, 775 lb-ft torque.
NHTSA: 5-star (SuperCrew 4WD). Best for buyers needing serious towing and hauling capability."""
    },
    {
        "id": "doc_006",
        "topic": "Hyundai Ioniq 6 — Long-Range Electric Sedan",
        "text": """The Hyundai Ioniq 6 is a sleek electric sedan built on the E-GMP platform.
Starting price: ~$38,615 USD (SE Standard Range RWD). Long Range RWD tops out at ~$45,900 USD.
Battery options: 53 kWh (Standard Range, 151 miles) and 77.4 kWh (Long Range, up to 361 miles RWD).
Charging: 800V architecture supports 350 kW DC fast charging — 10% to 80% in about 18 minutes.
Performance: Long Range AWD — 320 hp, 0-60 in 5.1 sec.
Aerodynamics: drag coefficient of 0.21 Cd — one of the lowest in production.
NHTSA: 5-star overall. IIHS Top Safety Pick+.
Eligible for $7,500 federal EV tax credit. Warranty: 10 years/100,000 battery."""
    },
    {
        "id": "doc_007",
        "topic": "Toyota RAV4 — Best-Selling SUV Specs",
        "text": """The Toyota RAV4 is the world's best-selling SUV. Starting at ~$28,975 USD (LE FWD).
Engine: 2.5L 4-cylinder (203 hp). Fuel economy: 27 MPG city / 35 MPG highway (AWD).
RAV4 Hybrid (starts ~$32,150): 219 hp combined, 38 MPG city / 38 MPG highway.
RAV4 Prime PHEV: 302 hp, 42 miles electric-only range, then operates as a full hybrid.
Cargo space: 37.6 cu ft behind rear seats; 69.8 cu ft max. Towing: up to 3,500 lbs.
Toyota Safety Sense 2.0 standard on all trims. NHTSA: 5-star. IIHS Top Safety Pick+.
Excellent resale value and reliability history."""
    },
    {
        "id": "doc_008",
        "topic": "Porsche 911 — Sports Car Performance",
        "text": """The Porsche 911 is the definitive sports car, in production since 1963. Starting at ~$115,000 USD (Carrera).
Engine: Rear-mounted 3.0L twin-turbo flat-6 (379 hp in Carrera; 473 hp in Carrera S).
GT3 uses naturally aspirated 4.0L (502 hp).
Turbo S: 640 hp, 0-60 in 2.6 seconds, top speed 205 mph.
Transmission: 8-speed PDK dual-clutch (manual available on select models).
Fuel economy (Carrera): 18 MPG city / 25 MPG highway.
Seats 4 (2+2 layout). Resale value is exceptional — 911s depreciate very slowly.
Best for driving purists who want a daily driver that is also track-capable."""
    },
    {
        "id": "doc_009",
        "topic": "Car Safety Ratings — How to Read Them",
        "text": """Two major organizations rate car safety in the US: NHTSA and IIHS.
NHTSA uses a 5-star system across frontal crash, side crash, rollover. 5 stars = best.
IIHS uses Good / Acceptable / Marginal / Poor ratings.
IIHS Top Safety Pick (TSP) = passes all crashworthiness tests + acceptable/good on at least one headlight.
IIHS Top Safety Pick+ (TSP+) = TSP + good headlights on all trims. TSP+ is the gold standard.
Euro NCAP operates similarly in Europe using a 0-5 star rating.
Active safety systems to look for: Automatic Emergency Braking (AEB), Blind Spot Monitoring,
Lane Keeping Assist (LKA), Rear Cross Traffic Alert (RCTA), and adaptive cruise control."""
    },
    {
        "id": "doc_010",
        "topic": "EV Buying Guide — Key Factors",
        "text": """When buying an electric vehicle (EV), consider these key factors:
1. Range: EPA-rated range varies by trim and conditions. Cold weather reduces range by 20-40%.
   Aim for 50-100 miles more than your typical daily need.
2. Charging infrastructure: Check DCFC (DC Fast Charging) availability on your routes.
   As of 2025, most major automakers have adopted NACS (Tesla) connector.
3. Home charging: Level 1 (120V outlet) adds ~4 miles/hour. Level 2 (240V) adds 15-37 miles/hour.
   Estimate $800-$2,500 for Level 2 home charger installation.
4. Federal tax credit: Up to $7,500 for new EVs. Income limits apply.
5. Battery warranty: Look for at least 8 years / 100,000 miles battery warranty.
6. Real-world efficiency: Ioniq 6 LR: 25 kWh/100 mi. Model 3 LR: 25 kWh/100 mi."""
    },
    {
        "id": "doc_011",
        "topic": "Car Maintenance Costs — Comparison by Type",
        "text": """Annual maintenance costs vary significantly by car type and brand:
EVs: Average $550-$900/year. No oil changes, no spark plugs, fewer brake replacements.
Japanese brands (Toyota, Honda): Average $400-$700/year. Renowned for reliability and affordable parts.
German luxury brands (BMW, Mercedes, Audi): Average $1,000-$1,800/year. Complex engineering, expensive parts.
American trucks (Ford F-150, Chevy Silverado): Average $700-$950/year.
Full brake service (pads + rotors, all 4 wheels): $400-$900 typical.
Tire replacement (set of 4): $400-$1,200 depending on size and brand.
A Toyota RAV4 costs significantly less to maintain over 10 years than a BMW X3 of similar class."""
    },
    {
        "id": "doc_012",
        "topic": "Car Buying Tips — New vs Used, Negotiation",
        "text": """Key tips for buying a car wisely:
New vs Used: New cars offer warranties and latest tech but depreciate 15-25% in year one.
Certified Pre-Owned (CPO) vehicles offer manufacturer warranties — best of both worlds.
Best time to buy: End of month, end of quarter (March, June, September, December), or holiday weekends.
Know the invoice price: Use TrueCar, Edmunds, or CarGurus to find dealer invoice price before negotiating.
Target 2-4% above invoice for popular models; at or below invoice during slow sales periods.
Finance wisely: Get pre-approved by your bank or credit union before visiting a dealer.
Avoid extending loan term beyond 60 months to prevent being underwater (owing more than car is worth).
Total cost of ownership matters more than sticker price: include insurance, fuel, maintenance, depreciation."""
    },
]

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


@st.cache_resource
def load_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection("cars_kb")
    except:
        pass
    collection = client.create_collection("cars_kb")
    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    class CarsAgentState(TypedDict):
        question: str
        messages: List[dict]
        route: str
        retrieved: str
        sources: List[str]
        tool_result: str
        answer: str
        faithfulness: float
        eval_retries: int
        car_brand: str

    def memory_node(state):
        msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
        return {"messages": msgs[-6:]}

    def router_node(state):
        msgs = state.get("messages", [])
        recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in msgs[-3:-1]) or "none"
        prompt = f"""You are a router for a car information chatbot covering specs, pricing, safety, and buying advice.

Available routing options:
- retrieve: search the knowledge base for car specs, pricing, safety ratings, EV guides, maintenance costs
- memory_only: answer from conversation history (e.g. 'what did you just say?', 'tell me more')
- tool: use web search for LIVE data — current market prices, latest car releases, ongoing deals, 2024/2025 news

Recent conversation: {recent}
Current question: {state["question"]}

Reply with ONLY one word: retrieve / memory_only / tool"""
        d = llm.invoke(prompt).content.strip().lower()
        if "memory" in d:
            d = "memory_only"
        elif "tool" in d:
            d = "tool"
        else:
            d = "retrieve"
        return {"route": d}

    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()
        res = collection.query(query_embeddings=q_emb, n_results=3)
        topics = [m["topic"] for m in res["metadatas"][0]]
        ctx = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{res['documents'][0][i]}" for i in range(len(topics))
        )
        return {"retrieved": ctx, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(state["question"] + " car 2025", max_results=4))
            if results:
                return {
                    "tool_result": "\n".join(
                        f"{r['title']}: {r['body'][:250]}" for r in results
                    )
                }
            return {"tool_result": "No web search results found. Check Edmunds or CarGurus for live data."}
        except Exception as e:
            return {
                "tool_result": (
                    f"Web search unavailable: {e}. "
                    "Please check official manufacturer websites or Edmunds/CarGurus for live pricing."
                )
            }

    def answer_node(state):
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        eval_retries = state.get("eval_retries", 0)
        ctx_parts = []
        if retrieved:
            ctx_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            ctx_parts.append(f"WEB SEARCH RESULTS (live data):\n{tool_result}")
        ctx = "\n\n".join(ctx_parts)

        if ctx:
            sys_content = (
                "You are an expert automotive advisor. Answer using ONLY the information provided "
                "in the context below. Be specific with numbers, figures, and model names. "
                "If the answer is not in the context, say: "
                "'I don't have that specific information — please check the manufacturer's website or Edmunds.'\n\n"
                + ctx
            )
        else:
            sys_content = "You are a helpful automotive advisor. Answer based on the conversation history. Be concise."

        if eval_retries > 0:
            sys_content += "\n\nIMPORTANT: Stick strictly to the provided data. Do not add information not in the context."

        msgs = [SystemMessage(content=sys_content)]
        for m in state.get("messages", []):
            msgs.append(
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
            )
        return {"answer": llm.invoke(msgs).content}

    def eval_node(state):
        ctx = state.get("retrieved", "")[:400]
        retries = state.get("eval_retries", 0)
        if not ctx:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = (
            f"Rate faithfulness 0.0-1.0: does this answer use ONLY information from the context? "
            f"Reply with only a number.\n\nContext: {ctx}\nAnswer: {state.get('answer', '')[:200]}"
        )
        try:
            score = max(0.0, min(1.0, float(llm.invoke(prompt).content.strip().split()[0])))
        except:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state):
        msgs = state.get("messages", []) + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": msgs}

    def route_dec(state):
        r = state.get("route", "retrieve")
        if r == "tool":
            return "tool"
        if r == "memory_only":
            return "skip"
        return "retrieve"

    def eval_dec(state):
        if (
            state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD
            or state.get("eval_retries", 0) >= MAX_EVAL_RETRIES
        ):
            return "save"
        return "answer"

    g = StateGraph(CarsAgentState)
    for name, fn in [
        ("memory", memory_node),
        ("router", router_node),
        ("retrieve", retrieval_node),
        ("skip", skip_retrieval_node),
        ("tool", tool_node),
        ("answer", answer_node),
        ("eval", eval_node),
        ("save", save_node),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges(
        "router", route_dec, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )
    for n in ["retrieve", "skip", "tool"]:
        g.add_edge(n, "answer")
    g.add_edge("answer", "eval")
    g.add_conditional_edges("eval", eval_dec, {"answer": "answer", "save": "save"})
    g.add_edge("save", END)

    agent_app = g.compile(checkpointer=MemorySaver())
    return agent_app, embedder, collection


# ── Load agent ─────────────────────────────────────────────
try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Cars knowledge base loaded — {collection.count()} documents ready")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ── Session state ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("🚗 About This Agent")
    st.write(
        "An expert car advisor powered by LangGraph + ChromaDB RAG + DuckDuckGo web search. "
        "Covers specs, pricing, safety ratings, EVs, maintenance, and buying tips."
    )
    st.write(f"**Session ID:** `{st.session_state.thread_id}`")
    st.divider()
    st.write("**📚 Knowledge Base Topics:**")
    for d in DOCUMENTS:
        st.write(f"• {d['topic']}")
    st.divider()
    st.write("**💡 Try asking:**")
    sample_qs = [
        "Compare RAV4 Hybrid vs CR-V Hybrid MPG",
        "What's the 0-60 of the Porsche 911 Turbo S?",
        "How much does an EV charger cost to install?",
        "When is the best time to buy a car?",
        "What does IIHS TSP+ mean?",
    ]
    for q in sample_qs:
        st.write(f"*\"{q}\"*")
    st.divider()
    if st.button("🗑️ Start New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Chat history ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ─────────────────────────────────────────────
if prompt := st.chat_input("Ask about any car — specs, price, safety, EV range, buying tips..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Researching your question..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")

        st.write(answer)

        # Show metadata footer
        faith = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        route = result.get("route", "?")
        route_icon = {"retrieve": "📚", "tool": "🌐", "memory_only": "💭"}.get(route, "❓")
        st.caption(
            f"{route_icon} Route: **{route}** | "
            f"Faithfulness: **{faith:.2f}** | "
            f"Sources: {', '.join(sources) if sources else 'conversation history'}"
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
