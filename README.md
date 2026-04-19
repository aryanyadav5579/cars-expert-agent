# Cars Expert Agent

### An Intelligent Automotive Advisory System

An AI-powered conversational agent that helps users make informed car-buying decisions by providing accurate information on car specs, pricing, EVs, safety ratings, and maintenance.

Built using **LangGraph + ChromaDB (RAG) + Groq LLM + Streamlit UI**

---

##  Project Overview

The Cars Expert Agent is a full-stack **Agentic AI system** designed to act as a smart automotive advisor.

It combines:

*  A structured knowledge base (car specs, EV guides, safety, maintenance)
*  Real-time web search for live data
*  Memory for multi-turn conversations
*  Intelligent routing to decide how to answer each query

👉 Unlike traditional chatbots, this system can **think, route, retrieve, and evaluate its own answers**.

---

##  Key Features

###  Retrieval-Augmented Generation (RAG)

* Uses **ChromaDB vector database**
* Retrieves top relevant documents using embeddings
* Covers 12 automotive topics (Camry, Tesla, BMW, EVs, safety, etc.)

###  Live Web Search (Tool Use)

* Uses DuckDuckGo search for:

  * Latest car prices
  * New model updates
  * Real-time information

###  Conversation Memory

* Maintains context across chat
* Handles follow-up queries intelligently

###  Intelligent Routing System

The agent decides:

* `retrieve` → use knowledge base
* `tool` → use web search
* `memory_only` → use conversation history

###  Self-Evaluation (Faithfulness Scoring)

* Automatically checks answer quality
* Retries if score < 0.7
* Reduces hallucinations

###  Streamlit Chat Interface

* Clean UI with chat-based interaction
* Sidebar shows:

  * Knowledge topics
  * Sample queries
  * Session ID

---

##  Architecture

User Query
⬇
Memory Node
⬇
Router Node
⬇
Retrieve / Tool / Memory
⬇
LLM Response Generation
⬇
Faithfulness Evaluation
⬇
Final Answer

---

##  Tech Stack

| Component       | Technology            |
| --------------- | --------------------- |
| LLM             | Groq (Llama 3)        |
| Agent Framework | LangGraph             |
| Vector Database | ChromaDB              |
| Embeddings      | Sentence Transformers |
| Web Search      | DuckDuckGo            |
| UI              | Streamlit             |
| Orchestration   | LangChain             |
| Language        | Python                |

---

## 📂 Project Structure

```
cars-expert-agent/
│
├── cars_agent_streamlit.py
├── cars_production_agent.ipynb
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add environment variables

Create `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

### 3. Run the app

```bash
streamlit run cars_agent_streamlit.py
```

---

## 💡 Example Questions

* Compare RAV4 Hybrid vs CR-V Hybrid MPG
* Tesla Model 3 specs
* What does IIHS Top Safety Pick+ mean?
* How much does EV charging cost?
* Best time to buy a car

---

## 📊 Capabilities

* ✅ Accurate RAG-based responses
* ✅ Multi-turn conversation memory
* ✅ Real-time information retrieval
* ✅ Self-correcting answers
* ✅ Transparent output (route + sources + score)

---

## 🚀 Future Improvements

* Expand knowledge base (50+ cars)
* Add personalization (budget, preferences)
* Deploy on cloud (Streamlit Cloud / AWS)
* Add image-based car recognition
* EMI calculator & dealer integration

---

## 📧 Author

 **Aryan Yadav**
  [**aryankyadav5579@gmail.com**](mailto:aryankyadav5579@gmail.com)

---

## ⭐ Final Note

This project demonstrates a real-world **Agentic AI system** combining:

* reasoning
* retrieval
* tool usage
* memory
* evaluation

👉 Not just a chatbot — a smart AI decision-making assistant
