# Lab 1 — Introduction to LLMs with LangChain

A hands-on lab exploring Large Language Models (LLMs) using the LangChain framework. Covers local and cloud-based inference, token counting, structured output extraction, and multimodal (vision) capabilities.

---

## Objectives

- Connect to different LLM providers (Ollama, Groq)
- Understand LangChain's message structure (SystemMessage / HumanMessage)
- Count and understand tokens using TikToken
- Build a structured NLP task: aspect-based sentiment analysis
- Use a vision-capable LLM to analyze images

---

## Project Structure

```
lab1_project/
├── lab1.ipynb                  # Main notebook — all demonstrations
├── main.py                     # Minimal entry point
├── key-benefits-of-ai-1.jpg    # Image used in vision demo
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependency versions
├── .env                        # API keys (not committed)
├── .python-version             # Python version pin (3.14)
└── .gitignore
```

---

## Prerequisites

- Python 3.14+
- [`uv`](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.com/) installed locally with the `llama3.2:3b` model pulled
- A [Groq](https://console.groq.com/) API key

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mohammedsouhli/ai-agentique-labs
cd lab1_project
```

### 2. Create and activate the virtual environment

```bash
uv venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure environment variables

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Pull the local Ollama model

```bash
ollama pull llama3.2:3b
```

### 6. Launch Jupyter

```bash
jupyter notebook lab1.ipynb
```

---

## Lab Steps

### Step 1 — Local LLM with Ollama

Run a local LLM entirely offline using `ChatOllama` with the `llama3.2:3b` model. No API key required.

```python
from langchain_ollama import ChatOllama
from IPython.display import Markdown

llm = ChatOllama(model="llama3.2:3b")
response = llm.invoke("C'est quoi un Agent AI")
Markdown(response.content)
```

**Concept:** Local inference — privacy-first, no cloud dependency.

---

### Step 2 — Token Counting with TikToken

Encode a text string using GPT-4o's tokenizer to understand how LLMs split text into tokens.

```python
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")
tokens = encoder.encode("Your system message here")
print(f"Token count: {len(tokens)}")
print(f"Token IDs: {tokens}")
```

**Concept:** Tokens are the unit of cost and context for LLM APIs. Counting them helps estimate costs and stay within context limits.

---

### Step 3 — Cloud LLM with Groq (SystemMessage + HumanMessage)

Use LangChain's message format to send a structured conversation to Groq's `llama-3.3-70b-versatile` model.

```python
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

llm = ChatGroq(model="llama-3.3-70b-versatile")
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="C'est quoi un Agent AI")
]
response = llm.invoke(messages)
```

**Concept:** Separating system instructions from user input gives the model a role and context before it sees the question.

---

### Step 4 — Aspect-Based Sentiment Analysis

Build a structured NLP pipeline using a detailed system prompt. The model analyzes a product review, extracts named aspects (screen, keyboard, trackpad), and assigns a polarity (positive / negative / neutral) to each.

**Input:**
```
"L'écran est très bon, mais je n'ai pas aimé la souris. le clavier est correct"
```

**Expected JSON output:**
```json
{
  "aspects": [
    {"aspect": "écran", "polarity": "positive"},
    {"aspect": "souris", "polarity": "negative"},
    {"aspect": "clavier", "polarity": "neutral"}
  ]
}
```

**Concept:** Prompt engineering for structured outputs — using the LLM as an extraction engine rather than a chat assistant.

---

### Step 5 — Vision / Image Analysis with Groq

Send an image to a multimodal LLM (`meta-llama/llama-4-scout-17b-16e-instruct`) and ask it to describe the content. The image shows AI market size projections (2025–2030).

```python
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import base64

with open("key-benefits-of-ai-1.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
message = HumanMessage(content=[
    {"type": "text", "text": "Décris cette image"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
])
response = llm.invoke([message])
```

**Concept:** Multimodal LLMs can process both text and images in a single API call.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | LLM orchestration and message formatting |
| `langchain-groq` | Groq API integration |
| `langchain-ollama` | Local Ollama model support |
| `langchain-openai` | OpenAI API integration |
| `python-dotenv` | Load API keys from `.env` |
| `tiktoken` | Token counting (OpenAI tokenizer) |
| `ipykernel` | Jupyter notebook kernel |

---

## Key Concepts Covered

| Concept | Tool Used |
|---|---|
| Local LLM inference | Ollama + llama3.2:3b |
| Cloud LLM inference | Groq + llama-3.3-70b-versatile |
| Token encoding | TikToken (gpt-4o) |
| Structured prompting | LangChain SystemMessage / HumanMessage |
| JSON output extraction | Prompt engineering |
| Multimodal (vision) | Groq + llama-4-scout-17b |
| Env variable management | python-dotenv |
| Package management | uv |
