from flask import Flask, render_template, request, Response
import os
from dotenv import load_dotenv

# LangChain
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

# ---------------- Load ENV ----------------
load_dotenv()

app = Flask(__name__)


def sse_message(data, event=None):
    payload = str(data).replace("\r\n", "\n").replace("\r", "\n")
    lines = []

    if event:
        lines.append(f"event: {event}")

    for line in payload.split("\n"):
        lines.append(f"data: {line}")

    return "\n".join(lines) + "\n\n"


def chunk_to_text(chunk):
    content = getattr(chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)

        return "".join(parts)

    return str(content or "")

# ---------------- LLM Loader ----------------
def get_model():
    if os.getenv("USE_OPENAI") == "true":
        return init_chat_model(
            model=os.getenv("OPENAI_MODEL"),
            model_provider="openai",
            streaming=True
        )

    elif os.getenv("USE_GEMINI") == "true":
        return init_chat_model(
            model=os.getenv("GEMINI_MODEL"),
            model_provider="google_genai",
            streaming=True
        )

    elif os.getenv("USE_GROQ") == "true":
        return init_chat_model(
            model=os.getenv("GROQ_MODEL"),
            model_provider="groq",
            streaming=True
        )

    elif os.getenv("USE_ANTHROPIC") == "true":
        return init_chat_model(
            model=os.getenv("ANTHROPIC_MODEL"),
            model_provider="anthropic",
            streaming=True
        )

    else:
        raise ValueError("❌ No LLM provider set to true in .env")


# ---------------- Initialize ----------------
model = get_model()

tavily_tool = TavilySearch(
    max_results=10,
    topic="general"
)


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream_news")
def stream_news():
    query = request.args.get("query", "").strip()

    def generate():
        try:
            yield sse_message("Searching trusted AI sources", event="status")
            search_results = tavily_tool.invoke(query)
            yield sse_message("Reviewing the latest updates", event="status")

            prompt = f"""
You are an AI news assistant.

Use the user's request and the search results to prepare a clean AI news briefing.

Return the answer in this Markdown structure:

### Header

1. [Short headline]
- [2 to 3 sentence summary]

2. [Short headline]
- [2 to 3 sentence summary]

[Continue numbering only if needed]

### Sources
- [Source name]: [URL]

Rules:
- Use valid Markdown spacing.
- Keep spacing natural and readable.
- The header must match the user's request. If the user asks for the last 24 hours, use a header like "AI News From the Last 24 Hours". If the user asks for today, use "Today's AI News". If the user asks for a weekly summary, reflect that.
- Decide the number of news items from the user's request and the available relevant results. Do not force exactly 3 items unless the user explicitly asks for 3.
- Respect the requested timeframe exactly. If the user asks for the past 24 hours, only include news from the past 24 hours when supported by the search results. If the timeframe is unclear or the results are limited, say so briefly in one of the summaries instead of pretending.
- Use short headlines.
- Put each summary directly below its headline.
- Add a blank line between each news item.
- Do not write everything in one paragraph.
- Always include source URLs if available.
- Focus only on the latest relevant AI news.
- If the user asks for simple English, write in simple English.
- If the user asks to avoid jargon, avoid jargon or explain it simply.
- Do not add any introduction or closing note outside this structure.

User Request:
{query}

Search Results:
{search_results}
"""

            yield sse_message("Writing your news briefing", event="status")

            for chunk in model.stream(prompt):
                chunk_text = chunk_to_text(chunk)
                if chunk_text:
                    yield sse_message(chunk_text)

            yield sse_message("complete", event="done")

        except Exception as e:
            yield sse_message(f"Error: {str(e)}")
            yield sse_message("complete", event="done")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
