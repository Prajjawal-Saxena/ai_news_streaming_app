from flask import Flask, render_template, request, Response, jsonify
import datetime, os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

# LangChain
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

# ---------------- Load ENV ----------------
load_dotenv()

EMAIL_USER = (os.getenv("EMAIL_USER") or "").strip()
EMAIL_PASS = (os.getenv("EMAIL_PASS") or "").strip().replace(" ", "")

app = Flask(__name__)
DATA_FILE = "data.txt"


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

# ---------------- Email Limit ----------------
def check_email_limit(email):
    today = str(datetime.date.today())

    if not os.path.exists(DATA_FILE):
        return True

    with open(DATA_FILE, "r") as f:
        for line in f:
            saved_email, saved_date = line.strip().split(",")
            if saved_email == email and saved_date == today:
                return False

    return True


def save_email(email):
    today = str(datetime.date.today())

    with open(DATA_FILE, "a") as f:
        f.write(f"{email},{today}\n")


# ---------------- Email Send ----------------
def send_email_to_user(receiver_email, content):
    if not EMAIL_USER:
        raise ValueError("EMAIL_USER is missing in .env")

    if not EMAIL_PASS:
        raise ValueError("EMAIL_PASS is missing in .env")

    if EMAIL_USER.endswith("@gmail.com") and len(EMAIL_PASS) != 16:
        raise ValueError(
            "Gmail requires a 16-character app password in EMAIL_PASS. "
            "Please generate a new app password and restart the app."
        )

    msg = MIMEText(content)
    msg["Subject"] = "Your AI News Update"
    msg["From"] = EMAIL_USER
    msg["To"] = receiver_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)


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
    max_results=5,
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

Return the answer in this exact Markdown structure:

### Top 3 AI News Today

1. [Short headline]
- [2 to 3 sentence summary]

2. [Short headline]
- [2 to 3 sentence summary]

3. [Short headline]
- [2 to 3 sentence summary]

### Sources
- [Source name]: [URL]

Rules:
- Use valid Markdown spacing.
- Keep spacing natural and readable.
- Use short headlines.
- Put each summary directly below its headline.
- Add a blank line between each news item.
- Do not write everything in one paragraph.
- Always include source URLs if available.
- Focus only on the latest relevant AI news.
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


@app.route("/send_email", methods=["POST"])
def send_email():
    email = request.form.get("email")
    content = request.form.get("content")

    if not check_email_limit(email):
        return jsonify({
            "status": "fail",
            "message": "Email already used today."
        })

    try:
        send_email_to_user(email, content)
        save_email(email)

        return jsonify({
            "status": "success",
            "message": "Email sent successfully!"
        })

    except Exception as e:
        return jsonify({
            "status": "fail",
            "message": str(e)
        })


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
