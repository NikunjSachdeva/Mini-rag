# gradio_ui.py
import gradio as gr
import requests
import json
from typing import Dict, List

# --- CONFIG ---
BACKEND_URL = "http://0.0.0.0:8000"  # Update if your FastAPI is hosted elsewhere

# --- Helper Functions ---
def ingest_text_pretty(text: str, source: str = "user_input", title: str = "Untitled") -> str:
    if not text.strip():
        return "<div style='color: #ff6b6b; font-weight: bold;'>❌ Text cannot be empty.</div>"

    try:
        response = requests.post(f"{BACKEND_URL}/ingest", json={
            "text": text,
            "source": source,
            "title": title
        })
        result = response.json()

        if result.get("status") == "success":
            details = result["details"]
            chunks = details["chunks_ingested"]
            words = details["total_words"]
            time_sec = details["processing_time"]
            dim = details["chunk_stats"]["embedding_dimensions"]
            avg_size = details["chunk_stats"]["avg_chunk_size"]

            return f"""
            <div style="
                background: linear-gradient(135deg, #1e1e2f, #2d2d40);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 8px 24px rgba(0,0,0,0.2);
                border: 1px solid #3a3a50;
                font-family: 'Inter', sans-serif;
                color: #e0e0e0;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span style="
                        font-size: 2rem;
                        margin-right: 0.5rem;
                    ">✅</span>
                    <h3 style="color: #bb86fc; margin: 0;">Ingestion Successful!</h3>
                </div>

                <p style="margin: 0.8rem 0; font-size: 1.1em;">
                    <strong>'{title}'</strong> from <code style="color: #03dac6;">{source}</code> was ingested successfully.
                </p>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
                    <div class="metric-box">
                        <div class="metric-value">{chunks}</div>
                        <div>Chunks</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{words}</div>
                        <div>Words</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{avg_size}</div>
                        <div>Avg Size</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{dim}</div>
                        <div>Dimensions</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value" style="color:#03dac6;">{time_sec:.2f}s</div>
                        <div>Time</div>
                    </div>
                </div>

                <div style="margin-top: 1rem; font-size: 0.9em; color: #aaa;">
                    📅 <strong>Status:</strong> {result['status'].title()} |
                    🔖 <strong>Chunks:</strong> {', '.join(map(str, details['chunk_ids']))}
                </div>
            </div>
            """
        else:
            error = result.get("message", "Unknown error")
            return f"""
            <div style="
                background: #3a1a1a;
                color: #ff6b6b;
                padding: 1.2rem;
                border-radius: 12px;
                border: 1px solid #ff4444;
                font-weight: bold;
            ">
                ❌ <strong>Ingestion Failed:</strong><br>
                <span style="font-size: 0.9em;">{error}</span>
            </div>
            """

    except Exception as e:
        return f"""
        <div style="
            background: #2a1a2a;
            color: #ff5588;
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid #b03060;
        ">
            🌐 <strong>Connection Error:</strong><br>
            <code style="font-size: 0.9em;">{str(e)}</code>
        </div>
        """

def query_backend(query: str, top_k: int = 5) -> Dict:
    try:
        response = requests.post(f"{BACKEND_URL}/query", json={
            "query": query,
            "top_k": top_k,
            "include_scores": True
        })
        return response.json()
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def batch_query_backend(queries: List[str], top_k: int = 5) -> List[Dict]:
    try:
        response = requests.post(f"{BACKEND_URL}/batch-query", json={
            "queries": queries,
            "top_k": top_k,
            "include_scores": True
        })
        return response.json().get("results", [])
    except Exception as e:
        return [{"error": f"Batch query failed: {str(e)}"} for _ in queries]

def get_stats() -> Dict:
    try:
        response = requests.get(f"{BACKEND_URL}/stats")
        return response.json()
    except:
        return {"error": "Could not fetch stats"}

def get_health() -> Dict:
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        return response.json()
    except:
        return {"status": "unhealthy"}

# --- Custom CSS for Modern Look ---
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #111118 0%, #1e1e2d 100%);
    color: #e0e0e0;
}
h1 {
    color: #bb86fc !important;
    text-align: center;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
h2, h3 {
    color: #03dac6 !important;
}
.gr-button-primary {
    background: linear-gradient(90deg, #bb86fc, #03dac6) !important;
    border: none !important;
    color: #111 !important;
    font-weight: 600;
    border-radius: 8px;
}
.gr-button-secondary {
    background: #2d2d3a !important;
    color: #bb86fc !important;
    border-radius: 8px;
}
#output-box {
    background: #1e1e2a;
    border: 1px solid #3a3a50;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.citation {
    background: #2d2d40;
    border-left: 3px solid #bb86fc;
    padding: 0.5rem;
    margin: 0.5rem 0;
    border-radius: 6px;
    font-size: 0.9em;
}
.source-item {
    background: #2a2a3a;
    padding: 0.7rem;
    margin: 0.4rem 0;
    border-radius: 8px;
    border: 1px solid #3a3a50;
}
.metric-box {
    background: #2d2d3a;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-size: 0.95em;
}
.metric-value {
    font-size: 1.4em;
    font-weight: bold;
    color: #03dac6;
}
"""

# --- UI Components ---
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 1rem;">
        <h1>🧠 Mini-RAG Pro</h1>
        <p style="color: #aaa; font-size: 1.1em;">Intelligent Retrieval-Augmented Generation with Citations &amp; Scoring</p>
    </div>
    """)

    with gr.Tabs():
        # === INGEST TAB ===
        with gr.Tab("📥 Ingest Documents"):
            with gr.Row():
                with gr.Column(scale=3):
                    ingest_textbox = gr.Textbox(
                        label="Document Text",
                        placeholder="Paste your article, notes, or research paper here...",
                        lines=12
                    )
                    with gr.Row():
                        ingest_source = gr.Textbox(label="Source", value="user_input", scale=1)
                        ingest_title = gr.Textbox(label="Title", value="Untitled", scale=2)
                    ingest_btn = gr.Button("💾 Ingest Text", variant="primary")
                with gr.Column(scale=2):
                    ingest_output = gr.JSON(label="Ingestion Result", elem_id="output-box")

            ingest_btn.click(
                fn=ingest_text_pretty,
                inputs=[ingest_textbox, ingest_source, ingest_title],
                outputs=ingest_output  # Now outputs HTML/Markdown
)

        # === QUERY TAB ===
        with gr.Tab("🔍 Query & Answer"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything based on the ingested documents...",
                        lines=3
                    )
                    top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K Results")
                    query_btn = gr.Button("🧠 Get Answer", variant="primary")
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.HTML("<h3>📊 Response Stats</h3>")
                        stats_output = gr.JSON(label="Performance", elem_id="output-box")

            answer_output = gr.Markdown(label="Answer with Citations", elem_id="output-box")
            sources_output = gr.JSON(label="Source Documents", elem_id="output-box")

            def handle_query(query, k):
                if not query.strip():
                    return "Please enter a valid question.", {}, {}
                result = query_backend(query, k)
                if "error" in result:
                    return f"❌ {result['error']}", {}, {}
                answer = result.get("answer", "No answer generated.")
                citations = result.get("citations", [])
                sources = result.get("sources", [])
                metadata = result.get("metadata", {})
                return answer, metadata, sources

            query_btn.click(
                fn=handle_query,
                inputs=[query_input, top_k_slider],
                outputs=[answer_output, stats_output, sources_output]
            )

        # === BATCH QUERY TAB ===
        with gr.Tab("📦 Batch Query"):
            batch_input = gr.Textbox(
                label="Multiple Questions (one per line)",
                placeholder="Question 1\nQuestion 2\n...",
                lines=6
            )
            batch_top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K per Query")
            batch_btn = gr.Button("🚀 Process Batch", variant="primary")
            batch_output = gr.JSON(label="Batch Results", elem_id="output-box")

            def run_batch(questions, k):
                q_list = [q.strip() for q in questions.split("\n") if q.strip()]
                if not q_list:
                    return {"error": "No valid questions provided."}
                results = batch_query_backend(q_list, k)
                return {"results": results}

            batch_btn.click(
                fn=run_batch,
                inputs=[batch_input, batch_top_k],
                outputs=batch_output
            )

        # === STATS TAB ===
        with gr.Tab("📊 System Stats"):
            gr.HTML("<h3>System & Index Information</h3>")
            health_status = gr.Textbox(label="Health")
            stats_display = gr.JSON(label="Index Stats")

            def refresh_stats():
                health = get_health()
                stats = get_stats()
                health_text = health.get("status", "unknown")
                return health_text, stats

            refresh_btn = gr.Button("🔄 Refresh Stats")
            refresh_btn.click(fn=refresh_stats, outputs=[health_status, stats_display])

    # --- Footer ---
    gr.HTML("""
    <div style="text-align: center; margin-top: 2rem; color: #666; font-size: 0.9em;">
        Powered by FastAPI • Pinecone • Google Gemini • Cohere Rerank • Gradio
    </div>
    """)

# --- Launch App ---
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,  # Set to True for public link
        debug=True
    )