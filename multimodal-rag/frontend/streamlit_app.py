from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent
for path in (str(ROOT_DIR), str(FRONTEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.core.config import is_rag_debug_enabled
from frontend.api_client import APIClient

client = APIClient()


def main():
    st.set_page_config(page_title="Multimodal RAG", layout="wide")
    st.title("Multimodal RAG")

    with st.sidebar:
        st.header("Documents")
        if st.button("Refresh list"):
            st.rerun()

        # Upload
        uploaded = st.file_uploader(
            "Upload document",
            type=["pdf", "doc", "docx", "txt", "md", "png", "jpg", "mp3", "wav", "m4a", "flac", "ogg"],
        )
        if uploaded is not None:
            save_path = ROOT_DIR / "data" / "uploads" / uploaded.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as fh:
                fh.write(uploaded.getbuffer())
            # Call API upload
            try:
                res = client.upload_document(str(save_path))
                st.success(f"Uploaded {res.get('filename')} as {res.get('document_id')}")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

        # List documents
        try:
            docs = client.list_documents()
            st.write(f"Documents: {docs.get('count')}")
        except Exception:
            st.write("Documents: (unavailable)")

    st.header("Chat")
    query = st.text_input("Ask a question", key="query_input")
    if st.button("Clear"):
        st.session_state["query_input"] = ""
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)
    if st.button("Ask") and query:
        with st.spinner("Retrieving..."):
            try:
                resp = client.query(query, top_k=top_k)
                st.subheader("Answer")
                answer = resp.get("answer")
                if not answer:
                    st.warning("No answer was generated for this query.")
                else:
                    # If the backend returned aggregated sub-answers prefixed with Q1:, Q2:,
                    # render them as a list for improved UX.
                    import re

                    lines = [l.strip() for l in answer.splitlines() if l.strip()]
                    q_pattern = re.compile(r"^Q\d+:\s*(.*)$")
                    parsed = [q_pattern.match(l) for l in lines]
                    if parsed and any(m for m in parsed if m):
                        for m in parsed:
                            if m:
                                st.markdown(f"- {m.group(1)}")
                    else:
                        st.markdown(answer)

                st.subheader("Sources")
                def fmt_time(s):
                    try:
                        s = float(s)
                    except Exception:
                        return ""
                    mins = int(s // 60)
                    secs = int(s % 60)
                    return f"{mins}:{secs:02d}"

                import streamlit.components.v1 as components

                for i, src in enumerate(resp.get("sources", []), start=1):
                    md = src.get("metadata") or {}
                    filename = md.get("filename") or src.get("filename")
                    source_path = md.get("source_path") or filename
                    source_type = str(md.get("content_type") or src.get("content_type") or "").lower()
                    fileext = Path(filename or "").suffix.lower() if filename else ""
                    is_audio = "audio" in source_type or fileext in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
                    st.markdown(f"**Source {i}:** {filename}")

                    # Show pretty timestamps when available
                    start = md.get("start_time")
                    end = md.get("end_time")
                    if start is not None or end is not None:
                        start_label = fmt_time(start) if start is not None else ""
                        end_label = fmt_time(end) if end is not None else ""
                        st.write(f"Segment: {start_label} — {end_label}")

                    if is_audio:
                        try:
                            url = client.fetch_file_url(filename)
                            st.audio(url)
                            st.markdown(f"[Download clip]({url})")
                        except Exception:
                            st.write("(file unavailable)")

                        if start is not None:
                            try:
                                audio_url = client.fetch_file_url(filename)
                                start_sec = float(start or 0)
                                player_id = f"player_{i}"
                                html = f"""
                                <audio id='{player_id}' src='{audio_url}' controls></audio>
                                <script>
                                const audio_{i} = document.getElementById('{player_id}');
                                audio_{i}.addEventListener('loadedmetadata', function() {{
                                    audio_{i}.currentTime = {start_sec};
                                }});
                                function play_{i}() {{ audio_{i}.currentTime = {start_sec}; audio_{i}.play(); }}
                                </script>
                                """
                                components.html(html, height=100)
                                if st.button(f"Play segment ({start_label})", key=f"play_{i}"):
                                    st.rerun()
                            except Exception:
                                pass
                    elif fileext in {".pdf", ".doc", ".docx", ".txt", ".md", ".markdown", ".png", ".jpg", ".jpeg"}:
                        try:
                            url = client.fetch_file_url(filename)
                            st.markdown(f"[Open file]({url})")
                        except Exception:
                            st.write("(file unavailable)")

                if resp.get("images"):
                    st.subheader("Images")
                    for img in resp.get("images"):
                        # Display via API file endpoint
                        url = client.fetch_file_url(Path(img).name)
                        st.image(url, caption=Path(img).name)

                if is_rag_debug_enabled() and resp.get("retrieval_debug"):
                    debug = resp["retrieval_debug"]
                    with st.expander("RAG Retrieval Debug"):
                        st.subheader("QUERY")
                        st.write(debug.get("query"))

                        st.subheader("TEXT RETRIEVED")
                        text_rows = [
                            {
                                "Rank": item.get("rank"),
                                "Source": item.get("source"),
                                "Page": item.get("page"),
                                "Score": item.get("score"),
                                "Text": item.get("text_preview"),
                            }
                            for item in debug.get("text_results", [])
                        ]
                        st.dataframe(text_rows, use_container_width=True)

                        st.subheader("IMAGE RETRIEVED")
                        image_rows = [
                            {
                                "Rank": item.get("rank"),
                                "Source": item.get("source"),
                                "Page": item.get("page"),
                                "Score": item.get("score"),
                                "Caption": item.get("caption"),
                            }
                            for item in debug.get("image_results", [])
                        ]
                        st.dataframe(image_rows, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
