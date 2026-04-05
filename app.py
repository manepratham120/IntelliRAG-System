import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Document QA Engine",
    page_icon="📄",
    layout="centered",
)

# ── Force light theme + styling ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
        background-color: #F7F5F2 !important;
        color: #1A1A1A !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: #F7F5F2 !important;
    }

    .block-container {
        padding-top: 3rem !important;
        max-width: 660px !important;
    }

    p, span, label, div {
        color: #1A1A1A !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .step-pill {
        display: inline-block;
        background: #E8E2D9;
        color: #6B5E4E !important;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 6px;
    }

    .step-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1A1A1A !important;
        margin-bottom: 0.7rem;
        margin-top: 0.2rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #C0B5A8 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: #4A4A4A !important;
    }

    textarea {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 2px solid #D4C9BB !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.97rem !important;
    }
    textarea::placeholder {
        color: #A09080 !important;
    }
    textarea:focus {
        border-color: #C47F3A !important;
        box-shadow: 0 0 0 3px rgba(196,127,58,0.12) !important;
    }

    .stButton > button {
        background-color: #C47F3A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stButton > button:hover {
        background-color: #A96B2C !important;
    }

    .divider {
        border: none;
        border-top: 1.5px solid #DDD6CC;
        margin: 0.4rem 0 1.8rem;
    }

    /* ── Chat bubbles ── */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .chat-row-user {
        display: flex;
        justify-content: flex-end;
    }

    .chat-row-bot {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 0.55rem;
    }

    .bubble-user {
        background: #C47F3A;
        color: #FFFFFF !important;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1.1rem;
        max-width: 82%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 1px 4px rgba(196,127,58,0.18);
    }

    .bubble-bot {
        background: #FFFFFF;
        color: #1A1A1A !important;
        border-radius: 18px 18px 18px 4px;
        padding: 0.85rem 1.15rem;
        max-width: 82%;
        font-size: 0.95rem;
        line-height: 1.75;
        border-left: 4px solid #C47F3A;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }

    .bot-avatar {
        width: 30px;
        height: 30px;
        min-width: 30px;
        background: #E8E2D9;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        margin-top: 2px;
    }

    .chat-input-area textarea {
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "user"/"bot", "text": "..."}
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📄 RAGnarok AI")
st.markdown(
    "<p style='color:#5C524A; font-size:0.97rem; margin-top:-0.3rem;'>"
    "RAG Based PDF Question-Answering System. </p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#5C524A; font-size:0.97rem; margin-top:-0.3rem;'>"
    "Upload any PDF and ask questions about it — in plain English.</p>",
    unsafe_allow_html=True,
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Step 1: Upload ────────────────────────────────────────────────────────────
st.markdown('<span class="step-pill">Step 1</span>', unsafe_allow_html=True)
st.markdown('<p class="step-title">Upload your PDF</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["pdf"],
    label_visibility="hidden",
)

if uploaded_file is not None:
    # Only reprocess if it's a new file
    if uploaded_file.name != st.session_state.pdf_name:
        save_path = os.path.join(working_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("📖 Reading your document, please wait…"):
            process_document_to_chroma_db(uploaded_file.name)

        # Reset chat for new document
        st.session_state.chat_history = []
        st.session_state.pdf_ready = True
        st.session_state.pdf_name = uploaded_file.name

    st.success(f"✅ **{st.session_state.pdf_name}** is ready! Ask your questions below.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Step 2: Chat ──────────────────────────────────────────────────────────────
st.markdown('<span class="step-pill">Step 2</span>', unsafe_allow_html=True)
st.markdown('<p class="step-title">Ask questions about your document</p>', unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-row-user">'
                f'<div class="bubble-user">{msg["text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-row-bot">'
                f'<div class="bot-avatar">📄</div>'
                f'<div class="bubble-bot">{msg["text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
with st.container():
    user_question = st.text_area(
        label="Your question",
        placeholder="e.g.  What is the main topic of this document?",
        height=100,
        label_visibility="collapsed",
        key="question_input",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        ask_clicked = st.button("Send →", use_container_width=True)
    with col2:
        clear_clicked = st.button("Clear Chat", use_container_width=True)

# ── Handle clear ──────────────────────────────────────────────────────────────
if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

# ── Handle ask ────────────────────────────────────────────────────────────────
if ask_clicked:
    if not st.session_state.pdf_ready:
        st.warning("⚠️ Please upload a PDF first (Step 1 above).")
    elif not user_question.strip():
        st.warning("⚠️ Please type a question before sending.")
    else:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "text": user_question.strip()})

        with st.spinner("🔍 Looking through your document…"):
            answer = answer_question(user_question)

        # Append bot response
        st.session_state.chat_history.append({"role": "bot", "text": answer})
        st.rerun()