import streamlit as st
from rag import process_urls, generate_answer

#  UI Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(10,20,30,0.75), rgba(10,20,30,0.75)),
                url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
    background-size: cover;
    background-position: center;
    color: #f1f5f9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

/* 🔥 Fix: input label color (Ask your question) */
label, .stTextInput label {
    color: #e2e8f0 !important;
    font-weight: 600;
}

/* Inputs */
input, textarea {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 8px !important;
}

/* Buttons */
button {
    background: linear-gradient(90deg, #2563eb, #38bdf8) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Card */
.card {
    background: rgba(15, 23, 42, 0.88);
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.5);
    backdrop-filter: blur(10px);
    margin-top: 20px;
}

/* 🔥 STRONG FIX: keep links inside box */
.link {
    word-break: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
    display: block;
}

/* Ensure anchor wraps */
.link a {
    word-break: break-word;
    overflow-wrap: anywhere;
    display: inline-block;
    max-width: 100%;
}
</style>
""", unsafe_allow_html=True)

# 🎯 Header
st.markdown("""
<h1 style='text-align:center;'>🏡 AI Real Estate Intelligence Engine</h1>
<p style='text-align:center;'>Ask questions from real estate articles using AI + RAG</p>
""", unsafe_allow_html=True)

# 📦 Sidebar
st.sidebar.markdown("## 🔍 Input URLs")

url1 = st.sidebar.text_input("🌐 URL 1")
url2 = st.sidebar.text_input("🌐 URL 2")
url3 = st.sidebar.text_input("🌐 URL 3")

status_box = st.empty()

#  Process
if st.sidebar.button("🚀 Process Data"):
    urls = [u for u in (url1, url2, url3) if u.strip()]

    if not urls:
        status_box.error("⚠️ Please enter at least one URL")
    else:
        for step in process_urls(urls):
            status_box.info(step)

#  Query (Label now visible)
query = st.text_input("💬 Ask your question")

#  Output
if query:
    try:
        answer, sources = generate_answer(query)

        # Answer
        st.markdown(f"""
        <div class="card">
        <h3>📌 Answer</h3>
        <p style="line-height:1.6;">{answer}</p>
        </div>
        """, unsafe_allow_html=True)

        # Sources
        if sources:
            st.markdown("""
            <div class="card">
            <h3>🔗 Sources</h3>
            """, unsafe_allow_html=True)

            for src in sources:
                st.markdown(f"""
                <div class="link">
                🔗 <a href="{src}" target="_blank">{src}</a>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    except RuntimeError:
        st.error("⚠️ Please process URLs first")