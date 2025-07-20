import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chat With Torben's CV", page_icon="🤖")
st.title("🤖 Chat With My CV (Free, No API Key Needed)")
st.markdown(
    "Ask me about my background, projects, education, or skills. "
    "This demo uses local semantic search over my CV instead of a paid API."
)

# -------- Knowledge Base --------
# Each item: (id, title, text)
KB_ITEMS = [
    ("profile", "Profile", "I'm a mechatronic engineering student at Stellenbosch University with a strong work ethic and an innate desire to learn. I love mathematics and am keen to pursue a meaningful career in data analytics."),
    ("education_uni", "Stellenbosch University", "I am studying toward a BEng in Mechatronics (2022–present) at Stellenbosch University. My current average is 80.35%."),
    ("education_bishops", "Bishops Diocesan College", "I matriculated from Bishops Diocesan College with a 92.1% average, including 98% Physical Sciences, 97% Mathematics, and 83% AP Mathematics."),
    ("leadership_simonsberg", "Leadership: Simonsberg Residence", "I served on the Simonsberg Men's Residence House Committee (2024) with portfolios in maintenance, rugby, and sustainability."),
    ("awards_merit", "Academic Awards", "I received Senior Merit Awards for being in the top 5% of my faculty in 2022, 2023, and 2024. I also won the UCT Mathematics Gold Award in 2018."),
    ("work_aerobotics", "Aerobotics Internship", "At Aerobotics I assisted with machine learning models for fruit detection, including data annotation and quality control on agricultural imaging datasets."),
    ("work_rooibos", "Rooibos LTD Work", "At Rooibos LTD I worked with operations on CAD tasks, pipeline layout, and efficiency forecasting projects."),
    ("skills_technical", "Technical Skills", "I work in Python (PyTorch), R, and C. Comfortable with data analysis, statistics, CAD, and Microsoft Excel/Word/PowerPoint."),
    ("strengths", "Strengths", "My strengths include critical thinking, strong work ethic, time management, and situational awareness."),
    ("interests", "Interests", "I enjoy hiking, rugby (I captain my residence team and coach U20), running, and community outreach like organising football tournaments or painting schools."),
    ("project_fyp", "Final Year Project", "My final year project developed deep learning models to predict the efficiency of a large industrial boiler. The results were good and satisfying."),
    ("why_role", "Why AI Automation", "I believe AI automation is a growing field and I want to be part of it, applying my engineering and data background to real problems."),
    ("future_goals", "5-Year Goals", "In 5 years I hope to have a few strong years of work experience, possibly be pursuing a master's or MBA, and be in a good, healthy relationship."),
    ("why_hire", "Why Hire Me", "I'm aligned with what your company stands for and believe the businesses you work with share a conscious, values-driven message. I bring engineering discipline plus data skills."),
]

# -------- Load Embedding Model Once --------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Precompute embeddings
@st.cache_resource(show_spinner=False)
def build_kb_embeddings(items):
    texts = [t for (_, _, t) in items]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

KB_EMBED = build_kb_embeddings(KB_ITEMS)

# -------- Simple Retrieval Function --------
def retrieve_answer(query, top_k=3, threshold=0.35):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(q_emb, KB_EMBED)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        score = sims[idx]
        if score < threshold:
            continue
        kb_id, title, text = KB_ITEMS[idx]
        results.append((score, title, text))
    return results

def compose_response(query, results):
    if not results:
        return (
            "I'm not sure I understood that. Could you rephrase or ask about one of these: "
            "education, work experience, technical skills, final year project, strengths, or interests?"
        )
    # Use best match as main answer
    main = results[0]
    score, title, text = main
    resp = text
    # If other good matches, add short follow-on suggestions
    extras = [r for r in results[1:] if r[0] > 0.5 * score]
    if extras:
        resp += "\n\nYou might also be interested in: " + ", ".join(t for _, t, _ in extras) + "."
    return resp

# -------- Chat Session State --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- Chat Input --------
user_input = st.chat_input("Ask a question about my background...")

if user_input:
    # Add user msg
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        results = retrieve_answer(user_input)
        reply = compose_response(user_input, results)
    st.session_state.chat_history.append(("assistant", reply))

# -------- Display Chat --------
for role, content in st.session_state.chat_history:
    st.chat_message(role).markdown(content)

# Footer / disclosure
st.markdown(
    "<hr><small>This is a free demo chatbot that uses local semantic search over my CV and "
    "pre-written answers. No external AI API calls are made.</small>",
    unsafe_allow_html=True
)
