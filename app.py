import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chat With Torben's CV", page_icon="🤖")
st.title("🤖 Chat With My CV")
st.markdown("Ask me anything about my background, education, skills or interests!")

# -------- Knowledge Base --------
KB_ITEMS = [
    ("overview", "Overview",
     "I'm Torben Bergh, a mechatronic engineering student from Clanwilliam, South Africa. I'm currently studying at Stellenbosch University and looking to begin a meaningful career in data analytics or AI automation. I'm open to relocating for the right opportunity."),
    
    ("education", "Education",
     "I'm currently pursuing a BEng in Mechatronic Engineering at Stellenbosch University, with an average of 80.35%. Prior to that, I matriculated from Bishops Diocesan College with a 92.1% average, including 98% in Physical Sciences, 97% in Mathematics, and 83% in AP Mathematics."),
    
    ("work", "Work Experience",
     "At Aerobotics, I worked with the data and tech team to help annotate fruit detection datasets for machine learning models. At Rooibos LTD, I assisted with CAD, pipeline layouts, and forecasting efficiency for industrial operations."),
    
    ("leadership", "Leadership & Awards",
     "I served on the Simonsberg Men’s Residence House Committee in 2024, managing portfolios in maintenance, rugby, and sustainability. I’ve consistently ranked in the top 5% of my faculty, receiving Senior Merit Awards in 2022, 2023, and 2024. I also received the UCT Mathematics Gold Award in 2018."),
    
    ("skills", "Skills & Tools",
     "I’m proficient in Python (with PyTorch), R, and C, and skilled in data analysis, statistics, and CAD design. I also use Microsoft Office tools and speak both English and Afrikaans fluently."),
    
    ("project", "Final Year Project",
     "My final-year project involved developing deep learning models to predict the efficiency of a large industrial boiler system using inputs like fuel and air flow rates. The results were accurate and validated on real plant data."),
    
    ("interests", "Hobbies & Interests",
     "I enjoy hiking, running, and rugby – I captain my residence team and coach an under-20 squad. I'm passionate about community outreach, helping organize events like football tournaments and school painting days."),
    
    ("strengths", "Strengths",
     "I’m a critical thinker with a strong work ethic, excellent time management, and good situational awareness. I'm proactive and take pride in finishing tasks thoroughly and efficiently."),
    
    ("whyrole", "Why AI/Automation",
     "I see AI and automation as a growing field that aligns well with both my engineering background and interest in data science. I’d love to work on meaningful systems that make a real impact."),
    
    ("future", "5-Year Vision",
     "In five years, I hope to have solid industry experience behind me, perhaps be working toward a master’s or MBA, and continue growing both personally and professionally."),
    
    ("whyhire", "Why Hire Me",
     "I’m passionate about the mission your company stands for and value working with businesses that have a socially conscious message. I bring a rare combination of technical knowledge, leadership, and values-driven motivation.")
]

# -------- Embedding Model --------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_resource(show_spinner=False)
def build_kb_embeddings(items):
    texts = [t for (_, _, t) in items]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

KB_EMBED = build_kb_embeddings(KB_ITEMS)

# -------- Keyword Matching & Overrides --------
KEYWORD_MAP = {
    "education": "education",
    "university": "education",
    "school": "education",
    "studied": "education",
    "degree": "education",
    "graduated": "education",

    "project": "project",
    "final year": "project",
    "boiler": "project",

    "skills": "skills",
    "tools": "skills",
    "languages": "skills",

    "experience": "work",
    "internship": "work",
    "work": "work",
    "aerobotics": "work",
    "rooibos": "work",

    "strength": "strengths",
    "strong point": "strengths",

    "leadership": "leadership",
    "awards": "leadership",

    "hobby": "interests",
    "interest": "interests",
    "free time": "interests",

    "goal": "future",
    "future": "future",
    "5 years": "future",

    "hire": "whyhire",
    "fit": "whyhire",

    "why": "whyrole",
    "ai": "whyrole",
    "automation": "whyrole",

    "relocate": "relocate",  # Handled below
    "clanwilliam": "relocate"
}

def check_keyword_match(query):
    query_lower = query.lower()
    for keyword, chunk_id in KEYWORD_MAP.items():
        if keyword in query_lower:
            return chunk_id
    return None

def check_override(query):
    query_lower = query.lower()
    if "weakness" in query_lower:
        return "One area I'm working on is balancing my attention to detail with speed. I'm learning to delegate and not overcommit."
    if "relocate" in query_lower or "clanwilliam" in query_lower:
        return "I'm from Clanwilliam, South Africa, and very open to relocating for the right opportunity."
    return None

# -------- Retrieval Logic --------
def retrieve_answer(query, top_k=3, threshold=0.30):
    keyword_match = check_keyword_match(query)
    if keyword_match:
        for (kb_id, title, text) in KB_ITEMS:
            if kb_id == keyword_match:
                return [(1.0, title, text)]

    # Fallback to semantic match
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
            "Hmm, I couldn’t find a perfect match. You can ask me about my education, project, experience, skills, or goals."
        )
    score, title, text = results[0]
    resp = f"**{title}**\n\n{text}"
    extras = [r for r in results[1:] if r[0] > 0.5 * score]
    if extras:
        resp += "\n\nYou might also be interested in: " + ", ".join(t for _, t, _ in extras) + "."
    return resp

# -------- Chat Logic --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about my background...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    # Handle exact keyword or override
    override_reply = check_override(user_input)
    if override_reply:
        reply = override_reply
    else:
        with st.spinner("Thinking..."):
            results = retrieve_answer(user_input)
            reply = compose_response(user_input, results)

    st.session_state.chat_history.append(("assistant", reply))

# -------- Display Chat --------
for role, content in st.session_state.chat_history:
    st.chat_message(role).markdown(content)

st.markdown("<hr><small>This chatbot uses local semantic search to answer interview questions based on my CV and background. No paid APIs or external servers are used.</small>", unsafe_allow_html=True)
