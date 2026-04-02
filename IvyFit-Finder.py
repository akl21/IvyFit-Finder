# app.py
import streamlit as st
import openai
import numpy as np
from typing import Dict

#Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"

# --- SCHOOL DESCRIPTORS ---
schools: Dict[str, str] = {
    "Harvard University": "Cosmopolitan, ambitious, tradition-infused, globally networked, intellectually rigorous, civic-minded; Leadership cultivation, research excellence, global citizenship, access, societal impact.",
    "Yale University": "Collegial, historically rich, introspective, community-focused, worldly, artistic; Liberal-arts depth, ethical leadership, global engagement, diversity, service orientation.",
    "Princeton University": "Scholarly, intimate, service-oriented, tradition-conscious, reflective, idealistic; Undergraduate focus, research-teaching integration, ethical leadership, service to humanity, academic excellence.",
    "Columbia University": "Urban, activist, international, intellectually demanding, culturally engaged, politically aware; Urban research hub, diversity, global affairs leadership, innovation, public discourse.",
    "Brown University": "Open-minded, inventive, progressive, student-driven, interdisciplinary, socially conscious; Freedom in learning, interdisciplinary collaboration, equity and inclusion, social justice innovation, community engagement.",
    "University of Pennsylvania": "Entrepreneurial, pragmatic, interdisciplinary, urban, energetic, collaborative; Innovation ecosystem, social entrepreneurship, community partnerships, applied research, global connectivity.",
    "Cornell University": "Inclusive, civic-minded, practical, curious, collegial, nature-connected; Land-grant mission, public engagement, scientific discovery, sustainability, access and equity.",
    "Dartmouth College": "Close-knit, outdoorsy, loyal, tradition-loving, adventurous, academically strong; Undergraduate leadership, experiential learning, sustainability, civic service, alumni community engagement.",
    "Stanford University": "Visionary, entrepreneurial, global, forward-thinking, innovative, tech-immersed; Research breakthroughs, technology transfer, global impact, interdisciplinary innovation.",
    "MIT (Massachusetts Institute of Technology)": "Rational, innovative, meritocratic, problem-focused, collaborative, understated; Scientific excellence, engineering solutions, global problem-solving, real-world impact, equity in STEM.",
    "University of Chicago": "Intellectual, contrarian, philosophical, rigorous, urban, principled; Fundamental inquiry, critical discourse, freedom of expression, academic rigor, truth-seeking.",
    "Duke University": "Energetic, collaborative, dynamic, community-oriented, forward-looking, ethically aware; Interdisciplinary research, innovation in health and environment, civic service, leadership training, global engagement.",
    "Northwestern University": "Creative, multidisciplinary, flexible, urbane, balanced, ambitious; Integration of arts and sciences, communication and design innovation, entrepreneurship, research excellence, diversity.",
    "Caltech (California Institute of Technology)": "Intense, focused, precise, experimental, collaborative, science-centered; Fundamental research, scientific integrity, technology advancement, discovery-to-application pipeline, global collaboration.",
    "Johns Hopkins University": "Research-driven, analytical, global, health-conscious, pragmatic, serious-minded; Translational research, health and policy leadership, innovation through data science, societal benefit, global partnerships.",
    "UC Berkeley (University of California, Berkeley)": "Activist, diverse, intellectually bold, public-spirited, innovative, independent; Public mission, social equity, academic freedom, scientific excellence, innovation for the common good.",
}

def get_embedding(text: str) -> np.ndarray:
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def rank_schools(student_input: str) -> Dict[str, float]:
    student_emb = get_embedding(student_input)
    results = {}
    for school, desc in schools.items():
        school_emb = get_embedding(desc)
        score = cosine_similarity(student_emb, school_emb)
        results[school] = score
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

st.title("🎓 IvyFit Finder")
st.write("Match your student vibe to an Ivy or Alt-Ivy school using semantic similarity.")

student_descriptors = st.text_input(
    "Describe yourself in 3–5 words or short phrases (comma-separated):",
    placeholder="e.g. creative, socially conscious, research-oriented, collaborative"
)

if st.button("Find My Match") and student_descriptors:
    with st.spinner("Crunching the Ivy matrix..."):
        rankings = rank_schools(student_descriptors)
    st.subheader("Top Matching Schools:")
    for i, (school, score) in enumerate(list(rankings.items())[:5], start=1):
        st.write(f"**{i}. {school}** — similarity score: `{score:.3f}`")
    st.markdown("---")
    st.write("*(Uses OpenAI embeddings to measure conceptual similarity between your traits and each school's published ethos.)*")
