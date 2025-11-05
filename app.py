# =========================================
# HireMatch-AI — AI Resume Fit Checker (Hugging Face Ready)
# =========================================

import os
import re
import pdfplumber
from docx import Document
import gradio as gr
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import plotly.graph_objects as go

# ----------------------------
# Setup NLTK
# ----------------------------
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# ----------------------------
# Load AI models
# ----------------------------
print("Loading models... (this may take 1-2 minutes the first time)")
ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Models loaded successfully!")

# ----------------------------
# Extract text from file
# ----------------------------
def extract_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file.name) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file.name.endswith(".docx"):
        doc = Document(file.name)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError("Unsupported file type. Please upload PDF or DOCX.")
    return text.strip()

# ----------------------------
# Clean text
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ----------------------------
# Auto Skill Extraction using NER
# ----------------------------
def extract_skills_auto(text):
    entities = ner_model(text)
    skills = [e['word'] for e in entities if e['entity_group'] in ['ORG', 'MISC']]
    skills = list(set([s.lower() for s in skills if len(s) > 2]))
    return skills

# ----------------------------
# Semantic Skills Scoring
# ----------------------------
def get_skills_score_semantic(resume_text, jd_text, similarity_threshold=0.7):
    resume_skills = extract_skills_auto(resume_text)
    jd_skills = extract_skills_auto(jd_text)

    matched_skills = set(resume_skills).intersection(set(jd_skills))
    missing_skills = set(jd_skills) - set(resume_skills)

    if resume_skills and jd_skills:
        resume_embeddings = embedding_model.encode(resume_skills, convert_to_tensor=True)
        jd_embeddings = embedding_model.encode(jd_skills, convert_to_tensor=True)
        cos_sim = util.cos_sim(resume_embeddings, jd_embeddings)

        for i, res_skill in enumerate(resume_skills):
            for j, jd_skill in enumerate(jd_skills):
                if cos_sim[i][j] > similarity_threshold:
                    matched_skills.add(jd_skill)

    score = len(matched_skills) / (len(jd_skills) or 1)
    return round(score, 2), matched_skills, missing_skills

# ----------------------------
# ATS Scoring System
# ----------------------------
def ats_experience_score(resume_text, jd_text):
    years = re.findall(r'(\d+)\s+years?', resume_text.lower())
    resume_years = max([int(y) for y in years]) if years else 0
    jd_years = re.findall(r'(\d+)\+?\s+years?', jd_text.lower())
    jd_required = int(jd_years[0]) if jd_years else 0
    return round(min(resume_years / jd_required, 1), 2) if jd_required else 0

def ats_format_score(resume_text):
    headings = ["experience", "education", "skills", "projects"]
    heading_score = sum(1 for h in headings if h in resume_text.lower()) / len(headings)
    bullets = len(re.findall(r'•|\*|-', resume_text))
    bullet_score = 1 if bullets >= 3 else 0.5
    return round(0.7 * heading_score + 0.3 * bullet_score, 2)

def ats_total_score(resume_text, jd_text, skills_score):
    exp_score = ats_experience_score(resume_text, jd_text)
    fmt_score = ats_format_score(resume_text)
    total = round((skills_score * 0.6) + (exp_score * 0.3) + (fmt_score * 0.1), 2)
    breakdown = {"skills": skills_score, "experience": exp_score, "formatting": fmt_score}
    return total, breakdown

# ----------------------------
# Dashboard Visualization
# ----------------------------
def create_dashboard(breakdown):
    categories = list(breakdown.keys())
    values = [v * 100 for v in breakdown.values()]
    fig = go.Figure([go.Bar(x=categories, y=values, marker_color='teal')])
    fig.update_layout(
        title="Resume ATS Sub-Scores",
        yaxis=dict(title="Score (%)", range=[0, 100]),
        xaxis=dict(title="Factors"),
        template="plotly_white"
    )
    return fig

# ----------------------------
# Main Analysis Function
# ----------------------------
def analyze_resume_and_jd(resume_file, resume_text, jd_text):
    try:
        if resume_file:
            resume_content = extract_text_from_file(resume_file)
        elif resume_text:
            resume_content = resume_text
        else:
            return "❌ Please upload or paste your resume.", None, None, None, None, None

        if not jd_text:
            return "❌ Please paste a Job Description.", None, None, None, None, None

        skills_score, matched_skills, missing_skills = get_skills_score_semantic(resume_content, jd_text)
        total_score, breakdown = ats_total_score(resume_content, jd_text, skills_score)
        dashboard = create_dashboard(breakdown)

        return (
            resume_content[:2000] + "..." if len(resume_content) > 2000 else resume_content,
            f"ATS Total Score: {total_score*100:.1f}%",
            f"Matched Skills: {', '.join(matched_skills) or 'None'}",
            f"Missing Skills: {', '.join(missing_skills) or 'None'}",
            f"Sub-scores → Skills: {breakdown['skills']*100:.0f}%, Exp: {breakdown['experience']*100:.0f}%, Format: {breakdown['formatting']*100:.0f}%",
            dashboard
        )
    except Exception as e:
        return f"⚠️ Error: {str(e)}", None, None, None, None, None

# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 HireMatch-AI — Auto Skill Extraction + ATS Dashboard")

    with gr.Row():
        with gr.Column():
            resume_file = gr.File(label="📁 Upload Resume (PDF/DOCX)")
            resume_text = gr.Textbox(lines=5, placeholder="Or paste your resume text...", label="Or paste resume text")
            jd_text = gr.Textbox(lines=5, placeholder="Paste Job Description (JD)...", label="Job Description (JD)")
            analyze_button = gr.Button("🚀 Analyze Resume & JD")

        with gr.Column():
            resume_preview = gr.Textbox(label="Resume Preview")
            total_score = gr.Textbox(label="ATS Total Score")
            matched_box = gr.Textbox(label="Matched Skills")
            missing_box = gr.Textbox(label="Missing Skills")
            subscores_box = gr.Textbox(label="Detailed Sub-Scores")
            dashboard_plot = gr.Plot(label="ATS Score Dashboard")

    analyze_button.click(
        analyze_resume_and_jd,
        inputs=[resume_file, resume_text, jd_text],
        outputs=[resume_preview, total_score, matched_box, missing_box, subscores_box, dashboard_plot]
    )

# ----------------------------
# Launch for Hugging Face
# ----------------------------
if __name__ == "__main__":
    demo.launch()
