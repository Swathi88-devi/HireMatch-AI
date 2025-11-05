# =========================================
#  HireMatch-AI Resume Fit Checker — Dynamic ATS + Dashboard (Render Ready)
# =========================================

import os
import pdfplumber
from docx import Document
import gradio as gr
import nltk
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go

# ----------------------------
# NLTK Stopwords
# ----------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ----------------------------
# Semantic Skills Dictionary
# ----------------------------
SKILL_SYNONYMS = {
    "ml": ["machine learning", "ml"],
    "ai": ["artificial intelligence", "ai"],
    "python": ["python"],
    "sql": ["sql"],
    "tensorflow": ["tensorflow"],
    "keras": ["keras"],
    "react": ["react", "reactjs"],
    "java": ["java"],
    "pandas": ["pandas"],
    "numpy": ["numpy"]
}

# ----------------------------
# Education Synonyms
# ----------------------------
EDU_SYNONYMS = {
    "btech": ["b.tech", "btech", "bachelor of technology", "bachelor in technology"],
    "mtech": ["m.tech", "mtech", "master of technology"],
    "bsc": ["b.sc", "bsc", "bachelor of science"],
    "msc": ["m.sc", "msc", "master of science"],
    "mba": ["mba", "master of business administration"]
}

CERTIFICATIONS = ["aws", "gcp", "azure", "kaggle", "coursera"]
HEADINGS = ["experience", "education", "skills", "projects", "certifications"]

# ----------------------------
# Load embedding model
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Extract text from resume
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
# Preprocess text
# ----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return tokens

# ----------------------------
# Semantic Skills Score
# ----------------------------
def get_skills_score_semantic(resume_text, jd_text, similarity_threshold=0.7):
    resume_text_lower = resume_text.lower()
    jd_text_lower = jd_text.lower()

    matched_skills = set()
    missing_skills = set()

    for skill, synonyms in SKILL_SYNONYMS.items():
        jd_has_skill = any(s in jd_text_lower for s in synonyms)
        resume_has_skill = any(s in resume_text_lower for s in synonyms)
        if jd_has_skill and resume_has_skill:
            matched_skills.add(skill)
        elif jd_has_skill and not resume_has_skill:
            missing_skills.add(skill)

    resume_phrases = [line.strip() for line in resume_text.split('\n') if line.strip()]
    jd_phrases = [line.strip() for line in jd_text.split('\n') if line.strip()]

    if resume_phrases and jd_phrases:
        resume_embeddings = model.encode(resume_phrases, convert_to_tensor=True)
        jd_embeddings = model.encode(jd_phrases, convert_to_tensor=True)
        cos_sim_matrix = util.cos_sim(resume_embeddings, jd_embeddings)

        for i, r_phrase in enumerate(resume_phrases):
            for j, jd_phrase in enumerate(jd_phrases):
                if cos_sim_matrix[i][j] >= similarity_threshold:
                    matched_skills.add(r_phrase)

    score = len(matched_skills) / (len(matched_skills) + len(missing_skills)) if (matched_skills | missing_skills) else 0
    return round(score, 2), matched_skills, missing_skills

# ----------------------------
# Experience Score
# ----------------------------
def ats_experience_score(resume_text, jd_text):
    years = re.findall(r'(\d+)\s+years?', resume_text.lower())
    resume_years = max([int(y) for y in years]) if years else 0
    jd_years = re.findall(r'(\d+)\+?\s+years?', jd_text.lower())
    jd_required = int(jd_years[0]) if jd_years else 0
    score = min(resume_years / jd_required, 1) if jd_required else 0
    return round(score, 2)

# ----------------------------
# Education Score
# ----------------------------
def ats_education_score(resume_text):
    resume_text_lower = resume_text.lower()
    degree_score = 1 if any(d in resume_text_lower for d in sum(EDU_SYNONYMS.values(), [])) else 0
    cert_score = 1 if any(c in resume_text_lower for c in CERTIFICATIONS) else 0
    return round(0.7 * degree_score + 0.3 * cert_score, 2)

# ----------------------------
# Achievements Score
# ----------------------------
def ats_achievement_score(resume_text):
    metrics = re.findall(r'\d+[%]?', resume_text)
    score = min(len(metrics) / 5, 1)
    return round(score, 2)

# ----------------------------
# Formatting Score
# ----------------------------
def ats_format_score(resume_text):
    heading_count = sum(1 for h in HEADINGS if h in resume_text.lower())
    heading_score = heading_count / len(HEADINGS)
    bullets = len(re.findall(r'•|\*|-', resume_text))
    bullet_score = 1 if bullets >= 3 else 0.5
    return round(0.7 * heading_score + 0.3 * bullet_score, 2)

# ----------------------------
# Dynamic ATS Weights
# ----------------------------
def get_dynamic_weights(jd_text):
    jd_lines = jd_text.lower().split("\n")
    skills_lines = sum(1 for line in jd_lines if any(s in line for s in sum(SKILL_SYNONYMS.values(), [])))
    experience_lines = sum(1 for line in jd_lines if "year" in line)
    education_lines = sum(1 for line in jd_lines if any(d in line for d in sum(EDU_SYNONYMS.values(), [])))
    achievement_lines = sum(1 for line in jd_lines if re.search(r'\d+[%]?', line))
    total = skills_lines + experience_lines + education_lines + achievement_lines
    if total == 0: total = 1
    weights = {
        "skills": skills_lines / total,
        "experience": experience_lines / total,
        "education": education_lines / total,
        "achievement": achievement_lines / total,
        "formatting": 0.1
    }
    sum_w = sum(weights.values())
    for k in weights: weights[k] /= sum_w
    return weights

# ----------------------------
# Total ATS Score
# ----------------------------
def ats_total_score(resume_text, jd_text, skills_score):
    experience = ats_experience_score(resume_text, jd_text)
    education = ats_education_score(resume_text)
    achievement = ats_achievement_score(resume_text)
    formatting = ats_format_score(resume_text)
    weights = get_dynamic_weights(jd_text)
    total = round(
        weights["skills"] * skills_score +
        weights["experience"] * experience +
        weights["education"] * education +
        weights["achievement"] * achievement +
        weights["formatting"] * formatting, 2
    )
    breakdown = {"skills": skills_score, "experience": experience, "education": education,
                 "achievement": achievement, "formatting": formatting}
    return total, breakdown

# ----------------------------
# Dashboard
# ----------------------------
def create_dashboard(breakdown):
    categories = list(breakdown.keys())
    values = [v * 100 for v in breakdown.values()]
    fig = go.Figure([go.Bar(x=categories, y=values, marker_color='teal')])
    fig.update_layout(title="Resume ATS Sub-Scores", yaxis=dict(title="Score (%)", range=[0, 100]),
                      xaxis=dict(title="Factors"), template="plotly_white")
    return fig

# ----------------------------
# Optional: Project / Job Title Detection
# ----------------------------
PROJECT_KEYWORDS = ["github", "kaggle", "portfolio", "project"]
JOB_TITLE_KEYWORDS = ["engineer", "analyst", "developer", "manager", "intern"]

def detect_projects_and_titles(resume_text):
    resume_lower = resume_text.lower()
    projects = [p for p in PROJECT_KEYWORDS if p in resume_lower]
    job_titles = [j for j in JOB_TITLE_KEYWORDS if j in resume_lower]
    return projects, job_titles

# ----------------------------
# Main Analysis
# ----------------------------
def analyze_resume_and_jd(resume_file, resume_text, jd_text):
    try:
        if resume_file is not None:
            resume_content = extract_text_from_file(resume_file)
        elif resume_text:
            resume_content = resume_text
        else:
            return "❌ Please upload or paste your resume text.", None, None, None, None, None, None, None

        if not jd_text:
            return "❌ Please enter a Job Description (JD).", None, None, None, None, None, None, None

        skills_score, matched_skills, missing_skills = get_skills_score_semantic(resume_content, jd_text)
        total_score, breakdown = ats_total_score(resume_content, jd_text, skills_score)
        dashboard_fig = create_dashboard(breakdown)
        projects, job_titles = detect_projects_and_titles(resume_content)

        return (
            resume_content[:2000] + "..." if len(resume_content) > 2000 else resume_content,
            f"ATS Total Score: {total_score * 100}%",
            f"Skills Score: {skills_score * 100}%, Matched: {', '.join(matched_skills)}",
            f"Missing Skills: {', '.join(missing_skills)}",
            f"Experience: {breakdown['experience'] * 100}%, Education: {breakdown['education'] * 100}%, Achievement: {breakdown['achievement'] * 100}%, Formatting: {breakdown['formatting'] * 100}%",
            dashboard_fig,
            f"Projects Detected: {', '.join(projects) if projects else 'None'}",
            f"Job Titles Detected: {', '.join(job_titles) if job_titles else 'None'}"
        )
    except Exception as e:
        return f"⚠️ Error: {str(e)}", None, None, None, None, None, None, None

# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 HireMatch-AI Resume Fit Checker — Dynamic ATS + Dashboard")

    with gr.Row():
        with gr.Column():
            resume_file = gr.File(label="📁 Upload Resume (PDF/DOCX)")
            resume_text = gr.Textbox(lines=5, placeholder="Or paste resume text here...", label="Or paste resume text")
            jd_text = gr.Textbox(lines=5, placeholder="Paste Job Description (JD) here...", label="Job Description (JD)")
            analyze_button = gr.Button("🚀 Analyze Resume & JD")

        with gr.Column():
            resume_preview = gr.Textbox(label="Resume Preview (extracted)")
            match_score = gr.Textbox(label="ATS Total Score")
            matched_skills_box = gr.Textbox(label="Matched Skills")
            missing_skills_box = gr.Textbox(label="Missing Skills")
            other_scores = gr.Textbox(label="Experience / Education / Achievements / Formatting")
            dashboard = gr.Plot(label="ATS Sub-Score Dashboard")
            projects_box = gr.Textbox(label="Projects Detected")
            job_titles_box = gr.Textbox(label="Job Titles Detected")

    analyze_button.click(
        analyze_resume_and_jd,
        inputs=[resume_file, resume_text, jd_text],
        outputs=[resume_preview, match_score, matched_skills_box, missing_skills_box,
                 other_scores, dashboard, projects_box, job_titles_box]
    )

# ----------------------------
# ✅ Render Port Binding (Important)
# ----------------------------
# ----------------------------
# ✅ Render Port Binding (Final Fix)
# ----------------------------
# ----------------------------
# ✅ Render Port Binding — Final Fix (tested)
# ----------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting app on port {port} (Render expected port).")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inline=False,
        show_error=True,
        debug=True
    )
