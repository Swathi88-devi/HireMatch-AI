******HireMatch-AI-Resume Fit Analyzer*****
**Overview**
This project is a Resume Fit Checker that simulates real ATS (Applicant Tracking System) scoring.
It allows candidates to upload PDF/DOCX resumes or paste text, compares it against a Job Description (JD), and outputs a dynamic ATS score with a visual dashboard.
This tool is portfolio-ready, interactive, and free for public use.
**1) Features**
Dynamic ATS Scoring
Scores resumes based on:
Skills (semantic + synonym matching)
Experience (years relevant to JD)
Education (degrees + certifications)
Achievements / Metrics
Formatting / Structure
Uses JD-adaptive weights to mimic real ATS scoring.
Semantic Skill Matching
Detects skills not only from a fixed list but also semantically similar phrases in JD and resume.
Example: “Power BI dashboards” matches “data visualization” in JD.
**Dashboard Visualization**
Interactive bar chart shows sub-scores for:
Skills
Experience
Education
Achievements
Formatting
Helps users quickly identify strong and weak areas.
Project & Job Title Detection (Optional)
Detects GitHub, Kaggle, or portfolio mentions.
Detects relevant job titles like Engineer, Analyst, Developer, etc.
Supports PDF, DOCX, or Pasted Text
Automatic text extraction from resumes.
Preprocessing with NLP techniques.
**2) Demo Instructions**
Go to the Hugging Face Space:
Your Space URL
Upload your resume (PDF or DOCX) or paste text directly.
Paste the Job Description (JD) you want to compare.
Click “Analyze Resume & JD”.
**View results:**
Resume Preview — extracted text.
ATS Total Score — overall resume fit.
Matched / Missing Skills
Experience, Education, Achievements, Formatting
Dashboard — visual breakdown of all factors.
Detected Projects / Job Titles
**3)Example Output**
Factor	Score (%)
Skills :80
Experience :	60
Education :	100
Achievements :	50
Formatting :	90
Total ATS Score: 76%
Matched Skills: Python, ML, SQL
Missing Skills: TensorFlow, React
**How it Works**
Extract Resume Text → PDF/DOCX parsing or pasted text.
Preprocess Text → Lowercase, remove stopwords, tokenize.
Semantic Skill Matching → Uses Sentence-BERT embeddings to detect skills and phrases.
Sub-Scores Calculation → Experience, Education, Achievements, Formatting.
Dynamic Weighting → Weights derived from JD emphasis; total ATS score calculated.
Dashboard Visualization → Interactive bar chart for sub-scores.
**Caveats / Limitations**
Not fully ML-based — scoring is heuristic + semantic similarity, does not learn from historical hiring data.
JD quality dependent — vague or poorly written JDs may affect scoring.
Formatting limitations — scanned images, tables, or unusual layouts may not parse correctly.
Soft skills / cultural fit — cannot be scored automatically.
ATS proprietary algorithms — real ATS may use ML ranking or boolean filters that cannot be fully replicated.
Privacy Note: Uploaded resumes are processed temporarily and not stored.
**Installation / Local Run (Optional)**
Clone repository:
git clone https://github.com/yourusername/HireMatch-AI.git
cd ai-ats-resume-checker
Install dependencies:
pip install -r requirements.txt
**Run the app**:
python app.py
Open local link to test.
**Tech Stack**
Python: Core language
Gradio: Interactive UI
NLTK: Text preprocessing
pdfplumber / python-docx: Resume text extraction
Sentence-Transformers (BERT embeddings): Semantic similarity
Plotly: Dashboard visualization
**Next Steps / Enhancements**
Integrate ML-based scoring trained on historical resumes for more realistic ATS behavior.
Add resume formatting suggestions for improvement.
Detect industry-specific keywords to refine scoring.
Optional cover letter analysis.
**Notes for Users**
Best used for resume self-assessment and preparing for ATS-friendly resumes.
Not a guarantee for real ATS pass; simulates ATS logic using available data.
Free and public — anyone can use the tool via Hugging Face Space link.
