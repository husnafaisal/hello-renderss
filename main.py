from flask import Flask, request, render_template, redirect, url_for
import os
import pdfplumber
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text as sk_text
import json
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


# --- TEXT EXTRACTION ---
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"❌ Error reading PDF {file_path}: {e}")
        return ""


def extract_text_from_docx(file_path):
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"❌ Error reading DOCX {file_path}: {e}")
        return ""


def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"❌ Error reading TXT {file_path}: {e}")
        return ""


def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        return ""


# --- TEXT NORMALIZATION ---
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r'[\-\*•–—:]', ' ', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- CLEANUP FUNCTION ---
def cleanup_uploads(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'⚠️ Failed to delete {file_path}. Reason: {e}')


# --- CONFIDENCE TIERS ---
def get_confidence_tier(score):
    if score >= 80:
        return {'label': 'Tier 1 (Excellent Match)', 'style': 'bg-success'}
    elif score >= 60:
        return {'label': 'Tier 2 (Strong Fit)', 'style': 'bg-primary'}
    elif score >= 40:
        return {'label': 'Tier 3 (Average Fit)', 'style': 'bg-warning text-dark'}
    else:
        return {'label': 'Tier 4 (Low Fit)', 'style': 'bg-danger'}


# --- ROUTES ---
@app.route("/")
def index():
    cleanup_uploads(app.config['UPLOAD_FOLDER'])
    return render_template('matchresume.html')


@app.route("/matcher", methods=['POST'])
def matcher():
    print("✅ /matcher route triggered!")

    job_description = request.form.get('job_description')
    resume_files = request.files.getlist('resume_file')

    if not job_description:
        return render_template('matchresume.html', error_message="Please paste a job description to begin matching.")
    if not resume_files or not any(f.filename for f in resume_files):
        return render_template('matchresume.html', error_message="Please upload at least one resume file.")

    resumes_text = []
    uploaded_names = []

    for resume_file in resume_files:
        if resume_file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(file_path)
            uploaded_names.append(resume_file.filename)

            extracted = extract_text(file_path)
            cleaned = normalize_text(extracted)
            resumes_text.append(cleaned)

    normalized_jd = normalize_text(job_description)

    custom_words = ['skills', 'experience', 'candidate', 'job', 'responsibilities']
    combined_stopwords = list(sk_text.ENGLISH_STOP_WORDS.union(custom_words))

    all_docs = [normalized_jd] + resumes_text
    vectorizer = TfidfVectorizer(stop_words=combined_stopwords)
    vectors = vectorizer.fit_transform(all_docs)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    results = []
    for i, sim in enumerate(similarities):
        score = round(sim * 100, 2)
        tier = get_confidence_tier(score)
        results.append({
            'name': uploaded_names[i],
            'score': score,
            'tier_label': tier['label'],
            'tier_style': tier['style']
        })

    results.sort(key=lambda x: x['score'], reverse=True)

    chart_data = [{'name': r['name'], 'score': r['score']} for r in results]
    chart_data_json = json.dumps(chart_data)

    return render_template(
        'matchresume.html',
        message="Analysis Complete. Top Matches:",
        results=results[:5],
        chart_data_json=chart_data_json
    )


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    print("🚀 Flask Resume Matcher Running at: http://127.0.0.1:5000")
    app.run(debug=True)
