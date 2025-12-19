from flask import Flask, render_template, request, flash, redirect, url_for
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from werkzeug.utils import secure_filename
from career_data import career_info

app = Flask(__name__)
app.secret_key = "secret_key_for_flash_messages"  # Needed for flashing error messages

os.makedirs("model", exist_ok=True)
os.makedirs("uploads", exist_ok=True)  # ✅ Folder for resume uploads

model_path = "model/recommender.pkl"

# ✅ Train model if not present
if not os.path.exists(model_path):
    df = pd.read_csv("data/careers.csv")
    X = df['text']
    y = df['career']

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    with open(model_path, "wb") as f:
        pickle.dump((model, vectorizer), f)
else:
    with open(model_path, "rb") as f:
        model, vectorizer = pickle.load(f)

# Global variable to store top 3 predictions
global_top3 = []

# ----------------------------------------------------
# ✅ Existing Features
# ----------------------------------------------------

@app.route('/route', methods=['POST'])
def route_page():
    selected_career = request.form['selected_career']
    info = career_info.get(selected_career, {
        "skills": ["No data available"],
        "roadmap": [{"title": "No roadmap", "desc": "No details available"}],
        "salary": "N/A"
    })

    return render_template('result.html', career=selected_career, info=info, top3=global_top3)


@app.route('/predict', methods=['POST'])
def predict():
    skill = request.form.get('skill_input', '').strip()
    interest = request.form.get('interest_input', '').strip()
    education = request.form.get('education_input', '').strip()

    # ✅ Validation — check for empty fields
    if not skill or not interest or not education:
        flash("⚠️ Please fill in all fields before submitting.")
        return redirect(url_for('index'))

    user_input = f"{skill} {interest} {education}"
    user_vec = vectorizer.transform([user_input])

    # ✅ Get top 3 predictions with confidence
    proba = model.predict_proba(user_vec)[0]
    classes = model.classes_
    top3_idx = proba.argsort()[-3:][::-1]
    top3_predictions = [(classes[i], round(proba[i] * 100, 2)) for i in top3_idx]

    # ✅ Roadmap for top prediction
    top_career = top3_predictions[0][0]
    info = career_info.get(top_career, {
        "skills": ["No data available"],
        "roadmap": [{"title": "No roadmap", "desc": "No details available"}],
        "salary": "N/A"
    })

    # ✅ Save top 3 globally so route_page can use it
    global global_top3
    global_top3 = top3_predictions

    return render_template(
        'result.html',
        top3=top3_predictions,
        career=top_career,
        info=info
    )

# ----------------------------------------------------
# ✨ NEW FEATURE 1: Smart Resume Upload
# ----------------------------------------------------

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        flash("⚠️ No file part.")
        return redirect(url_for('index'))

    file = request.files['resume']
    if file.filename == '':
        flash("⚠️ No selected file.")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # 🧠 Extract text content
        text_content = extract_text_from_resume(filepath)

        # 🔮 Predict career from resume
        user_vec = vectorizer.transform([text_content])
        proba = model.predict_proba(user_vec)[0]
        classes = model.classes_
        top3_idx = proba.argsort()[-3:][::-1]
        top3_predictions = [(classes[i], round(proba[i] * 100, 2)) for i in top3_idx]

        top_career = top3_predictions[0][0]
        info = career_info.get(top_career, {
            "skills": ["No data available"],
            "roadmap": [{"title": "No roadmap", "desc": "No details available"}],
            "salary": "N/A"
        })

        global global_top3
        global_top3 = top3_predictions

        return render_template(
            'result.html',
            top3=top3_predictions,
            career=top_career,
            info=info
        )
    else:
        flash("❌ Invalid file format. Please upload a PDF or TXT file.")
        return redirect(url_for('index'))

def extract_text_from_resume(filepath):
    """Basic text extraction — can be improved with NLP later."""
    text = ""
    if filepath.lower().endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif filepath.lower().endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print("PDF Read Error:", e)
    return text.strip()


# ----------------------------------------------------
# ✨ NEW FEATURE 2: Skill Gap Analyzer
# ----------------------------------------------------
@app.route('/analyze_skills', methods=['POST'])
def analyze_skills():
    entered_skills = request.form.get('user_skills', '').strip().lower().split(',')
    selected_career = request.form.get('career_for_analysis', '')

    if not entered_skills or not selected_career:
        flash("⚠️ Please provide your skills and select a career.")
        return redirect(url_for('index'))

    career_data = career_info.get(selected_career, {})
    required_skills = [s.lower() for s in career_data.get('skills', [])]

    # 🔸 Find missing skills
    missing_skills = [skill for skill in required_skills if skill not in entered_skills]

    return render_template(
        'skill_gap.html',
        selected_career=selected_career,
        user_skills=entered_skills,
        required_skills=required_skills,
        missing_skills=missing_skills
    )

# ----------------------------------------------------
# 🏠 Home
# ----------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)