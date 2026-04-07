# app.py
import os
import uuid
import numpy as np
import pandas as pd
import torch
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash
)
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

# -------- CONFIG --------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {'pdf'}
USERS_CSV = "users.csv"
OFFERS_CSV = "ALL_Offers.csv"
EMBED_CACHE = "job_embeddings.npy"
SECRET_KEY = "replace_with_secure_key"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

# -------- LOAD JOBS --------
if not os.path.exists(OFFERS_CSV):
    raise FileNotFoundError(f"Missing {OFFERS_CSV} in project folder.")
jobs_df = pd.read_csv(OFFERS_CSV).fillna("").reset_index(drop=True)

# normalize columns
if 'company_name' not in jobs_df.columns and 'organization' in jobs_df.columns:
    jobs_df['company_name'] = jobs_df['organization']
if 'location' not in jobs_df.columns and 'city' in jobs_df.columns:
    jobs_df['location'] = jobs_df['city']

# -------- USERS CSV INIT --------
if not os.path.exists(USERS_CSV):
    pd.DataFrame(columns=['id','email','username','firstname','lastname','resume_path','skills','applied_jobs']).to_csv(USERS_CSV, index=False)

# -------- MODEL & EMBEDDINGS --------
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

def build_job_texts(df):
    return (df['job_title'].astype(str) + ". " + df['description'].astype(str) + " Skills: " + df['skills'].astype(str)).tolist()

if os.path.exists(EMBED_CACHE):
    job_embeddings = torch.tensor(np.load(EMBED_CACHE))
else:
    job_texts = build_job_texts(jobs_df)
    job_embeddings = model.encode(job_texts, convert_to_tensor=True)
    np.save(EMBED_CACHE, job_embeddings.cpu().numpy())

# -------- SKILL HELPERS --------
MASTER_SKILLS = [
    'python','sql','excel','machine learning','data analysis','pandas','numpy',
    'tensorflow','keras','pytorch','flask','django','html','css','javascript',
    'react','angular','aws','azure','gcp','git','docker','kubernetes','tableau',
    'power bi','r','java','c++','node','devops','spring boot'
]

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        return ""

def extract_skills_from_text(text):
    t = str(text).lower()
    found = []
    for s in MASTER_SKILLS:
        if s in t:
            found.append(s)
    return list(dict.fromkeys(found))

def extract_skills_from_any_text(text):
    """Robust skill extraction for messy job 'skills' strings or descriptions."""
    t = str(text).lower()
    mapping = {
        "python": ["python"],
        "sql": ["sql","mysql","postgres","sqlite"],
        "html": ["html"],
        "css": ["css"],
        "javascript": ["javascript","js"],
        "react": ["react","reactjs"],
        "angular": ["angular","angularjs"],
        "node": ["node","nodejs"],
        "java": ["java"],
        "spring boot": ["spring boot","springboot"],
        "aws": ["aws","amazon web services"],
        "azure": ["azure"],
        "gcp": ["gcp","google cloud"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes","k8s"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch"],
        "machine learning": ["machine learning","ml"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "devops": ["devops"],
    }
    found = []
    for skill, keys in mapping.items():
        for k in keys:
            if k in t:
                found.append(skill)
    return list(dict.fromkeys(found))

def normalize_skills_str(skills_str):
    # If already comma-separated, parse; else extract keywords
    s = str(skills_str).strip()
    if not s:
        return []
    if "," in s or "|" in s or "/" in s:
        sep = "," if "," in s else ("|" if "|" in s else "/")
        parts = [p.strip().lower() for p in s.split(sep) if p.strip()]
        # normalize tokens via mapping extractor
        out = []
        for p in parts:
            out += extract_skills_from_any_text(p)
        return list(dict.fromkeys(out))
    else:
        return extract_skills_from_any_text(s)

def compute_missing_skills(user_skills_list, job_skills_str):
    job_sk = normalize_skills_str(job_skills_str)
    missing = [s for s in job_sk if s not in user_skills_list]
    return missing

# -------- USER CSV HELPERS --------
def load_users_df():
    return pd.read_csv(USERS_CSV, dtype={"applied_jobs": "object"}).fillna("")


def save_users_df(df):
    df.to_csv(USERS_CSV, index=False)

def get_user_by_username(username):
    df = load_users_df()
    if df.empty:
        return None
    df['username'] = df['username'].astype(str).str.strip()
    username = str(username).strip()
    row = df[df['username'].str.lower() == username.lower()]
    return None if row.empty else row.iloc[0].to_dict()

def add_user_and_save(email, username, firstname, lastname, resume_path, skills_list):
    df = load_users_df()
    new_id = str(uuid.uuid4())
    skills_str = ', '.join(skills_list)
    new_row = {'id':new_id,'email':email,'username':username,'firstname':firstname,'lastname':lastname,'resume_path':resume_path,'skills':skills_str,'applied_jobs':''}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users_df(df)
    return new_id

def add_applied_job_to_user(username, job_idx):
    df = load_users_df()
    idx = df.index[df['username'].astype(str).str.lower() == username.lower()]
    if len(idx) == 0:
        return False
    i = idx[0]
    existing = df.at[i, 'applied_jobs']
    if not isinstance(existing, str):
        existing = ""
    applied = [int(x) for x in str(existing).split(',') if str(x).strip().isdigit()]
    if job_idx in applied:
        return True
    applied.append(job_idx)
    df.at[i, 'applied_jobs'] = ','.join(map(str, applied))
    save_users_df(df)
    return True

# -------- RECOMMENDERS --------
def compute_recommendations(user_skill_text, top_k=8, strict=True):
    user_skills_list = normalize_skills_str(user_skill_text)
    if not user_skills_list:
        return []
    user_emb = model.encode(user_skill_text, convert_to_tensor=True)
    sims = util.cos_sim(user_emb, job_embeddings)[0]
    topk = torch.topk(sims, k=min(len(jobs_df), top_k * 8))
    recs = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        row = jobs_df.iloc[int(idx)]
        missing = compute_missing_skills(user_skills_list, row.get('skills',''))
        if strict and len(missing) > 0:
            continue
        recs.append({
            'idx': int(idx),
            'title': row.get('job_title',''),
            'company': row.get('company_name', row.get('organization','')),
            'location': row.get('location',''),
            'salary': row.get('salary','N/A'),
            'skills': row.get('skills',''),
            'description': row.get('description',''),
            'score': float(score)
        })
        if len(recs) >= top_k:
            break
    return recs

def compute_semantic_top_jobs(user_skill_text, top_k=30):
    if not user_skill_text.strip():
        return []
    user_emb = model.encode(user_skill_text, convert_to_tensor=True)
    sims = util.cos_sim(user_emb, job_embeddings)[0]
    topk = torch.topk(sims, k=min(top_k, len(jobs_df)))
    results = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        row = jobs_df.iloc[int(idx)]
        results.append({
            'idx': int(idx),
            'skills': row.get('skills',''),
            'title': row.get('job_title',''),
            'score': float(score)
        })
    return results

# -------- ROUTES --------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email','').strip()
        username = request.form.get('username','').strip()
        firstname = request.form.get('firstname','').strip()
        lastname = request.form.get('lastname','').strip()
        manual_skills_raw = request.form.get('manual_skills','').strip()
        resume = request.files.get('resume', None)

        if not username or not email:
            flash("Provide username and email."); return redirect(url_for('signup'))

        resume_path = ''
        skills_found = []
        if resume and '.' in resume.filename and resume.filename.rsplit('.',1)[1].lower() in ALLOWED_EXT:
            filename = secure_filename(username + "_" + resume.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume.save(save_path)
            resume_path = save_path
            text = extract_text_from_pdf(save_path)
            skills_found = extract_skills_from_text(text)
        elif manual_skills_raw:
            skills_found = normalize_skills_str(manual_skills_raw)

        if not skills_found:
            flash("Upload resume or enter skills (comma separated)."); return redirect(url_for('signup'))

        add_user_and_save(email, username, firstname, lastname, resume_path, skills_found)
        flash("Signup successful. Please sign in.")
        return redirect(url_for('signin'))

    return render_template('signup.html')

@app.route('/signin', methods=['GET','POST'])
def signin():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        user = get_user_by_username(username)
        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']   # important for apply
            return redirect(url_for('dashboard'))
        else:
            flash("User not found. Please sign up.")
            return redirect(url_for('signup'))
    return render_template('signin.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('signin'))

    user = get_user_by_username(session['username'])
    if not user:
        flash("User missing.")
        return redirect(url_for('signin'))

    # Normalize user skills
    user_skills = normalize_skills_str(user.get('skills', ''))
    skills_text = ', '.join(user_skills)

    # Compute recommendations
    recs = compute_recommendations(skills_text, top_k=8, strict=True)
    semantic = compute_semantic_top_jobs(skills_text, top_k=30)

    # Compute missing skills for learning resources
    missing_union = set()
    for job in semantic:
        for m in compute_missing_skills(user_skills, job.get('skills', '')):
            if m and m.strip():
                missing_union.add(m.strip())

    resources = [
        {
            'skill': s,
            'youtube': f'https://www.youtube.com/results?search_query={s.replace(" ", "+")}+tutorial'
        } for s in sorted(list(missing_union))
    ]

    # Safely parse applied jobs
    raw_applied = user.get('applied_jobs', '')
    if raw_applied is None or pd.isna(raw_applied):
        raw_applied = ''
    applied_jobs = [x.strip() for x in str(raw_applied).split(',') if x.strip().isdigit()]

    return render_template(
        'dashboard.html',
        user=user,
        recs=recs,
        resources=resources,
        applied_jobs=applied_jobs
    )

@app.route('/job/<int:idx>')
def job_detail(idx):
    if idx < 0 or idx >= len(jobs_df):
        return jsonify({'error':'invalid index'}), 404
    row = jobs_df.iloc[int(idx)]
    user_skills = []
    if 'username' in session:
        user = get_user_by_username(session['username'])
        if user:
            user_skills = normalize_skills_str(user.get('skills',''))
    missing = compute_missing_skills(user_skills, row.get('skills',''))
    return jsonify({
        'title': row.get('job_title',''),
        'company': row.get('company_name', row.get('organization','')),
        'location': row.get('location',''),
        'salary': row.get('salary','N/A'),
        'skills': row.get('skills',''),
        'description': row.get('description',''),
        'missing_skills': missing
    })

from flask import jsonify, session

from flask import jsonify, request
from flask_login import current_user, login_required

@app.route('/apply/<int:idx>', methods=['POST'])
def apply_job(idx):
    if 'username' not in session:
        return jsonify({'error': 'not_logged_in'}), 403

    username = session['username'].lower()

    # Load CSV safe
    try:
        df = pd.read_csv(USERS_CSV).fillna("")
    except Exception as e:
        print("CSV LOAD ERROR:", e)
        return jsonify({"error": "csv_load_failed"}), 500

    # Find user
    df['username'] = df['username'].astype(str).str.lower()
    user_row = df[df['username'] == username]

    if user_row.empty:
        return jsonify({"error": "user_not_found"}), 404

    i = user_row.index[0]

    # --- FIX: READ APPLIED_JOBS SAFELY ---
    raw = str(df.at[i, "applied_jobs"]).strip()

    if raw == "" or raw.lower() == "nan":
        applied = []
    else:
        applied = [int(x) for x in raw.split(",") if x.strip().isdigit()]

    # --- ADD NEW JOB ---
    if idx not in applied:
        applied.append(idx)

    df.at[i, "applied_jobs"] = ",".join(map(str, applied))

    # --- SAVE CSV SAFELY ---
    try:
        df.to_csv(USERS_CSV, index=False)
    except Exception as e:
        print("CSV SAVE ERROR:", e)
        return jsonify({"error": "csv_save_failed"}), 500

    return jsonify({"ok": True})



@app.route("/applied_jobs")
def applied_jobs():
    if "username" not in session:
        return redirect(url_for('signin'))

    username = session['username'].strip().lower()
    users_df = pd.read_csv(USERS_CSV).fillna("")
    jobs_df = pd.read_csv(OFFERS_CSV).fillna("")

    # Normalize jobs_df columns
    if 'company_name' not in jobs_df.columns and 'organization' in jobs_df.columns:
        jobs_df['company_name'] = jobs_df['organization']
    if 'location' not in jobs_df.columns and 'city' in jobs_df.columns:
        jobs_df['location'] = jobs_df['city']

    # Find user
    user_mask = users_df['username'].astype(str).str.lower() == username
    if not user_mask.any():
        flash("User not found.")
        return redirect(url_for('signin'))

    user_row = users_df.loc[user_mask].iloc[0]
    raw_applied = user_row.get('applied_jobs', '')
    job_ids = [int(x) for x in str(raw_applied).split(',') if x.strip().isdigit()]

    jobs = []
    for jid in job_ids:
        if 0 <= jid < len(jobs_df):
            row = jobs_df.iloc[jid]
            jobs.append({
                "idx": jid,
                "title": row.get('job_title', ''),
                "company": row.get('company_name', ''),
                "location": row.get('location', ''),
                "salary": row.get('salary', 'N/A')
            })

    return render_template("applied_jobs.html", jobs=jobs)

@app.route("/admin")
def admin_panel():
    df_users = pd.read_csv(USERS_CSV).fillna("")
    df_jobs = pd.read_csv(OFFERS_CSV).fillna("")

    # Normalize columns so the admin template always gets valid fields
    if 'company_name' not in df_jobs.columns and 'organization' in df_jobs.columns:
        df_jobs['company_name'] = df_jobs['organization']

    if 'location' not in df_jobs.columns and 'city' in df_jobs.columns:
        df_jobs['location'] = df_jobs['city']

    return render_template(
        "admin.html",
        users=df_users.to_dict(orient="records"),
        jobs=df_jobs.to_dict(orient="records")
    )

@app.route('/profile', methods=['GET','POST'])
def profile():
    if 'username' not in session:
        return redirect(url_for('signin'))
    user = get_user_by_username(session['username'])
    if request.method == 'POST':
        firstname = request.form.get('firstname','').strip()
        lastname = request.form.get('lastname','').strip()
        skills_raw = request.form.get('skills','').strip()
        skills_list = normalize_skills_str(skills_raw)
        resume = request.files.get('resume')
        resume_path = user.get('resume_path','')
        if resume and '.' in resume.filename and resume.filename.rsplit('.',1)[1].lower() in ALLOWED_EXT:
            filename = secure_filename(user['username'] + "_updated.pdf")
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            resume.save(save_path)
            resume_path = save_path
        df = load_users_df()
        df.loc[df['id'] == user['id'], ['firstname','lastname','skills','resume_path']] = [firstname, lastname, ', '.join(skills_list), resume_path]
        save_users_df(df)
        flash("Profile updated.")
        return redirect(url_for('profile'))
    return render_template('profile.html', user=user)

@app.route('/signout')
def signout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
