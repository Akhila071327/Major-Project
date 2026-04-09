# 🚀 Career Recommendation System (LLM-Based)

## 📌 Overview

The **Career Recommendation System** is a web-based application that suggests relevant job roles based on user skills. Users can upload their resume or manually enter skills, and the system recommends jobs using NLP embeddings.

---

## 🎯 Features

* 👤 User Signup & Login
* 📄 Resume Upload (PDF)
* ✍️ Manual Skill Entry
* 🤖 AI-based Job Recommendations
* 📊 Skill Gap Analysis
* 🎥 Learning Resource Suggestions (YouTube)
* 🔍 Job Filtering & Apply Option
* 🧑 Profile Management

---

## 🛠️ Tech Stack

* **Backend:** Python (Flask)
* **Frontend:** HTML, CSS, Bootstrap
* **ML/NLP:** Sentence Transformers
* **Database:** CSV (users.csv, ALL_Offers.csv)
* **Libraries:** Pandas, NumPy, PyPDF2, Torch

---

## 📂 Project Structure

```
project/
│── app.py
│── users.csv
│── ALL_Offers.csv
│── offers.csv
│── organizations.csv
│── job_embeddings.npy (auto-generated)
│── uploads/
│── templates/
│   ├── base.html
│   ├── index.html
│   ├── signup.html
│   ├── signin.html
│   ├── dashboard.html
│   ├── profile.html
│   ├── edit_skills.html
│   ├── admin_dashboard.html
```
---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/career-recommendation-system.git
cd career-recommendation-system
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
```

---

### 3️⃣ Install Required Libraries

```bash
pip install flask pandas numpy scikit-learn sentence-transformers torch PyPDF2
```

---

### 4️⃣ Fix Transformers Compatibility (IMPORTANT)

```bash
pip install tf-keras
```

---

### 5️⃣ Run the Application

```bash
python app.py
```

---

### 6️⃣ Open in Browser

```
http://127.0.0.1:5000/
```

---

## 📌 Important Notes

* First run may take time ⏳ (embedding generation)
* `job_embeddings.npy` will be created automatically
* Ensure `ALL_Offers.csv` exists in project folder
* Resume must be in **PDF format**

---

## 🧠 How It Works

1. User uploads resume or enters skills
2. Skills are extracted using NLP
3. Text is converted into embeddings
4. Compared with job dataset
5. Top matching jobs are recommended

---

## 🔍 Example Workflow

```
Resume → Skill Extraction → Embeddings → Similarity → Job Recommendations
```

---

## ⚠️ Known Issues

* First login may be slow due to embedding computation
* CSV-based storage (not scalable for production)
* Resume parsing depends on PDF quality

---

## 🚀 Future Improvements

* Integrate real-time job APIs
* Use advanced LLMs (GPT)
* Deploy to cloud
* Add chatbot assistant

---

## 👩‍💻 Author

Developed as a **Final Year Major Project**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
