# AI CareerPath — IBM SkillsBuild Final Project

AI CareerPath is an explainable ML-assisted career recommendation system for students and early job seekers. It accepts education, skills, interests, experience/projects and an optional PDF/TXT resume, then produces a ranked list of career roles, skill gaps, a learning roadmap and project recommendations.

## What makes this a real ML recommendation system?

The recommendation is not a single keyword score. It uses three independent signals:

1. **Supervised ML prediction** — a Logistic Regression classifier trained on TF-IDF word and character features.
2. **Semantic similarity** — cosine similarity between the profile and curated role descriptions.
3. **Weighted skill coverage** — transparent role-skill weights used to explain matched and missing skills.

The final ranking is a weighted hybrid of those signals. The target role is deliberately excluded from the ML input to avoid label leakage.

### Model evaluation

The included academic benchmark contains **776 synthetic/curated student-profile examples** across four roles. On a stratified 20% hold-out test split, the current model achieved **96.8% accuracy and 96.8% macro F1**. These are internal academic evaluation metrics, not evidence of production-level accuracy or employment outcomes. The dataset is intentionally labelled as synthetic/academic rather than real job-market data.

## Roles

- AI Engineer
- Machine Learning Engineer
- Data Scientist
- Full Stack Developer

The role taxonomy and skill vocabulary were informed by current O*NET occupation/technology information and ESCO occupation-skill concepts. O*NET provides occupation-specific technology skill information and notes that software/job-posting data are updated over time; ESCO provides a structured skills-occupations classification.

## Features

- Student profile collection
- Optional PDF/TXT resume extraction
- ML career-role ranking
- Hybrid recommendation score
- ML probability shown separately from the final recommendation score
- Explainable evidence / matched skills
- Priority skill gaps
- Personalized learning roadmap
- Project recommendations
- Profile completeness indicator
- Model audit/evaluation section
- Input validation and clear error handling

## Architecture

**Student → Web UI → Flask API → Resume/Text Processing → ML Classifier + Semantic Matcher + Skill Knowledge Layer → Hybrid Ranking → Skill Gaps → Roadmap & Projects**

## Run on Windows

Use **Python 3.10 or 3.11** for the most predictable scikit-learn environment. (The supplied model was trained with scikit-learn 1.8.0.)

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open:

`http://127.0.0.1:5000`

There is also a `run_windows.bat` file that automates these steps.

## Suggested demo profile

**Name:** Harsha Vardhan

**Education:** B.Tech Computer Science

**Skills:** Python, NumPy, Pandas, SQL, Machine Learning, PyTorch, NLP, FastAPI, Git

**Interests:** Artificial Intelligence, LLM applications, Machine Learning

**Experience / Projects:** Built an NLP-based student project, trained a prediction model and deployed a Flask/FastAPI endpoint.

**Target:** Let AI Recommend

The system should strongly rank an AI/ML role while still showing the full ranking and skill gaps.

## API

`GET /api/health` — service/model status.

`GET /api/model-info` — evaluation and model metadata.

`POST /api/analyze` — multipart form or JSON profile analysis. Supported upload types: PDF and TXT, maximum 5 MB.

## Important submission note

The model is a **student-project prototype**. Do not describe the benchmark as real labor-market data and do not claim that the system predicts hiring decisions. Career recommendations are guidance only.

## External references used to inform the role/skill knowledge layer

- O*NET OnLine — Data Scientists: current occupation summary and technology/software skills.
- O*NET OnLine — Software Developers / Web Developers: current technology skill and in-demand software listings.
- European Commission ESCO — Skills-Occupations Matrix and Software Developer concepts.

The external sources inform the ontology/skill vocabulary; the model training profiles included in this repository are academic/synthetic examples.
