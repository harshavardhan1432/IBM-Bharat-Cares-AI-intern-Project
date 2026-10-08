from __future__ import annotations
from pathlib import Path
import re
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "career_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

ROLE_SKILLS = {
    "AI Engineer": {
        "python": 10, "machinelearning": 10, "deeplearning": 9, "pytorch": 8, "tensorflow": 8,
        "numpy": 7, "pandas": 7, "nlp": 8, "llm": 8, "modeldeployment": 7,
        "restapi": 5, "flask": 5, "fastapi": 6, "sql": 4, "git": 4, "docker": 5,
        "transformers": 6, "computervision": 5, "cloud": 4, "scikitlearn": 5
    },
    "Machine Learning Engineer": {
        "python": 10, "machinelearning": 10, "scikitlearn": 9, "featureengineering": 9,
        "modelevaluation": 8, "numpy": 7, "pandas": 7, "mlpipeline": 8,
        "modeldeployment": 7, "pytorch": 7, "tensorflow": 7, "sql": 5, "git": 5,
        "docker": 6, "cloud": 5, "restapi": 5, "flask": 4, "fastapi": 4,
        "mlflow": 5, "kubernetes": 4, "datapreprocessing": 7
    },
    "Data Scientist": {
        "python": 10, "pandas": 10, "sql": 9, "statistics": 9, "numpy": 7,
        "datavisualization": 8, "eda": 8, "scikitlearn": 7, "machinelearning": 7,
        "experimentation": 7, "tableau": 5, "powerbi": 5, "matplotlib": 5,
        "seaborn": 5, "hypothesistesting": 7, "datacleaning": 7, "abtesting": 5,
        "r": 4, "spark": 4, "git": 3
    },
    "Full Stack Developer": {
        "html": 8, "css": 8, "javascript": 10, "react": 9, "node": 8,
        "express": 8, "mongodb": 7, "sql": 6, "restapi": 7, "git": 5,
        "typescript": 6, "nextjs": 5, "authentication": 6, "crud": 5, "jwt": 5,
        "docker": 4, "aws": 4, "github": 4, "postgresql": 4, "graphql": 4,
        "responsivedesign": 5
    }
}

SKILL_LABELS = {
    "python": "Python", "machinelearning": "Machine Learning", "deeplearning": "Deep Learning",
    "pytorch": "PyTorch", "tensorflow": "TensorFlow", "numpy": "NumPy", "pandas": "Pandas",
    "nlp": "NLP", "llm": "LLMs", "modeldeployment": "Model Deployment", "restapi": "REST API",
    "flask": "Flask", "fastapi": "FastAPI", "sql": "SQL", "git": "Git", "docker": "Docker",
    "transformers": "Transformers", "computervision": "Computer Vision", "cloud": "Cloud",
    "scikitlearn": "Scikit-learn", "featureengineering": "Feature Engineering", "modelevaluation": "Model Evaluation",
    "mlpipeline": "ML Pipelines", "mlflow": "MLflow", "kubernetes": "Kubernetes", "datapreprocessing": "Data Preprocessing",
    "statistics": "Statistics", "datavisualization": "Data Visualization", "eda": "Exploratory Data Analysis",
    "experimentation": "Experimentation", "tableau": "Tableau", "powerbi": "Power BI", "matplotlib": "Matplotlib",
    "seaborn": "Seaborn", "hypothesistesting": "Hypothesis Testing", "datacleaning": "Data Cleaning",
    "abtesting": "A/B Testing", "r": "R", "spark": "Apache Spark", "html": "HTML", "css": "CSS",
    "javascript": "JavaScript", "react": "React", "node": "Node.js", "express": "Express.js",
    "mongodb": "MongoDB", "typescript": "TypeScript", "nextjs": "Next.js", "authentication": "Authentication",
    "crud": "CRUD", "jwt": "JWT", "aws": "AWS", "github": "GitHub", "postgresql": "PostgreSQL",
    "graphql": "GraphQL", "responsivedesign": "Responsive Design"
}

# Regex patterns support common abbreviations/aliases without substring false positives.
PATTERN_TEXT = {
    "python": r"\bpython(?:\s+programming)?\b",
    "machinelearning": r"\b(?:machine[\s-]?learning|\bml\b)\b",
    "deeplearning": r"\bdeep[\s-]?learning\b",
    "pytorch": r"\bpytorch\b",
    "tensorflow": r"\btensorflow\b",
    "numpy": r"\bnumpy\b",
    "pandas": r"\bpandas\b",
    "nlp": r"\b(?:nlp|natural language processing)\b",
    "llm": r"\b(?:llm|llms|large language model|large language models)\b",
    "modeldeployment": r"\b(?:model deployment|model serving|serving models|deploy(?:ed|ing)? models)\b",
    "restapi": r"\b(?:rest\s*api|restful\s*api|apis?|api development|api integration)\b",
    "flask": r"\bflask\b",
    "fastapi": r"\bfastapi\b",
    "sql": r"\bsql\b",
    "git": r"\bgit\b",
    "docker": r"\bdocker\b",
    "transformers": r"\btransformers?\b",
    "computervision": r"\b(?:computer vision|opencv)\b",
    "cloud": r"\b(?:cloud|aws|azure|gcp)\b",
    "scikitlearn": r"\b(?:scikit[\s-]?learn|sklearn)\b",
    "featureengineering": r"\bfeature[\s-]?engineering\b",
    "modelevaluation": r"\b(?:model evaluation|model metrics|precision|recall|f1 score|roc auc)\b",
    "mlpipeline": r"\b(?:ml pipeline|ml pipelines|machine learning pipeline|machine learning pipelines)\b",
    "mlflow": r"\bmlflow\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "datapreprocessing": r"\b(?:data preprocessing|preprocessing data|pre-processing)\b",
    "statistics": r"\bstatistics?\b",
    "datavisualization": r"\b(?:data visuali[sz]ation|data visualization|visualization)\b",
    "eda": r"\b(?:eda|exploratory data analysis)\b",
    "experimentation": r"\b(?:experimentation|experiments|experimental analysis)\b",
    "tableau": r"\btableau\b",
    "powerbi": r"\b(?:power\s*bi|powerbi)\b",
    "matplotlib": r"\bmatplotlib\b",
    "seaborn": r"\bseaborn\b",
    "hypothesistesting": r"\bhypothesis[\s-]?testing\b|\bstatistical tests?\b",
    "datacleaning": r"\b(?:data cleaning|cleaning datasets|cleaning data)\b",
    "abtesting": r"\b(?:a/?b testing|a/?b tests?|split testing)\b",
    "r": r"\bR programming\b|\bR language\b",
    "spark": r"\b(?:apache spark|spark)\b",
    "html": r"\bhtml5?\b",
    "css": r"\bcss3?\b",
    "javascript": r"\b(?:javascript|java[\s-]?script)\b|(?<!node\.)\bjs\b",
    "react": r"\breact(?:\.js|js)?\b",
    "node": r"\bnode(?:\.js|js)?\b",
    "express": r"\bexpress(?:\.js|js)?\b",
    "mongodb": r"\b(?:mongodb|mongo)\b",
    "typescript": r"\btypescript\b",
    "nextjs": r"\bnext(?:\.js|js)?\b",
    "authentication": r"\bauthentication\b|\bauthorization\b|\bsign[ -]?in\b",
    "crud": r"\bcrud\b",
    "jwt": r"\bjwt\b|\bjson web token\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "github": r"\bgithub\b",
    "postgresql": r"\bpostgres(?:ql)?\b",
    "graphql": r"\bgraphql\b",
    "responsivedesign": r"\bresponsive (?:web )?design\b|\bresponsive websites?\b"
}
SKILL_PATTERNS = {k: re.compile(v, re.I) for k, v in PATTERN_TEXT.items()}

ROLE_DESCRIPTIONS = {
    "AI Engineer": "Build intelligent applications with Python, machine learning, deep learning, NLP, LLMs, model inference, APIs, deployment and production integration.",
    "Machine Learning Engineer": "Develop machine learning systems: data preprocessing, feature engineering, scikit-learn, model evaluation, pipelines, model serving, cloud and MLOps.",
    "Data Scientist": "Analyze data with Python, Pandas, SQL and statistics; perform exploratory data analysis, experiments, visualization and predictive modeling to communicate insights.",
    "Full Stack Developer": "Develop complete web applications using HTML, CSS, JavaScript, React, Node.js, Express.js, databases, REST APIs, authentication and Git."
}
SEMANTIC_TEXTS = [ROLE_DESCRIPTIONS[r] for r in ROLE_DESCRIPTIONS]
SEMANTIC_ROLES = list(ROLE_DESCRIPTIONS.keys())
SEMANTIC_VECTORIZER = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
SEMANTIC_MATRIX = SEMANTIC_VECTORIZER.fit_transform(SEMANTIC_TEXTS)

ROADMAP = {
    "python": "Strengthen Python fundamentals, functions, OOP basics and clean data/automation scripts.",
    "machinelearning": "Practice supervised and unsupervised learning, feature engineering and model evaluation.",
    "deeplearning": "Learn neural-network fundamentals, training loops, loss functions and validation.",
    "numpy": "Practice arrays, indexing, slicing, reshaping, vectorization and broadcasting.",
    "pandas": "Practice cleaning, filtering, grouping, merging and exploratory data analysis.",
    "tensorflow": "Build a small neural-network classifier and compare validation metrics.",
    "pytorch": "Learn tensors, datasets, training loops and build a small deep-learning model.",
    "scikitlearn": "Build an end-to-end scikit-learn pipeline and compare several models.",
    "statistics": "Revise probability, distributions, correlation, confidence intervals and hypothesis testing.",
    "datavisualization": "Create clear charts and dashboards and explain what each chart means.",
    "sql": "Practice JOINs, GROUP BY, window functions, subqueries and database design.",
    "flask": "Build a Flask REST API and connect a trained model to a browser frontend.",
    "fastapi": "Create a typed model-serving API with validation and structured JSON responses.",
    "restapi": "Practice REST design, HTTP methods, JSON payloads, validation and error handling.",
    "git": "Use Git branches, commits, pull requests and a clean project README.",
    "docker": "Containerize an application and run it locally with environment variables.",
    "featureengineering": "Practice creating informative features, encoding variables and avoiding leakage.",
    "modelevaluation": "Compare precision, recall, F1, ROC-AUC and confusion matrices on unseen data.",
    "mlpipeline": "Create a reproducible preprocessing + model pipeline and save it for inference.",
    "modeldeployment": "Deploy a model endpoint and test inference with representative inputs.",
    "nlp": "Build a text classification or keyword extraction project using NLP preprocessing.",
    "llm": "Explore embeddings, prompt design, retrieval and safe LLM application patterns.",
    "html": "Build semantic, accessible web pages with a clean component structure.",
    "css": "Practice responsive layouts with Flexbox, Grid, spacing and reusable styles.",
    "javascript": "Strengthen DOM, events, fetch, async/await and browser-side validation.",
    "react": "Build a React app using components, props, state, forms and API calls.",
    "node": "Structure a Node.js backend with modules, configuration and asynchronous operations.",
    "express": "Build Express routes, middleware, validation and REST CRUD endpoints.",
    "mongodb": "Practice MongoDB collections, CRUD queries, indexing and schema design.",
    "typescript": "Learn TypeScript types, interfaces and safer frontend/backend code.",
    "authentication": "Implement secure registration, login, sessions or token-based authentication.",
    "responsivedesign": "Test layouts across mobile, tablet and desktop breakpoints."
}

PROJECTS = {
    "AI Engineer": [
        "AI Resume Skill Analyzer — NLP + Python + model inference",
        "Career Recommendation System — ML ranking + explainable skill gaps",
        "Model Serving API — Flask/FastAPI + Docker + frontend dashboard"
    ],
    "Machine Learning Engineer": [
        "Customer Churn Prediction — preprocessing + feature engineering + evaluation",
        "End-to-End ML Pipeline — scikit-learn Pipeline + model persistence",
        "Model Monitoring Demo — inference API + validation + experiment tracking"
    ],
    "Data Scientist": [
        "Business Analytics Dashboard — Pandas + SQL + visualization",
        "Student Performance Prediction — EDA + statistics + ML",
        "Customer Segmentation Study — data cleaning + clustering + insights"
    ],
    "Full Stack Developer": [
        "MERN Job Tracker — React + Node + Express + MongoDB",
        "Student Management System — CRUD + authentication + REST API",
        "Portfolio & Project Dashboard — responsive UI + backend + deployment"
    ]
}

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as exc:
    MODEL = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None

try:
    MODEL_METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except Exception:
    MODEL_METRICS = {}


def extract_skills(text: str) -> list[str]:
    text = text or ""
    return [skill for skill, pattern in SKILL_PATTERNS.items() if pattern.search(text)]


def weighted_skill_score(role: str, skills: list[str]) -> tuple[float, list[str], list[str]]:
    weights = ROLE_SKILLS[role]
    skill_set = set(skills)
    matched = [k for k in weights if k in skill_set]
    missing = sorted(
        [k for k in weights if k not in skill_set],
        key=lambda key: weights[key], reverse=True
    )
    total = sum(weights.values()) or 1
    score = round(sum(weights[k] for k in matched) / total * 100, 1)
    return score, matched, missing


def semantic_probabilities(profile_text: str) -> dict[str, float]:
    vector = SEMANTIC_VECTORIZER.transform([profile_text])
    similarities = cosine_similarity(vector, SEMANTIC_MATRIX)[0]
    similarities = np.maximum(similarities, 0)
    if similarities.sum() <= 0:
        probs = np.ones(len(similarities)) / len(similarities)
    else:
        # Softmax-like normalization across role similarities.
        exp_scores = np.exp((similarities - similarities.max()) * 8.0)
        probs = exp_scores / exp_scores.sum()
    return {role: float(prob) for role, prob in zip(SEMANTIC_ROLES, probs)}


def model_probabilities(profile_text: str) -> dict[str, float]:
    if MODEL is None:
        raise RuntimeError(f"ML model is not loaded: {MODEL_LOAD_ERROR}")
    probs = MODEL.predict_proba([profile_text])[0]
    return {role: float(prob) for role, prob in zip(MODEL.classes_, probs)}


def hybrid_rank(profile_text: str, skills: list[str]) -> list[dict]:
    ml_probs = model_probabilities(profile_text)
    sem_probs = semantic_probabilities(profile_text)
    results = []
    for role in ROLE_SKILLS:
        skill_score, matched, missing = weighted_skill_score(role, skills)
        # Final recommendation score combines independent signals.
        final_score = (
            0.55 * ml_probs.get(role, 0.0) * 100
            + 0.25 * sem_probs.get(role, 0.0) * 100
            + 0.20 * skill_score
        )
        results.append({
            "role": role,
            "score": round(final_score, 1),
            "mlProbability": round(ml_probs.get(role, 0.0) * 100, 1),
            "semanticMatch": round(sem_probs.get(role, 0.0) * 100, 1),
            "skillCoverage": skill_score,
            "matchedSkills": [SKILL_LABELS[s] for s in matched],
            "skillGaps": [SKILL_LABELS[s] for s in missing],
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)



def build_roadmap(gaps: list[str]) -> list[str]:
    items = [ROADMAP[s] for s in gaps if s in ROADMAP]
    if items:
        return items[:7]
    return [
        "Build one role-specific project with a measurable outcome and publish a clear README.",
        "Add tests, screenshots and a short architecture diagram to make the project portfolio-ready."
    ]


def profile_completeness(data: dict, resume_text: str) -> int:
    fields = [
        bool((data.get("skills") or "").strip()),
        bool((data.get("interests") or "").strip()),
        bool((data.get("experience") or "").strip()),
        bool((data.get("education") or "").strip()),
        bool(resume_text.strip()),
    ]
    return round(sum(fields) / len(fields) * 100)


def analyze_payload(data: dict, resume_text: str = "") -> dict:
    name = (data.get("name") or "Student").strip() or "Student"
    education = (data.get("education") or "").strip()
    skills_text = (data.get("skills") or "").strip()
    interests = (data.get("interests") or "").strip()
    experience = (data.get("experience") or "").strip()
    target_role = (data.get("targetRole") or "auto").strip()

    # Target role is deliberately not fed into the ML model to avoid label leakage.
    profile_text = " ".join([education, skills_text, interests, experience, resume_text]).strip()
    if len(profile_text) < 10:
        raise ValueError("Please provide more profile information before running the recommendation.")

    detected_skills = extract_skills(profile_text)
    if not detected_skills:
        raise ValueError("No supported technical skills were detected. Add specific skills such as Python, React, SQL, Pandas, Machine Learning or MongoDB.")

    rankings = hybrid_rank(profile_text, detected_skills)
    ml_top = max(rankings, key=lambda x: x["mlProbability"])
    hybrid_top = rankings[0]

    selected_role = hybrid_top["role"] if target_role == "auto" else (target_role if target_role in ROLE_SKILLS else hybrid_top["role"])
    selected = next(item for item in rankings if item["role"] == selected_role)

    reasons = []
    if selected["mlProbability"] >= 50:
        reasons.append(f"The ML classifier strongly associates the profile with {selected_role}.")
    elif selected["mlProbability"] >= 25:
        reasons.append(f"The ML classifier sees meaningful evidence for {selected_role}.")
    if selected["skillCoverage"] >= 60:
        reasons.append(f"The profile covers {selected['skillCoverage']:.0f}% of the weighted skills in the role profile.")
    elif selected["skillCoverage"] > 0:
        reasons.append(f"The profile already covers {selected['skillCoverage']:.0f}% of the weighted skills in the role profile.")
    if selected["matchedSkills"]:
        reasons.append("Strongest evidence: " + ", ".join(selected["matchedSkills"][:5]) + ".")
    reasons.append("Final ranking combines independent ML, semantic similarity and role-skill coverage signals.")

    return {
        "profile": {
            "name": name,
            "education": education or "Not provided",
            "completeness": profile_completeness(data, resume_text)
        },
        "recommendedRole": selected_role,
        "mlRecommendedRole": ml_top["role"],
        "modelProbability": selected["mlProbability"],
        "hybridScore": selected["score"],
        "skillMatchScore": selected["skillCoverage"],
        "predictions": rankings,
        "matchedSkills": selected["matchedSkills"],
        "skillGaps": selected["skillGaps"],
        "roadmap": build_roadmap([k for k in ROLE_SKILLS[selected_role] if k not in detected_skills]),
        "projects": PROJECTS[selected_role],
        "detectedSkills": [SKILL_LABELS[s] for s in detected_skills],
        "resumeSkills": [SKILL_LABELS[s] for s in extract_skills(resume_text)] if resume_text else [],
        "reasons": reasons,
        "model": {
            "name": "TF-IDF (word + character features) + Logistic Regression",
            "holdoutAccuracy": round(MODEL_METRICS.get("holdout_accuracy", 0) * 100, 1),
            "holdoutF1": round(MODEL_METRICS.get("holdout_macro_f1", 0) * 100, 1),
            "trainingRows": MODEL_METRICS.get("dataset_rows", 0),
            "testRows": MODEL_METRICS.get("holdout_test_size", 0),
            "datasetNote": MODEL_METRICS.get("dataset_note", "Academic/demo dataset."),
        },
        "note": "Recommendation score is a hybrid ranking signal. ML probability is a model output, not a probability of getting a job. Results are career guidance, not employment guarantees."
    }
