from __future__ import annotations

from pathlib import Path
import json
import random

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

ROLE_BANK = {
    "AI Engineer": {
        "core": ["python", "machine learning", "deep learning", "pytorch", "tensorflow", "numpy", "pandas", "nlp", "llm", "model deployment", "rest api", "flask", "fastapi"],
        "support": ["sql", "git", "docker", "scikit learn", "computer vision", "transformers", "streamlit", "cloud"],
        "tasks": [
            "build intelligent applications", "train and evaluate neural network models", "deploy machine learning models as APIs",
            "work with NLP or computer vision", "create inference pipelines", "integrate AI models into products",
            "experiment with transformers and LLM applications", "serve models through Flask or FastAPI"
        ],
        "projects": [
            "AI resume analyzer", "career recommendation system", "NLP classification app", "image classification API",
            "LLM assistant", "model deployment dashboard"
        ],
        "phrases": [
            "artificial intelligence", "intelligent systems", "model inference", "prompt engineering", "embeddings", "vector database"
        ]
    },
    "Machine Learning Engineer": {
        "core": ["python", "machine learning", "scikit learn", "feature engineering", "model evaluation", "numpy", "pandas", "ml pipeline", "model deployment", "pytorch", "tensorflow"],
        "support": ["sql", "git", "docker", "cloud", "rest api", "flask", "fastapi", "data preprocessing", "mlflow", "kubernetes"],
        "tasks": [
            "prepare training data", "engineer predictive features", "train supervised learning models", "compare model metrics",
            "build reusable ML pipelines", "tune hyperparameters", "serve trained models", "monitor model performance"
        ],
        "projects": [
            "customer churn predictor", "loan risk model", "demand forecasting pipeline", "fraud detection model",
            "end to end ML pipeline", "model serving API"
        ],
        "phrases": ["cross validation", "classification", "regression", "hyperparameter tuning", "feature selection", "model monitoring"]
    },
    "Data Scientist": {
        "core": ["python", "pandas", "numpy", "statistics", "sql", "data visualization", "exploratory data analysis", "scikit learn", "machine learning", "experimentation"],
        "support": ["tableau", "power bi", "matplotlib", "seaborn", "r", "spark", "data cleaning", "hypothesis testing", "a b testing", "storytelling"],
        "tasks": [
            "analyze structured and unstructured data", "clean and transform datasets", "perform exploratory data analysis",
            "test hypotheses", "build predictive models", "create dashboards and visualizations", "communicate data insights",
            "evaluate experiments and business metrics"
        ],
        "projects": [
            "sales analytics dashboard", "student performance prediction", "customer segmentation study", "business KPI dashboard",
            "churn analysis report", "exploratory data analysis project"
        ],
        "phrases": ["business insights", "data storytelling", "statistical analysis", "descriptive statistics", "predictive analytics", "decision support"]
    },
    "Full Stack Developer": {
        "core": ["html", "css", "javascript", "react", "node.js", "express.js", "mongodb", "sql", "rest api", "git", "typescript"],
        "support": ["next.js", "redux", "authentication", "crud", "jwt", "docker", "aws", "github", "postgresql", "graphql", "responsive design"],
        "tasks": [
            "build responsive web applications", "create React user interfaces", "develop Node.js and Express APIs", "design database schemas",
            "implement authentication and authorization", "connect frontend and backend services", "build CRUD features",
            "deploy full stack applications"
        ],
        "projects": [
            "MERN job tracker", "e commerce web app", "student management system", "social media dashboard",
            "portfolio website", "full stack task manager"
        ],
        "phrases": ["frontend development", "backend development", "web application", "client server integration", "REST services", "user interface"]
    }
}

COMMON = ["teamwork", "problem solving", "software project", "internship", "college project", "development", "testing"]
ROLE_ALIASES = {
    "AI Engineer": ["AI engineer", "artificial intelligence engineer", "AI developer", "intelligent application developer"],
    "Machine Learning Engineer": ["ML engineer", "machine learning engineer", "machine learning developer", "ML developer"],
    "Data Scientist": ["data scientist", "applied data scientist", "data science professional", "analytics scientist"],
    "Full Stack Developer": ["full stack developer", "full-stack developer", "MERN developer", "web application developer"]
}


def build_sentence(role: str, rng: random.Random) -> str:
    bank = ROLE_BANK[role]
    core = rng.sample(bank["core"], rng.randint(5, 8))
    support = rng.sample(bank["support"], rng.randint(1, 3))
    tasks = rng.sample(bank["tasks"], rng.randint(2, 3))
    project = rng.choice(bank["projects"])
    phrases = rng.sample(bank["phrases"], rng.randint(1, 2))
    common = rng.sample(COMMON, rng.randint(1, 2))

    return (
        f"Student profile applying for technical roles. Skills include {', '.join(core)}. "
        f"Additional experience with {', '.join(support)}. "
        f"Interested in {', '.join(phrases)}. "
        f"Project experience: {project}. "
        f"Has worked on tasks such as {', '.join(tasks)}. "
        f"Also demonstrates {', '.join(common)}."
    )


def add_ambiguous_examples(rows: list[dict], rng: random.Random, per_role: int = 24) -> None:
    """Add intentionally overlapping profiles so the hold-out test is not trivial."""
    for role, bank in ROLE_BANK.items():
        for _ in range(per_role):
            other_role = rng.choice([r for r in ROLE_BANK if r != role])
            core = rng.sample(bank["core"], 4)
            cross = rng.sample(ROLE_BANK[other_role]["support"], 2)
            task = rng.choice(bank["tasks"])
            project = rng.choice(bank["projects"])
            rows.append({
                "text": (
                    f"Technical student profile. Main skills: {', '.join(core)}. "
                    f"Additional tools: {', '.join(cross)}. "
                    f"Built {project}. "
                    f"Experience includes {task}."
                ),
                "role": role,
            })


def build_dataset() -> pd.DataFrame:
    rng = random.Random(20261008)
    rows: list[dict] = []
    for role in ROLE_BANK:
        for _ in range(136):
            rows.append({"text": build_sentence(role, rng), "role": role})
    add_ambiguous_examples(rows, rng, per_role=24)
    # Sparse/overlapping profiles make evaluation less trivial and closer to real student inputs.
    sparse_common = ["python", "sql", "git", "docker", "api", "data", "javascript", "machine learning", "pandas", "project"]
    for role, bank in ROLE_BANK.items():
        for _ in range(34):
            pieces = rng.sample(sparse_common, 3) + rng.sample(bank["core"], 2)
            generic_task = rng.choice([
                "built a project", "worked on a college project", "created a prototype",
                "worked with a small dataset", "developed an application", "completed an internship project"
            ])
            rows.append({
                "text": f"Student portfolio: {', '.join(pieces)}. {generic_task} and used standard software development practices.",
                "role": role,
            })

    # A small amount of realistic wording variation, while keeping labels deterministic.
    variations = [
        ("developed", "built"), ("implemented", "created"), ("worked on", "developed"),
        ("predictive", "prediction"), ("application", "app"), ("analysis", "analytics")
    ]
    for row in rows:
        for old, new in variations:
            if rng.random() < 0.12:
                row["text"] = row["text"].replace(old, new)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def build_challenge_set() -> pd.DataFrame:
    samples = [
        ("I use Python and PyTorch to train neural networks, work with embeddings and deploy inference endpoints using FastAPI. I built an NLP assistant.", "AI Engineer"),
        ("My work focuses on Python, feature engineering, scikit-learn, cross-validation and model monitoring. I built a churn prediction pipeline and exposed it through Flask.", "Machine Learning Engineer"),
        ("I clean datasets in Pandas, use SQL and statistics, perform exploratory analysis and create Power BI dashboards to explain business KPIs.", "Data Scientist"),
        ("I build responsive interfaces with React and JavaScript, create Node and Express REST APIs and use MongoDB with JWT authentication.", "Full Stack Developer"),
        ("Python, transformers, LLM APIs, vector search, prompt engineering and FastAPI are the main tools in my AI projects.", "AI Engineer"),
        ("I compare regression and classification models, tune hyperparameters, build preprocessing pipelines and track experiments.", "Machine Learning Engineer"),
        ("My strongest areas are SQL, Pandas, hypothesis testing, visualization and communicating insights from customer data.", "Data Scientist"),
        ("I have completed MERN projects using React, Node.js, Express.js, MongoDB, HTML, CSS and REST APIs.", "Full Stack Developer"),
        ("Built an intelligent document classifier using Python, TensorFlow, OCR and an API that returns predictions.", "AI Engineer"),
        ("Created a reusable sklearn pipeline for feature engineering, model selection, cross-validation and deployment.", "Machine Learning Engineer"),
        ("Performed EDA on sales data, joined datasets with SQL, ran statistical tests and presented insights with Tableau.", "Data Scientist"),
        ("Created a production-style web app with React, Node, Express, PostgreSQL, authentication and Git-based deployment.", "Full Stack Developer"),
        ("Python and machine learning are both in my stack. I mainly analyze business data with Pandas, statistics, SQL and charts.", "Data Scientist"),
        ("I know JavaScript and React and have some Python. My main projects are APIs, authentication, MongoDB and Node.js backends.", "Full Stack Developer"),
        ("I enjoy training models with scikit-learn and PyTorch, handling feature engineering, evaluation and serving models for applications.", "Machine Learning Engineer"),
        ("I build AI products around LLMs and NLP, integrate model APIs, and turn prototypes into usable applications with Python.", "AI Engineer"),
        ("My profile is mostly SQL and data analysis with Python, Power BI, statistics and experiments for business decisions.", "Data Scientist"),
        ("React, TypeScript, Node.js, Express, MongoDB and REST are the main skills behind my web development projects.", "Full Stack Developer"),
        ("I focus on model pipelines, training data, preprocessing, feature selection and comparing metrics such as precision, recall and F1.", "Machine Learning Engineer"),
        ("I use Python for AI, NumPy and Pandas for data preparation, TensorFlow for deep learning and Flask for model serving.", "AI Engineer"),
        ("I use Python, Pandas, SQL and some machine learning. Most of my work is exploratory analysis and reporting for business users.", "Data Scientist"),
        ("I know Python, Pandas, scikit-learn and statistics, and I have built prediction models, but I also use Tableau for communicating results.", "Data Scientist"),
        ("I use Python, TensorFlow, Pandas and Flask. I package trained models behind APIs and focus on taking AI prototypes into applications.", "AI Engineer"),
        ("I use Python, scikit-learn, Pandas and ML pipelines. I focus on preprocessing, cross-validation and model evaluation rather than dashboards.", "Machine Learning Engineer"),
        ("I build React and Node applications and sometimes use Python for small data scripts. My main work is frontend, backend and database integration.", "Full Stack Developer"),
        ("Python, SQL and APIs are common in my projects, but I mainly create React interfaces and Express/Node services backed by MongoDB.", "Full Stack Developer")
    ]
    return pd.DataFrame(samples, columns=["text", "role"])


def main() -> None:
    df = build_dataset()
    data_path = DATA_DIR / "career_training_data.csv"
    df.to_csv(data_path, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["role"], test_size=0.20, random_state=42, stratify=df["role"]
    )

    features = FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=4000,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=5000,
        )),
    ])

    pipeline = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=2500, C=3.0, class_weight="balanced", random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)

    accuracy = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))
    labels = list(pipeline.classes_)
    cm = confusion_matrix(y_test, pred, labels=labels).tolist()
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)

    model_path = MODEL_DIR / "career_model.joblib"
    joblib.dump(pipeline, model_path)

    metrics = {
        "model": "TF-IDF word+character features + Logistic Regression",
        "dataset_rows": int(len(df)),
        "classes": labels,
        "holdout_test_size": int(len(X_test)),
        "holdout_accuracy": round(accuracy, 4),
        "holdout_macro_f1": round(macro_f1, 4),
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm,
        "classification_report": report,
        "dataset_note": "Synthetic academic profiles created from a curated role-skill/task taxonomy; not a real labor-market dataset.",
        "external_basis": [
            "O*NET OnLine software skills for Data Scientists and Software Developers (updated 2026).",
            "ESCO Skills-Occupations Matrix / Software Developer knowledge and skill concepts (ESCO v1.2.1, updated 2025)."
        ],
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training rows: {len(df)}")
    print(f"Hold-out validation accuracy: {accuracy:.2%}")
    print(f"Hold-out macro F1: {macro_f1:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, labels=labels, zero_division=0))
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
