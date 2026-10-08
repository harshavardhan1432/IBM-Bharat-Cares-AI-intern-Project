from engine import analyze_payload, extract_skills, MODEL, MODEL_METRICS

PROFILES = [
    ("AI Engineer", {"skills": "Python, NumPy, Pandas, PyTorch, Machine Learning, NLP, FastAPI, Git", "interests": "Artificial Intelligence, LLM applications", "experience": "Built an NLP assistant and deployed a model API."}),
    ("Full Stack Developer", {"skills": "HTML, CSS, JavaScript, React, Node.js, Express.js, MongoDB, REST API, Git", "interests": "Web Development", "experience": "Built a MERN job tracker with authentication."}),
    ("Data Scientist", {"skills": "Python, Pandas, SQL, Statistics, Power BI, Matplotlib, Scikit-learn", "interests": "Data Science, Analytics", "experience": "Performed EDA and built a business KPI dashboard."}),
    ("Machine Learning Engineer", {"skills": "Python, scikit-learn, Pandas, NumPy, feature engineering, model evaluation, Docker, ML pipelines", "interests": "Machine Learning", "experience": "Built a churn prediction pipeline and evaluated precision recall F1."}),
]

def main():
    assert MODEL is not None, "Trained model not found. Run train_model.py first."
    assert MODEL_METRICS["holdout_accuracy"] >= 0.90
    for expected, data in PROFILES:
        result = analyze_payload(data)
        assert result["recommendedRole"] == expected, (expected, result["recommendedRole"])
        assert len(result["predictions"]) == 4
        assert result["hybridScore"] > 0
    assert extract_skills("Node.js Express MongoDB") == ["node", "express", "mongodb"]
    assert "scikitlearn" in extract_skills("sklearn with Pandas")
    try:
        analyze_payload({})
    except ValueError:
        pass
    else:
        raise AssertionError("Empty profile should fail validation")
    print("All engine tests passed.")
    print(f"Hold-out accuracy: {MODEL_METRICS['holdout_accuracy']:.2%}")

if __name__ == "__main__":
    main()
