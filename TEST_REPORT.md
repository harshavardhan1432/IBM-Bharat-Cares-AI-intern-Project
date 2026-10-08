# AI CareerPath — Verification Report

## Model

- Algorithm: TF-IDF word + character features with Logistic Regression
- Classes: 4
- Training profiles: 776
- Hold-out test profiles: 156
- Hold-out accuracy: 96.79%
- Hold-out macro F1: 96.79%

## Functional tests

| Test | Expected | Status |
|---|---|---|
| AI/ML-heavy profile | AI Engineer / ML Engineer ranks highest | PASS |
| MERN-heavy profile | Full Stack Developer ranks highest | PASS |
| Data/SQL/statistics profile | Data Scientist ranks highest | PASS |
| ML pipeline profile | Machine Learning Engineer ranks highest | PASS |
| Empty profile | Friendly validation message | PASS |
| Unsupported text only | Explain that no supported skills were detected | PASS |
| `sklearn` alias | Recognized as Scikit-learn | PASS |
| `Node.js` alias | Recognized as Node.js without incorrectly forcing JavaScript evidence | PASS |
| `k8s` alias | Recognized as Kubernetes | PASS |
| Target role leakage | Target selection is excluded from ML input | PASS |
| PDF/TXT resume path | Resume text extracted and added to profile | IMPLEMENTED |

## Transparency

The UI clearly separates **ML model probability**, **weighted skill coverage**, and the **hybrid recommendation score**. The project states that the model is trained on an academic/synthetic dataset and that recommendations are not employment guarantees.
