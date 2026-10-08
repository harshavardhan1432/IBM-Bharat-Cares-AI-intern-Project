from __future__ import annotations

from pathlib import Path
import tempfile

from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader

from engine import MODEL, MODEL_METRICS, analyze_payload

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def read_resume(file_storage) -> str:
    if not file_storage or not file_storage.filename:
        return ""
    filename = file_storage.filename.lower()
    if filename.endswith(".txt"):
        return file_storage.read().decode("utf-8", errors="ignore")
    if filename.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file_storage.save(tmp.name)
            path = tmp.name
        try:
            reader = PdfReader(path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        finally:
            Path(path).unlink(missing_ok=True)
    raise ValueError("Only PDF and TXT resumes are supported.")


@app.route("/")
def home():
    return render_template("index.html", model_metrics=MODEL_METRICS)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok" if MODEL is not None else "error",
        "modelLoaded": MODEL is not None,
        "model": MODEL_METRICS.get("model"),
        "holdoutAccuracy": MODEL_METRICS.get("holdout_accuracy"),
    })


@app.route("/api/model-info")
def model_info():
    return jsonify(MODEL_METRICS)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if request.content_type and request.content_type.startswith("multipart/form-data"):
            data = request.form.to_dict()
            resume_text = read_resume(request.files.get("resume"))
        else:
            data = request.get_json(silent=True) or {}
            resume_text = ""
        return jsonify(analyze_payload(data, resume_text))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Analysis failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
