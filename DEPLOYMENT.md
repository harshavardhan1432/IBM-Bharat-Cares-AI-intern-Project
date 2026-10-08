# Deployment

## Local demo

On Windows, open this folder in VS Code and run `run_windows.bat`. The app will train/load the ML model and start Flask on port 5000.

## Render (recommended for a public demo)

1. Upload the repository to GitHub.
2. Create a Render **Web Service** from the GitHub repository.
3. Build command:

```bash
pip install -r requirements.txt && python train_model.py
```

4. Start command:

```bash
gunicorn app:app
```

5. Render will expose a public HTTPS URL.

## Important

The public demo is the Flask application, not the root `index.html` launcher. The Python backend is required for the trained scikit-learn model and PDF resume extraction.
