# 🏷️ TagFlow AI

A **Many-to-Many POS Tagging** project using a **Simple RNN**.

## Concept
Many input words → Many output POS tags.

## Run
```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.predict
streamlit run app.py
```

> The included POS corpus is a tiny runnable demo. Use a large annotated POS dataset for real model performance.
