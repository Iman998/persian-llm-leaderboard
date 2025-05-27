# Documentation for Running Persian LLM Leaderboard

This document provides detailed instructions on setting up and running the Persian LLM Leaderboard project.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/persian-llm-leaderboard.git
cd persian-llm-leaderboard
```

### Step 2: Setup Python Environment

Create and activate a virtual environment:

```bash
python -m venv leaderboard_env
source leaderboard_env/bin/activate  # Linux/Mac
leaderboard_env\Scripts\activate.bat  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Evaluations

To evaluate models against datasets, run:

```bash
bash run_all.sh
```

or manually run evaluations with:

```bash
python scripts/run_eval.py --model MODEL_NAME --dataset DATASET_NAME
```

Example:

```bash
python scripts/run_eval.py --model Qwen30 --dataset khayyam_challenge
```

---

## Running the Streamlit App

Launch the Streamlit web application with the following command:

```bash
streamlit run app/streamlit_app.py
```

The leaderboard interface will open in your browser at:

```
http://localhost:8501
```

---

## Adding New Models and Datasets

* Place new dataset CSV files into the `data/` directory.
* Modify evaluation scripts in `scripts/run_eval.py` to include new models and datasets.


