# 📚 Persian LLM Leaderboard Documentation

This guide provides comprehensive instructions on setting up, running, and contributing to the Persian LLM Leaderboard project.

---

## ⚙️ Installation

### Step 1: Clone the Repository

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/yourusername/persian-llm-leaderboard.git
cd persian-llm-leaderboard
````

### Step 2: Python Environment Setup

Create and activate a virtual environment:

**On Linux/Mac:**

```bash
python3 -m venv leaderboard_env
source leaderboard_env/bin/activate
```

**On Windows:**

```cmd
python -m venv leaderboard_env
leaderboard_env\Scripts\activate.bat
```

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Evaluations

To evaluate all predefined models and datasets at once, use:

```bash
bash run_all.sh
```

### Optional: Sample Random Subset of Rows

You can evaluate a random sample of `N` rows from each dataset using:

```bash
bash run_all.sh --n_rows N
```

**Example:**

```bash
bash run_all.sh --n_rows 250
```

This is useful for quick testing or partial evaluation when working with large datasets.

### Manually Run Individual Evaluations

You can also run evaluations manually using:

```bash
python scripts/run_eval.py --model MODEL_NAME --dataset DATASET_PATH [--n_rows N]
```

**Example:**

```bash
python scripts/run_eval.py --model Qwen30 --dataset data/khayyam_challenge/test.csv --n_rows 250
```

Evaluation results are saved under `results/<dataset>/<model>.csv`.

When you sample rows with `run_all.sh --n_rows N`, each run writes
`results/<dataset>/<model>_N.csv` and also copies it to
`results/<dataset>/<model>.csv` so the leaderboard has a consistent filename.
Both files remain in the dataset folder under `results/`.

---

## 🌐 Launching the Streamlit Dashboard

Start the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

Then, open your browser and navigate to:

```
http://localhost:8501
```

---

## 📊 Adding New Models and Datasets

To expand the leaderboard:

* **Datasets:**

  * Add new CSV files to the `data/` directory.
  * Ensure `scripts/run_eval.py` supports your dataset's format.

* **Models:**

  * Add model configuration files to `models/`.
  * Update any evaluation logic in `scripts/` or `evaluators/` if necessary.

---

## 🛠️ Contributing

We encourage contributions! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push your branch: `git push origin feature/my-feature`.
5. Open a pull request to the main branch.

---

## 📖 Further Assistance

For additional help or questions, please open an issue in the repository.

---
