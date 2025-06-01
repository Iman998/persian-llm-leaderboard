# 📚 Persian LLM Leaderboard Documentation

This guide provides comprehensive instructions on setting up, running, and contributing to the Persian LLM Leaderboard project.

---

## ⚙️ Installation

### Step 1: Clone the Repository

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/yourusername/persian-llm-leaderboard.git
cd persian-llm-leaderboard
```

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

To manually run individual evaluations, use:

```bash
python scripts/run_eval.py --model MODEL_NAME --dataset DATASET_NAME
```

**Example:**

```bash
python scripts/run_eval.py --model Qwen30 --dataset khayyam_challenge
```

Evaluation results are saved under the `results/` directory.

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
  * Update evaluation scripts in `scripts/run_eval.py` to recognize new datasets.

* **Models:**

  * Integrate new model configurations within the evaluation scripts located in the `scripts/` and `evaluators/` directories.

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
