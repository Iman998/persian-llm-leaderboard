# 📚 Persian LLM Leaderboard Documentation

This guide provides comprehensive instructions on setting up, running, and contributing to the Persian LLM Leaderboard project.

---

## ⚙️ Installation

### Step 1: Clone the Repository

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/iman998/persian-llm-leaderboard.git
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
The script automatically uses each dataset's `meta.yaml` to select the
evaluator, prompt template and metrics.

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
python scripts/run_eval.py --model MODEL_NAME --dataset DATASET_PATH --out results/<dataset>/<model>/<model>.csv [--n_rows N]
```

**Example:**

```bash
python scripts/run_eval.py --model Qwen30 --dataset data/khayyam_challenge/test.csv --out results/khayyam_challenge/Qwen30/Qwen30.csv --n_rows 250
```

Evaluation results are saved under `results/<dataset>/<model>/<model>.csv`.
After running `scripts/run_eval.py` you must rebuild the leaderboard:

```bash
python scripts/build_leaderboard.py --results_dir results --datasets_dir data --out dashboard/leaderboard.csv
```

When you sample rows with `run_all.sh --n_rows N` or pass `--n_rows` to
`scripts/run_eval.py`, each run writes `results/<dataset>/<model>/<model>_N.csv`
and also copies it to `results/<dataset>/<model>/<model>.csv` so the leaderboard
has a consistent filename.
Each model now has its own subfolder containing the main CSV, raw outputs, and per‑category scores.

### Running LLM Judge Evaluations

Some datasets use a separate model to score candidate answers. To run these evaluations you can either include the
judge datasets in `run_all.sh` or invoke `scripts/run_eval.py` manually. Example:

```bash
python scripts/run_eval.py --model JUDGE_MODEL \
    --dataset data/translation_quality/test.csv \
    --evaluator evaluators/judge_evaluator.py \
    --prompt prompts/judge_translation.jinja2 \
    --out results/translation_quality/JUDGE_MODEL/JUDGE_MODEL.csv
```

The judge model's YAML file only defines the model name and `base_url`.
Provide the OpenAI API key via the `OPENAI_API_KEY` environment variable or in
`secrets.toml` at the project root so the evaluator can contact the API. Each
model can specify a separate key using `OPENAI_API_KEY_<MODEL>` or the
`[model_keys]` table in `secrets.toml`.
Example:

```toml
[openai]
api_key = "default-key"

[model_keys]
Qwen30 = "sk-qwen"
gemma-3-27b-it = "sk-gemma"
```

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
The sidebar includes an **LLM Judge** page where you can explore scores from judge evaluations.
Navigation uses a sidebar radio widget with page icons, making it easy to switch between the
Leaderboard, Dataset view and Judge pages. Tables use the same red→yellow→green gradient as the
leaderboard so category comparisons share the color scheme.

Numeric columns in tables use a red→yellow→green gradient. The top three
``Average`` values are highlighted with gold, silver and bronze, and the
corresponding model names are coloured to match.

---

## 📐 Judge Metrics Explained

The repository defines several metrics used during evaluation (see the `metrics/` directory):

* **accuracy** – portion of predictions equal to the reference.【F:metrics/accuracy.py†L2-L16】
* **exact_match** – strict string match of prediction and label.【F:metrics/exact_match.py†L1-L16】
* **f1** – token-level F1 over shared words.【F:metrics/f1.py†L4-L18】
* **bleu** – 4‑gram BLEU with brevity penalty for translations.【F:metrics/bleu.py†L1-L27】
* **rouge** – ROUGE-L based on the longest common subsequence.【F:metrics/rouge.py†L1-L25】
* **llm_judge_score** – mean numeric rating returned by the judge model.【F:metrics/llm_judge_score.py†L1-L4】
* **precision**, **recall** and **matthews_corrcoef** – common classification metrics.【F:metrics/precision.py†L1-L19】【F:metrics/recall.py†L1-L18】【F:metrics/matthews_corrcoef.py†L1-L22】
* **tpr** and **fpr** – true and false positive rates.【F:metrics/tpr.py†L1-L12】【F:metrics/fpr.py†L1-L12】
* **bias_score** – `(1 - TPR) + FPR` to capture bias.【F:metrics/bias_score.py†L1-L15】
* **toxicity_rate** – fraction of outputs flagged toxic.【F:metrics/toxicity_rate.py†L1-L8】

---

## 📊 Adding New Models and Datasets

To expand the leaderboard:

* **Datasets:**

  * Add new CSV files to the `data/` directory.
  * Provide a `meta.yaml` describing the dataset:
    - `task`: task type (e.g. `multiple_choice`, `open_ended`, `text_generation`)
    - `metrics`: list of metric modules to compute
    - `evaluator`: path to the evaluator class
    - `prompt_template`: default prompt template
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
