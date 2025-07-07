# Persian LLM Leaderboard

A Streamlit-based leaderboard for evaluating Large Language Models (LLMs) specifically optimized for the Persian language.

## 🚀 Overview

This project provides an intuitive interface for comparing and benchmarking various Persian LLMs based on their performance across specialized datasets. It helps researchers and developers easily identify strengths and weaknesses of different language models.

## 🌟 Features

* **Interactive Leaderboard:** Real-time visual representation of LLM performances.
* **Model Outputs Comparison:** Directly compare predictions from multiple models side-by-side.
* **Persian-specific Datasets:** Utilize datasets specifically curated for Persian language evaluation.
* **Fairness Benchmarks:** Includes datasets for gender, religious, regional and age-related bias.
* **User-friendly Interface:** Built with Streamlit for an intuitive, hassle-free experience.
* **Interactive Quick Chart:** Use sliders to pick the metric, adjust the page size, and set the start index.
* **Paginated Category Tables:** Category comparisons now support page controls just like row outputs.
* **Refreshed Look:** Sidebar navigation with page icons and a red→yellow→green gradient highlights numeric columns.

## 📁 Project Structure

```
.
├── data
│   ├── khayyam_challenge
│   │   ├── test.csv              # Evaluation dataset
│   │   └── meta.yaml             # Dataset metadata
│   └── ...                       # Additional datasets
│
│   # meta.yaml fields
│   # -----------------
│   # task: dataset type (e.g. multiple_choice, open_ended, text_generation)
│   # metrics: list of metric names to compute
│   # evaluator: path to the evaluator class
│   # prompt_template: default prompt for the dataset
├── scripts
│   └── run_eval.py               # Evaluation script
├── evaluators
│   └── mcq_evaluator.py          # Evaluation logic for multiple-choice questions
├── app
│   └── streamlit_app.py          # Streamlit dashboard
├── README.md
└── DOCUMENTATION.md
```

## 🛠️ Technologies

* Python
* Streamlit
* Pandas

## 📋 Requirements

* Python 3.10 or higher
* Streamlit
* Pandas
* Additional dependencies listed in `requirements.txt`

## 🚦 Quick Start

Detailed setup and usage instructions are available in the [DOCUMENTATION.md](./DOCUMENTATION.md).


To start the dashboard:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The dashboard includes a **Quick chart** expander where you can choose the metric and use sliders to control how many models are displayed and where the chart starts.

To evaluate all models:

```bash
bash run_all.sh
```
The script now reads each dataset's `meta.yaml` to determine the
appropriate evaluator, prompt template and metrics.
This includes fairness metrics like TPR, FPR, Bias Score and Toxicity Rate when applicable.

To evaluate on a random subset of `N` rows (e.g. 250 rows per dataset):

```bash
bash run_all.sh --n_rows 250
```

> **Note:** The `--n_rows` option is useful for quick benchmarking or debugging with a smaller sample of data.

`run_all.sh` automatically rebuilds the leaderboard. If you run evaluations manually with `scripts/run_eval.py`, call `scripts/build_leaderboard.py` afterward to update `dashboard/leaderboard.csv`.
The script also writes `leaderboard_fa.csv` and `leaderboard_en.csv` filtered by the dataset language.
The Streamlit dashboard will also attempt to build the leaderboard automatically when results are present but the CSV is missing.

## 🤖 LLM Judge Evaluation

Some datasets use a second model to "judge" the quality of a candidate answer. To run these evaluations include the
desired judge datasets in `run_all.sh`'s `DATASET_LIST` or call `scripts/run_eval.py` directly:

```bash
python scripts/run_eval.py --model JUDGE_MODEL \
    --dataset data/summarization_quality/test.csv \
    --evaluator evaluators/judge_evaluator.py \
    --prompt prompts/judge_summarization.jinja2 \
    --out results/summarization_quality/JUDGE_MODEL/JUDGE_MODEL.csv
```

Model configuration files in `models/` specify only the model name and base URL.
The OpenAI API key is provided at runtime via the `OPENAI_API_KEY` environment
variable or a `secrets.toml` file in the project root.

Each model can also have its own key by setting `OPENAI_API_KEY_<MODEL>` or by
listing the key under `[model_keys]` in `secrets.toml`.
Example `secrets.toml`:

```toml
[openai]
api_key = "default-key"

[model_keys]
Qwen30 = "sk-qwen"
gemma-3-27b-it = "sk-gemma"
```

Be sure to add `secrets.toml` to `.gitignore` so the key is not committed:

```bash
echo 'secrets.toml' >> .gitignore
```

The model YAML files remain minimal:

```yaml
name: gpt-4.1-nano-2025-04-14
base_url: "https://api.example.com/v1"
model: "gpt-4.1-nano-2025-04-14"
```

After running a judge evaluation, restart the Streamlit dashboard and select the **LLM Judge** page to explore the scores.

## 🤝 Contributing

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests to enhance this project.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
