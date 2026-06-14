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
* **Comprehensive Metrics:** Evaluate translations with BLEU, METEOR, chrF and 1‑TER.
* **Pairwise Battle Board:** Compare two models with an independent judge and inspect win, loss, and equal rates.
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
│   # judge: set to true if evaluation uses an LLM judge
│   # use_reference: pass reference text to judge prompts (default true)
│   # judge: run LLM-judge evaluation after standard scoring
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
appropriate evaluator, prompt template, metrics and judge flag.
This includes fairness metrics like TPR, FPR, Bias Score and Toxicity Rate when applicable.

To evaluate on a random subset of `N` rows (e.g. 250 rows per dataset):

```bash
bash run_all.sh --n_rows 250
```

> **Note:** The `--n_rows` option is useful for quick benchmarking or debugging with a smaller sample of data.

`run_all.sh` automatically rebuilds the leaderboard. If you run evaluations manually with `scripts/run_eval.py`, call `scripts/build_leaderboard.py` afterward to update `dashboard/leaderboard.csv`.
The script also writes `leaderboard_fa.csv` and `leaderboard_en.csv` filtered by the dataset language.
The Streamlit dashboard will also attempt to build the leaderboard automatically when results are present but the CSV is missing.

### Multiple ROUGE metrics

Datasets can request any combination of ROUGE-1, ROUGE-2 and ROUGE-L. For example:

```yaml
metrics: [rouge1, rouge2, rougel]
```

You can also override the list when calling `scripts/run_eval.py`:

```bash
python scripts/run_eval.py --dataset data/summarization/test.csv \
    --model MY_MODEL --metrics rouge1,rouge2,rougel \
    --out results/summarization/MY_MODEL.csv
```

If several ROUGE metrics are provided their scores are averaged.

## 🤖 LLM Judge Evaluation

Some datasets use a second model to judge candidate outputs. Judge evaluation
runs only when you pass `--judge` and the dataset's `meta.yaml` enables it.
The recommended form keeps the evaluator model and rubric with the dataset:

```yaml
judge:
  enabled: true
  model: deepseek-chat-judge
  evaluator: evaluators/judge_evaluator.py
  reference_prompt_template: prompts/judge_zharfa_translation.jinja2
  no_reference_prompt_template: prompts/judge_zharfa_translation_noref.jinja2
  score_min: 0
  score_max: 100
  metrics: [llm_judge_score]
```

`judge: true` is still accepted for older datasets, but it falls back to the
candidate model unless `--judge-model` is supplied.

Run Zharfa generation followed by its configured judge:

```bash
python scripts/main.py \
    --models CANDIDATE_MODEL \
    --datasets zharfa_translate \
    --judge
```

To judge results that already exist without calling the candidate model again:

```bash
python scripts/main.py \
    --models CANDIDATE_MODEL \
    --datasets zharfa_translate \
    --judge \
    --judge-only \
    --judge-mode both
```

`--judge-mode` accepts `reference`, `no-reference`, or `both`. Use
`--judge-model MODEL_STUB` to override the dataset setting. Reference and
no-reference results are written separately under
`results/<dataset>_judge_reference/` and
`results/<dataset>_judge_no_reference/`.

The same options are available in `run_all.sh` through `RUN_JUDGE`,
`JUDGE_ONLY`, `JUDGE_MODE`, and `JUDGE_MODEL`.

## Battle Board

The **Battle** board compares two models row by row using an independent judge.
It reuses existing result CSVs, presents the outputs anonymously as Candidate A
and Candidate B, counterbalances their order, and records one of three outcomes:
`model_1`, `model_2`, or `equal`.

Configure a dataset for battles:

```yaml
battle:
  enabled: true
  model: deepseek-chat-judge
  evaluator: evaluators/battle_evaluator.py
  prompt_template: prompts/battle.jinja2
  use_reference: true
```

Run a battle from existing outputs:

```bash
python scripts/main.py \
    --datasets zharfa_translate \
    --battle \
    --battle-only \
    --battle-model-1 MODEL_1 \
    --battle-model-2 MODEL_2 \
    --battle-judge-model deepseek-chat-judge
```

Row-level decisions are stored at
`results/battle/<dataset>/<model-1>__vs__<model-2>/battle.csv`.
The generated `dashboard/battle_board.csv` contains win, loss, and equal rates
for both models on every dataset. The Streamlit **Battle** page displays the
summary table and comparison chart.

`run_all.sh` exposes the same behavior through `RUN_BATTLE`, `BATTLE_ONLY`,
`BATTLE_MODEL_1`, `BATTLE_MODEL_2`, and `BATTLE_JUDGE_MODEL`.

Model configuration files in `models/` include the model name and base URL.
You can also set optional request controls such as `enable_thinking` for reasoning-capable
models (for example Qwen 3.5): set it to `false` to disable thinking mode or `true` to enable it.
The OpenAI API key is provided at runtime via the `OPENAI_API_KEY` environment
variable or a `secrets.toml` file in the project root.

Each model can also have its own key by setting `OPENAI_API_KEY_<MODEL>` or by
listing the key under `[model_keys]` in `secrets.toml`.
For the included `deepseek-chat-judge` configuration, use
`OPENAI_API_KEY_DEEPSEEK_CHAT`.
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

Example model YAML:

```yaml
name: qwen3.5
base_url: "https://api.example.com/v1"
model: "qwen3.5"
enable_thinking: false  # optional
```

After running a judge evaluation, restart the Streamlit dashboard and select the **LLM Judge** page to explore the scores.

## 🤝 Contributing

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests to enhance this project.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
