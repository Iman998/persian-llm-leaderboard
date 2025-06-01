# Persian LLM Leaderboard

A Streamlit-based leaderboard for evaluating Large Language Models (LLMs) specifically optimized for the Persian language.

## 🚀 Overview

This project provides an intuitive interface for comparing and benchmarking various Persian LLMs based on their performance across specialized datasets. It helps researchers and developers easily identify strengths and weaknesses of different language models.

## 🌟 Features

* **Interactive Leaderboard:** Real-time visual representation of LLM performances.
* **Model Outputs Comparison:** Directly compare predictions from multiple models side-by-side.
* **Persian-specific Datasets:** Utilize datasets specifically curated for Persian language evaluation.
* **User-friendly Interface:** Built with Streamlit for an intuitive, hassle-free experience.

## 📁 Project Structure

```
.
├── data
│   ├── khayyam_challenge.csv     # Evaluation dataset
│   └── ...                       # Additional datasets
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

Briefly, to start the dashboard:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 🤝 Contributing

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests to enhance this project.

## 📄 License

This project is licensed under the MIT License.
