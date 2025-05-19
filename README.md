# ⚖️📊👩‍⚖️LexEmotion - An Intelligent Dashboard

> **An Intelligent Dashboard for Emotion and Theme Detection in Legal Case Reports**

LexEmotion is a cutting-edge NLP dashboard designed for legal professionals, law firms, and investigators. It leverages the latest advances in Natural Language Processing to extract emotions, detect key themes, and summarize incident or legal reports — in multiple languages and formats.

---

## ⚙️ Key Features

- **Multilingual Input Support**
  - Automatic language detection
  - Seamless offline translation to French (if needed)

- **Advanced Emotion Detection**
  - Uses transformer-based models fine-tuned on emotion-labeled datasets
  - Outputs emotion probabilities per sentence or paragraph

- **Topic Extraction & Summarization**
  - BERTopic or similar unsupervised techniques for high-level theme identification
  - Abstractive summarization via pretrained T5/BART models

- **Interactive Dashboard**
  - Upload documents (PDF, DOCX, TXT)
  - Real-time emotion analytics and visualizations (via Plotly Dash)
  - Exportable results for case files

- **Privacy by Design**
  - Entirely local processing (no external API calls)
  - GDPR-compliant architecture for sensitive case data

- **Scalable Architecture**
  - Modular NLP pipeline built with `spaCy`, `Transformers`, and `FastAPI`
  - Dockerized environment with CI/CD for seamless deployment

---

## 📁 Project Structure

```bash
LexEmotion/
│
├── app/                  # FastAPI backend
│   ├── main.py
│   ├── nlp_pipeline.py   # Core logic: extraction, language detection, translation, emotion detection
│   ├── summarizer.py
│   └── utils/
│
├── dashboard/            # Dash frontend
│   ├── app.py
│   ├── layout.py
│   └── callbacks.py
│
├── models/               # Pretrained or fine-tuned models
│   ├── emotion_model/
│   ├── translation_model/
│   └── summarization_model/
│
├── data/
│   └── uploads/
│
├── tests/                # Unit and integration tests (Pytest)
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
````

---

## 🚀 Quickstart

### Prerequisites

* Python 3.10+
* Docker & Docker Compose
* GPU recommended for inference speed (optional)

---

## 🧠 NLP Pipeline Overview

1. **Text Extraction**

   * Handles raw text, PDFs, and DOCX via `pdfminer.six` and `python-docx`

2. **Language Detection**

   * Based on `langdetect` or `langid`

3. **Translation (if required)**

   * MarianMT (offline HuggingFace translation)

4. **Emotion Detection**

   * Transformer-based model (e.g., BERT or RoBERTa fine-tuned for emotions)

5. **Topic Modeling (optional)**

   * BERTopic for unsupervised theme clustering

6. **Summarization**

   * T5/BART (HuggingFace pipeline)

---

## 📊 Use Cases

* **Legal firms** processing large volumes of testimonies or case files
* **Human rights organizations** analyzing multilingual incident reports
* **Internal legal departments** automating risk and sentiment assessments
* **Psycholegal experts** assessing emotional content in written reports

---

## 📦 Roadmap

* [ ] Add CSV export and PDF report generation
* [ ] Add emotion timeline for long texts
* [ ] Support audio-to-text input (via Whisper)
* [ ] Fine-tune models on law-specific corpora
* [ ] User authentication and multi-user mode

---

## 🔐 Security & Data Privacy

LexEmotion ensures data never leaves the local environment. No third-party APIs are used. Optionally, all logs and intermediate files can be encrypted or deleted after processing.

---

## 🧑‍💼 Maintainer

**Abdias Arsène** —
IT Consultant | NLP Engineer | Applied AI for Legal & Social Impact
[LinkedIn Profile](https://www.linkedin.com/in/abdias-arsene)

---

## 📄 License

MIT License. See `LICENSE` file for details.
