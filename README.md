# Exploratory Data Analysis and Emotion Detection on WhatsApp Chats

> **Final Year MCA Project** — a two-part system that (1) performs **Exploratory Data Analysis (EDA)** on an exported WhatsApp chat file and (2) **detects the emotion** behind text using Deep Learning (**LSTM / BiLSTM**).

![LSTM Network](LSTM.png)

<p align="center">
  <b>Author:</b> Purushottam Kumar (2020178043, MCA (R)) &nbsp;|&nbsp;
  <b>Guide:</b> Dr. T. Mala, Associate Professor
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Emotion Detection Models](#-emotion-detection-models)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [How to Export a WhatsApp Chat](#-how-to-export-a-whatsapp-chat)
- [Project Documents](#-project-documents)
- [Notes, Assumptions & Limitations](#-notes-assumptions--limitations)
- [Future Scope](#-future-scope)
- [Author & Acknowledgements](#-author--acknowledgements)

---

## 🔎 Overview

Existing WhatsApp chat analysers are mostly limited to plain text statistics and struggle to detect the **actual emotion** behind a conversation. This project addresses both problems:

1. **Exploratory Data Analysis (EDA)** — converts a raw exported WhatsApp `.txt` chat (individual or group) into a clean, feature-rich DataFrame and renders **interactive visual insights** through a **Streamlit** dashboard: activity timelines, heatmaps, top contributors, URL/media sharing patterns, emoji usage, word clouds, member clustering and sentiment scores.
2. **Emotion Detection** — classifies English sentences into emotions using Deep Learning sequence models (**LSTM with a Conv1D front-end** and a **stacked Bidirectional LSTM**) built with **TensorFlow / Keras**.

The project was designed around four modules — **Data Collection**, **Pre-processing**, **Exploratory Data Analysis** and the **LSTM (Deep Learning) Module**.

---

## 🎯 Objectives

- Develop a system that can analyse WhatsApp chats using Exploratory Data Analysis and present hidden insights in a visual, interactive format.
- Detect emotion from text using **LSTM (Long Short Term Memory)** networks.
- Handle real-world chat artefacts: multiple date/time formats, 12-hour → 24-hour conversion, system/group notifications, deleted messages, media placeholders, Hinglish/native-language stop-words and emoji-heavy messages.

---

## ✨ Key Features

### 1. Pre-processing Module (`Exploratory Data Analysis/preprocessor.py`)

| Step | What it does |
| --- | --- |
| Timestamp splitting | Regex-based separation of every message into its timestamp + content |
| Date format detection | Auto-detects `dd/mm/yy`, `dd/mm/yyyy` and `mm/dd/yyyy` variants |
| Time conversion | Converts 12-hour `hh:mm AM/PM` timestamps to 24-hour `HH:MM:SS` |
| User / message separation | Splits sender name from the message body (including multi-line messages) |
| Feature engineering | Adds day, date, month, year, hour, minute, weekday, period-of-day (`Late Night → Night`), word count, character count, unique words, URL count, media count, emoji count and punctuation — up to **26 attributes** |
| Cleaning | Removes newlines, system notifications (`group_notification`) and normalises the dataframe |

### 2. Exploratory Data Analysis Dashboard (`Exploratory Data Analysis/app.py` + `helper.py`)

A **Streamlit** app (35 helper functions) that renders:

- 🔹 **General statistics** — total messages, words, members, media, links, deleted messages, emojis, average words/message, first & last message date, most/least active day.
- 🔹 **Timeline & activity** — time-series plot of messages over time, most active **months**, **days** and **hours** (24-hour clock).
- 🔹 **Activity heatmaps** — *weekday × hour* and *month × week* heatmaps of group engagement.
- 🔹 **Member involvement** — top 15/top users, percentage contribution bars, member clustering scatter (messages vs. emojis, links, words, letters).
- 🔹 **Media & URL module** — who shares the most media, most-shared websites, URL bar/line/scatter-bubble charts.
- 🔹 **Text module** — top 25 most common words (Hinglish + multi-language stop-word filtering) and a **word cloud** built with a custom Devanagari-compatible font.
- 🔹 **Emoji module** — top 20 emojis with English descriptions (`emoji.demojize`) and a per-member emoji pie chart.
- 🔹 **Sentiment analysis** — member-wise polarity score using **TextBlob** to answer *"Who has a positive sentiment?"*
- 🔹 **Conversation data** — normal table view and an interactive **AgGrid** table with row selection.

### 3. Emotion Detection Module (`Emotion Detection/LSTM_BiLSTM/`)

- Two independent, fully documented Jupyter notebooks: **`Model_LSTM.ipynb`** and **`Model_BILSTM.ipynb`**.
- Full NLP pipeline: text cleaning → label encoding → one-hot / sparse targets → tokenization → padding → embedding → deep model → early stopping → evaluation (accuracy/loss curves, classification report, confusion matrix).
- A small **Streamlit sentiment demo app** (`app.py`) using **TextBlob** + **VADER** for sentence polarity, subjectivity and token-level sentiment.
- Reusable artefacts: `labelEncoder.pickle` (fitted label encoder) and `labelEncoder.jpg` (label-encoding illustration).

---

## 🏗 System Architecture

```
                    ┌────────────────────────────────────────────┐
                    │        WhatsApp Chat Export (.txt)         │
                    │   Mobile export  /  (PushBullet API*)      │
                    └───────────────────────┬────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                     PRE-PROCESSING MODULE                     │
        │  Regex timestamp split → DataFrame creation → date/time       │
        │  format detection & conversion → user/message separation →    │
        │  feature engineering → cleaning (26 attributes)               │
        └───────────────┬───────────────────────────────┬───────────────┘
                        │                               │
                        ▼                               ▼
    ┌───────────────────────────────┐   ┌──────────────────────────────────┐
    │  EDA MODULE  (Streamlit)      │   │  EMOTION DETECTION MODULE (DL)   │
    │  • Fetch_Stats                │   │  Read dataset → clean text →     │
    │  • Timeline / Involvement     │   │  Label Encoding → Tokenization → │
    │  • Activity Map (heatmaps)    │   │  Padding (256) → Embedding →     │
    │  • URL module                 │   │  Dropout → Conv1D → MaxPooling → │
    │  • Word Cloud + Stop words    │   │  LSTM / BiLSTM → Dense → Softmax │
    │  • Emoji module               │   │                │                 │
    │  • Clustering + Sentiment     │   │                ▼                 │
    │            │                  │   │      Emotion Prediction          │
    └────────────┴──────────────────┘   └──────────────────────────────────┘
                        │                               │
                        ▼                               ▼
                 Output [1]: Visual insights    Output [2]: Detected emotion
```

`*` PushBullet API-based collection and Googletrans-based language conversion were part of the design/future scope — see [Future Scope](#-future-scope).

---

## 📂 Repository Structure

```
Exploratory-Data-Analysis-and-Emotion-Detection-Using-LSTM/
│
├── README.md                        # This file
├── LSTM.png                         # LSTM architecture illustration
├── Final Report.pdf                 # Complete project report
├── Project_PPT.pdf                  # Presentation (PDF)
├── final_review.pptx                # Final review presentation (PPTX)
│
├── Exploratory Data Analysis/       # MODULE 1 & 2 — Pre-processing + EDA dashboard
│   ├── app.py                       # Streamlit dashboard (entry point)
│   ├── preprocessor.py              # Regex parsing, format detection, feature engineering
│   ├── helper.py                    # 35 analysis/plotting helper functions
│   ├── requirements.txt             # Pinned Python dependencies
│   ├── setup.sh                     # Heroku/Streamlit server config
│   ├── Procfile                     # Deployment process definition
│   ├── Laila-Regular.ttf            # Unicode/Devanagari font for word clouds
│   ├── stop_hinglish.txt            # Hinglish stop-words
│   ├── configs/
│   │   └── stopwords/               # 45 stop-word files (hinglish, hindi, arabic, ...)
│   ├── custom_modules/
│   │   └── func_analysis.py         # Alternate/legacy analysis helpers
│   └── processor/
│       └── common/configure.py      # Streamlit CSS / footer configuration
│
└── Emotion Detection/               # MODULE 3 — Deep learning emotion detection
    ├── dataset/
    │   ├── train.txt                # 16,000 labelled sentences
    │   ├── val.txt                  #  2,000 labelled sentences
    │   └── test.txt                 #  2,000 labelled sentences
    └── LSTM_BiLSTM/
        ├── Model_LSTM.ipynb         # Conv1D + LSTM model (4 emotions)
        ├── Model_BILSTM.ipynb       # Stacked Bidirectional LSTM (6 emotions)
        ├── app.py                   # Streamlit sentiment demo (TextBlob + VADER)
        ├── labelEncoder.pickle      # Fitted scikit-learn LabelEncoder
        └── labelEncoder.jpg         # Label-encoding illustration
```

---
