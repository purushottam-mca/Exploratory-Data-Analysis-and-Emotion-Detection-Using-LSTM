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

## 📊 Datasets

Located in `Emotion Detection/dataset/`, each file is a **`sentence;emotion`** pair (English):

| File | Records | Purpose |
| --- | ---: | --- |
| `train.txt` | 16,000 | The model learns from this data |
| `val.txt` | 2,000 | Unbiased evaluation while tuning hyper-parameters |
| `test.txt` | 2,000 | Unbiased evaluation of the final model |
| **Total** | **20,000** | 6 emotion classes |

**Class distribution**

| Emotion | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| joy | 5,362 | 704 | 695 |
| sadness | 4,666 | 550 | 581 |
| anger | 2,159 | 275 | 275 |
| fear | 1,937 | 212 | 224 |
| love | 1,304 | 178 | 159 |
| surprise | 572 | 81 | 66 |

> The corpus is **imbalanced** (joy ≈ 9.4× surprise), which noticeably affects recall on the minority class `surprise` — see [Results](#-results).

---

## 🧠 Emotion Detection Models

### Model 1 — Conv1D + LSTM (`Model_LSTM.ipynb`)

Pre-processing: non-alphabetic character removal + lowercasing → filtering the data to **4 emotions** (`joy`, `sadness`, `anger`, `fear`) → `LabelEncoder` → `to_categorical` → `Tokenizer` → `pad_sequences(maxlen=256, truncating='pre')`.

| Layer | Configuration |
| --- | --- |
| Embedding | `vocab_size × 200` (vocab derived from the tokenizer, ≈14,980 words) |
| Dropout | 0.25 |
| Conv1D | 128 filters, kernel size 5, stride 1, `relu`, `padding='valid'` |
| MaxPooling1D | pool size 4 |
| LSTM | 128 units |
| Dense → Activation | 4 units → `softmax` |

- **Compile:** `categorical_crossentropy`, `Adam(learning_rate=0.005)`, metric `accuracy`
- **Fit:** `batch_size=256`, `epochs=30`, `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`
- **Total parameters:** **3,256,228**
- **Artefacts produced:** `Emotion Detection.h5`, `tokenizer.pickle`, `labelEncoder.pickle`

### Model 2 — Stacked Bidirectional LSTM (`Model_BILSTM.ipynb`)

Pre-processing: NLTK tokenization + English stop-word removal → **Word2Vec (300-d, `min_count=2`) trained on the corpus** → `Tokenizer(num_words=15000)` → `pad_sequences(maxlen=MAX_LEN)` → static embedding matrix built by combining **pre-trained fastText wiki-news-300d vectors** with the corpus Word2Vec model.

| Layer | Configuration |
| --- | --- |
| Embedding | 15,000 × 300, initialised with the embedding matrix, `trainable=False` |
| Bidirectional LSTM | 128 units, `return_sequences=True` |
| Dropout | 0.2 |
| Bidirectional LSTM | 256 units, `return_sequences=True` |
| Dropout | 0.2 |
| Bidirectional LSTM | 128 units, `return_sequences=False` |
| Dense | 6 units, `softmax` |

- **Compile:** `sparse_categorical_crossentropy`, `Adam`, metric `accuracy`
- **Fit:** `batch_size=8`, `epochs=20`, `EarlyStopping(monitor='val_loss', patience=3)`
- **Label mapping:** `anger:0, love:1, fear:2, joy:3, sadness:4, surprise:5`
- **Evaluation:** accuracy/loss curves, `classification_report`, confusion matrix (counts **and** row-normalised)

---

## 📈 Results

| Model | Emotions | Test Accuracy | Test Loss | Remarks |
| --- | :---: | :---: | :---: | --- |
| **Conv1D + LSTM** | 4 | **95.15 %** | 17.51 % | Validation accuracy peaked at ≈94.5 %; trained only on `joy, sadness, anger, fear` |
| **Stacked BiLSTM** | 6 | **92.80 %** | 14.97 % | Macro-avg F1 **0.88**, weighted-avg F1 **0.93** on all 6 emotions |

**BiLSTM per-class performance (test set, 2,000 samples)**

| Label | Emotion | Precision | Recall | F1-score | Support |
| :---: | --- | :---: | :---: | :---: | ---: |
| 0 | anger | 0.94 | 0.92 | 0.93 | 275 |
| 1 | love | 0.93 | 0.70 | 0.80 | 159 |
| 2 | fear | 0.86 | 0.96 | 0.91 | 224 |
| 3 | joy | 0.91 | 0.98 | 0.95 | 695 |
| 4 | sadness | 0.97 | 0.96 | 0.96 | 581 |
| 5 | surprise | 0.95 | 0.58 | 0.72 | 66 |
| | **accuracy** | | | **0.93** | **2,000** |
| | macro avg | 0.93 | 0.85 | 0.88 | 2,000 |
| | weighted avg | 0.93 | 0.93 | 0.93 | 2,000 |

**Observations**

- The **BiLSTM** model generalises across all six emotions and handles the under-represented classes better than the 4-class LSTM variant, but minority-class recall (`surprise` 0.58, `love` 0.70) remains the weakest area.
- The 4-class LSTM shows very high training accuracy (≈99 %) with a lower validation accuracy (≈94 %), indicating mild over-fitting; **Dropout + Early Stopping** were used to control it.

---

## 🛠 Tech Stack

| Category | Tools |
| --- | --- |
| Language | Python 3.7 |
| Dashboard / Web App | Streamlit 1.10, Streamlit AgGrid |
| Data handling | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly (Express & Graph Objects), Altair, WordCloud |
| NLP utilities | NLTK (tokenization, stop-words), `emoji`, URLExtract, TextBlob, VADER Sentiment |
| Deep Learning | TensorFlow / Keras (Embedding, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, EarlyStopping, ModelCheckpoint) |
| Classical ML / preprocessing | scikit-learn (LabelEncoder, train/test split, metrics) |
| Embeddings | Gensim Word2Vec, pre-trained fastText wiki-news-300d |
| Notebooks | Jupyter / IPython |
| Deployment | Procfile + `setup.sh` (Heroku-style Streamlit deployment) |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.7** (the project was developed on 3.7 — the notebooks record `Python 3.7.0`)
- `pip`
- Git

```bash
git clone git@github.com:purushottam-mca/Exploratory-Data-Analysis-and-Emotion-Detection-Using-LSTM.git
cd Exploratory-Data-Analysis-and-Emotion-Detection-Using-LSTM
```

### 1️⃣ Run the EDA Dashboard

```bash
cd "Exploratory Data Analysis"
pip install -r requirements.txt

# st_aggrid is imported by app.py but is not listed in requirements.txt
pip install st_aggrid

streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

> **Tip:** `requirements.txt` starts with the line `python==3.7.2`, which is not a valid PyPI requirement. If `pip install -r requirements.txt` fails on that line, remove it before installing.

### 2️⃣ Run the Emotion Detection Notebooks

```bash
cd "Emotion Detection/LSTM_BiLSTM"
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
jupyter notebook
```

Open **`Model_LSTM.ipynb`** first — it is the most self-contained notebook. The dataset is read with the **relative path `../dataset/*.txt`**, so the notebooks must be launched from inside `Emotion Detection/LSTM_BiLSTM/`.

For **`Model_BILSTM.ipynb`** you additionally need:

```bash
pip install gensim nltk
```

and a `model_wiki.txt` file in the notebook's folder — the fastText `wiki-news-300d` vectors converted to word2vec text format. It is **not committed to this repository** because of its size; generate it by un-commenting and running the first cells of the notebook:

```python
import gensim.downloader as api
model_wiki = api.load('fasttext-wiki-news-subwords-300')
model_wiki.save_word2vec_format('model_wiki.txt', binary=False)
```

### 3️⃣ Run the Sentiment Demo App

```bash
cd "Emotion Detection/LSTM_BiLSTM"
pip install streamlit textblob vaderSentiment pandas altair
streamlit run app.py
```

### 🔁 Model artefacts

The trained weights are intentionally **not committed** (large binaries). Re-running `Model_LSTM.ipynb` regenerates:

- `Emotion Detection.h5` — trained Keras model
- `tokenizer.pickle` — fitted Keras `Tokenizer` (needed to convert new text to sequences)
- `labelEncoder.pickle` — fitted `LabelEncoder` (already committed for reference)

---

## 📱 How to Export a WhatsApp Chat

**Method 1 — via mobile (used by this project)**

1. Open WhatsApp and go to the individual or group chat.
2. Tap **⋮ More options** → **More** → **Export chat**.
3. Choose **Without media** (the EDA app expects a `.txt` file).
4. Save/share the `.txt` file to your computer.

**Method 2 — via PushBullet API** *(design/future scope)*

Export the chat to the PushBullet app, create an access token under *Settings → Account → Access Token*, then download the latest push to a local file. This automation is described in the project presentation but is not implemented in the current codebase.

> ℹ️ The pre-processor expects WhatsApp timestamps with an **AM/PM** marker (e.g. `12/05/21, 9:45 pm - User: message`). Chats exported in 24-hour format may not be parsed correctly.

---

## 📑 Project Documents

| File | Description |
| --- | --- |
| [`Final Report.pdf`](Final%20Report.pdf) | Complete final-year project report (problem statement, literature survey, design, implementation, results) |
| [`Project_PPT.pdf`](Project_PPT.pdf) | Presentation deck (PDF) |
| [`final_review.pptx`](final_review.pptx) | Final review presentation (editable PPTX) |
| [`LSTM.png`](LSTM.png) | LSTM architecture illustration |

---

