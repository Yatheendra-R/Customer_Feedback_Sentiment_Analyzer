# 🧠 Customer Feedback Sentiment Analyzer

An end-to-end Natural Language Processing (NLP) project for sentiment analysis of customer reviews. This project explores both traditional Machine Learning and Deep Learning approaches, comparing their performance on review classification tasks.

The system classifies reviews as **Positive** or **Negative** using advanced text preprocessing, feature engineering, and multiple sentiment classification models.

---

## 🚀 Features

- Text Cleaning and Normalization
- Contraction Expansion
- HTML and URL Removal
- Custom Stopword Handling
- Negation Propagation
- POS-Aware Lemmatization
- TF-IDF Feature Extraction
- GloVe Pretrained Word Embeddings
- Attention Mechanism
- Interactive Streamlit Web Application

---

## 🏗️ Project Architecture

```text
User Review
     ↓
Text Preprocessing
     ↓
Feature Extraction
 ├── TF-IDF
 └── GloVe Embeddings
     ↓
Classification Models
 ├── Logistic Regression
 ├── Linear SVM (LinearSVC)
 ├── LSTM
 ├── Stacked BiLSTM
 └── Attention-Based BiLSTM
     ↓
Sentiment Prediction
     ↓
Streamlit Web Interface
```

---

## 🤖 Models Implemented

### Traditional Machine Learning
- Logistic Regression
- Linear Support Vector Machine (LinearSVC)

### Deep Learning
- LSTM
- Stacked Bidirectional LSTM (BiLSTM)
- Attention-Based Stacked BiLSTM

### Embedding Techniques
- TF-IDF Vectorization
- GloVe Pretrained Embeddings

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- PyTorch
- NLTK
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn

---

## 📂 Repository Structure

```text
Customer_Feedback_Sentiment_Analyzer/
│
├── APP/
├── Basic/
├── Documentation/
├── Main/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📥 Trained Models

Due to GitHub file size limitations, trained model files are hosted separately on Hugging Face.

### Hugging Face Repository

https://huggingface.co/Yathii/customer-feedback-sentiment-analyzer

Download the contents of the `Saved/` directory and place them in:

```text
Saved/
```

---

## ▶️ Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Streamlit Application

```bash
streamlit run Final_APP.py
```

---

## ⚠️ Limitations

While the models achieve strong performance, sentiment analysis remains challenging in certain scenarios:

- Sarcasm
- Complex Context Understanding
- Implicit Sentiment
- Contrastive Statements
- Negation-Based Edge Cases

Example:

```text
I highly recommend you not to buy this product.
```

---

## 🔮 Future Improvements

- Transformer-based Models (BERT, RoBERTa)
- Multi-Class Sentiment Classification
- Emotion Detection
- Explainable AI Techniques
- Real-Time API Deployment

---

## 👨‍💻 Author

Yatheendra R

B.Tech Computer Science Student | AI/ML Enthusiast

GitHub: https://github.com/Yatheendra-R
