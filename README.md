# Spam Detection System

A machine learning project for classifying SMS/email messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

---

## 📂 Dataset

* **File:** `spam.csv`
* Contains two columns:

  * `label`: `ham` (not spam) or `spam`
  * `message`: the raw text message
* Encoding: `latin-1`

---

## 🔄 Workflow

### 1. Data Preprocessing

* Convert text to lowercase
* Remove punctuation
* Tokenize words
* Remove stopwords
* Apply stemming (Porter Stemmer)
* Store results in a new column `clean_message`

### 2. Feature Engineering

* Use **Bag of Words (CountVectorizer)**
* Limit vocabulary to 5000 most frequent words

### 3. Train-Test Split

* 80% training set
* 20% test set
* Stratified split to preserve spam/ham ratio

### 4. Model Training

* Algorithm: **Multinomial Naive Bayes**
* Train model on vectorized text

### 5. Evaluation

* **Metrics:** Precision, Recall, F1-score, Accuracy
* **Confusion Matrix:** Visualized with seaborn heatmap

### 6. Custom Testing

* Reads messages from `email.txt`
* Preprocesses text
* Predicts Spam/Ham for each message

---

## 📊 Results

* High accuracy with Naive Bayes
* Balanced precision & recall for Spam and Ham classes
* Effective for real-world spam detection tasks

*(Exact scores will depend on dataset split.)*

---

## ⚙️ How to Run

1. Install dependencies:

   ```bash
   pip install pandas scikit-learn nltk seaborn matplotlib
   ```

2. Download NLTK resources (only once):

   ```python
   import nltk
   nltk.download('all')
   ```

3. Place dataset file `spam.csv` in the working directory.

4. Run the Jupyter Notebook:

   ```bash
   jupyter notebook Spam_Detection_System.ipynb
   ```

5. To test on custom messages:

   * Create a file `email.txt`
   * Add one message per line
   * Run the final cells to see predictions

---

## 🛠️ Tools & Libraries

* Python 3
* Pandas
* Scikit-learn
* NLTK
* Seaborn & Matplotlib

---

## 📌 Future Improvements

* Try advanced models (Logistic Regression, SVM, Random Forest)
* Use **TF-IDF vectorization** instead of Bag of Words
* Apply deep learning (LSTM, BERT) for better performance
* Build a web app for real-time spam detection

---

## ✨ Author

Developed as a **Spam Detection System** project using NLP + Machine Learning.
