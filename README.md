# 🎬 Age Rating Prediction for Streaming Platforms using Naive Bayes

A machine learning project that predicts **age ratings** (e.g., PG, R, TV-MA) for movies and TV shows using metadata and content descriptions from major streaming platforms.


## 📌 Overview

This project applies **Multinomial Naive Bayes**, a text classification algorithm, to predict age ratings for shows and movies across Netflix, Disney+, Hulu, and Amazon Prime Video. By automating the classification based on descriptions, the model assists in improving viewer safety and content moderation.


## 🗂 Dataset

**Source:** Kaggle  
- [Netflix](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
- [Disney+](https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows)  
- [Amazon Prime](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows)  
- [Hulu](https://www.kaggle.com/datasets/shivamb/hulu-movies-and-tv-shows)

📌 These datasets contain:  
- `title`, `type`, `description`, `cast`, `genre`, `release_year`, `duration`, `rating`, and more  
- The `rating` column is the **target variable**

*Note:* Raw data files are excluded due to size. See [`data/README.md`](data/README.md) for download instructions.



## 🧠 Methodology

### Preprocessing
- Missing values handled
- Text normalization (lowercasing, punctuation removal, stopword removal)
- Feature selection focused on `description` and other key metadata

### Feature Extraction
- `CountVectorizer` and `TF-IDF` used to transform descriptions into feature vectors

### Model
- `MultinomialNB` (from `sklearn.naive_bayes`)
- Trained on processed description data
- Tested using multiple evaluation metrics



## 📈 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**



## 🛠 Tools Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn



