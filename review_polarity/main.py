import os
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load positive and negative movie reviews
positive_folder = "txt_sentoken/pos"
negative_folder = "txt_sentoken/neg"

positive_files = [os.path.join(positive_folder, f) for f in os.listdir(positive_folder) if os.path.isfile(os.path.join(positive_folder, f))]
negative_files = [os.path.join(negative_folder, f) for f in os.listdir(negative_folder) if os.path.isfile(os.path.join(negative_folder, f))]

positive_texts = []
negative_texts = []

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

for file_path in positive_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        # Lowercasing
        text = text.lower()
        # Removing punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        positive_texts.append(' '.join(lemmatized_tokens))

for file_path in negative_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        # Lowercasing
        text = text.lower()
        # Removing punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        negative_texts.append(' '.join(lemmatized_tokens))

# Combine positive and negative texts and labels
all_texts = positive_texts + negative_texts
all_labels = [1] * len(positive_texts) + [0] * len(negative_texts)

# Generate Word Cloud
all_texts_combined = ' '.join(all_texts)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_texts_combined)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Reviews')
plt.show()

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(all_texts)
y = all_labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifiers
classifiers = {
    "Support Vector Machine": SVC(kernel='linear'),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\nClassifier: {clf_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Plotting TF-IDF scores
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.sum(axis=0).A1
sorted_indices = tfidf_scores.argsort()[::-1]
top_n = 20  # Number of top features to display

top_features = [(feature_names[i], tfidf_scores[i]) for i in sorted_indices[:top_n]]
top_features.reverse()  # Reverse to display in descending order

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), [score for _, score in top_features], align='center', color='skyblue')
plt.yticks(range(top_n), [word for word, _ in top_features])
plt.xlabel('TF-IDF Score')
plt.ylabel('Top Words')
plt.title('Top TF-IDF Words in Movie Reviews')
plt.gca().invert_yaxis()
plt.show()
