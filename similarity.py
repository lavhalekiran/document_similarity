import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def calculate_similarity(document1, document2):
    processed_document1 = preprocess_text(document1)
    processed_document2 = preprocess_text(document2)

    # Calculate the count of similar words
    common_words = set(processed_document1.split()).intersection(processed_document2.split())
    similar_word_count = len(common_words)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([document1, document2])

    # Calculate the cosine similarity between the documents
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity_score, similar_word_count

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded files
        file1 = request.files['file1']
        file2 = request.files['file2']

        # Read the file contents
        document1 = file1.read().decode('utf-8')
        document2 = file2.read().decode('utf-8')

        # Calculate similarity
        similarity_score, similar_word_count = calculate_similarity(document1, document2)
        
        return render_template('result.html', similarity_score=similarity_score, similar_word_count=similar_word_count)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
