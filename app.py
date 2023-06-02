from flask import Flask, render_template, request
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from gensim import corpora
import gensim
from textblob import TextBlob

app = Flask(__name__)

# Mikrofonu başlatma
r = sr.Recognizer()
mic = sr.Microphone()

# İngilizce durdurucu kelimeleri yükleme
stop_words = set(stopwords.words('english'))

# Snowball Stemmer'ı başlatma
stemmer = SnowballStemmer('english')

# Metni ön işleme
def preprocess(text):
    result = []
    lemmatizer = WordNetLemmatizer()
    for token in word_tokenize(text):
        if token.lower() not in stop_words and len(token) > 3:
            result.append(lemmatizer.lemmatize(token.lower()))
    return result

# Konu tahmini için LDA modelini eğitme
def predict_topic(input_text):
    processed_text = preprocess(input_text)
    dictionary = corpora.Dictionary([processed_text])
    bow_corpus = [dictionary.doc2bow(processed_text)]
    lda_model = gensim.models.LdaModel(bow_corpus, num_topics=10, id2word=dictionary, passes=10)
    top_topic = max(lda_model[bow_corpus[0]], key=lambda x: x[1])[0]
    return lda_model.print_topic(top_topic)

# Metnin duygu analizini yap
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['input_text']
    topic = predict_topic(user_input)
    top_topic = topic.split('+')[0].split('*')[1].replace('"', '').strip()
    sentiment = analyze_sentiment(user_input)
    
    return render_template('result.html', input_text=user_input, topic=topic, top_topic=top_topic, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
