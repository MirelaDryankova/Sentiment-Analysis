import string
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import torch
from wordcloud import WordCloud
# Preprocessing and evaluation
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l1, l2
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.preprocessing import LabelBinarizer
import torch.nn.functional as F

#Model
from sklearn.linear_model import LogisticRegression
from utils import *
def rating(score):
    if score > 3:
        return 'Positive'
    elif score == 3:
        return 'Netral'
    else:
        return 'Negative'


@st.cache_data
def data():
    
    df = pd.read_csv('./cleaned_df.csv')
    df = df.dropna(subset=['Cleaned_Review', 'Raw_Rating'])
    df['Raw_Rating'] = df['Raw_Rating'].apply(lambda x: rating(x))

    X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Review'], df['Raw_Rating'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@st.cache_data
def bert_predict(text):
    # Load the BERT model using PyTorch
    model = torch.load('bert.pth', map_location=torch.device('cpu'))
    model.eval()

    # Define and fit LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(['Positive', 'Neutral', 'Negative'])

    # Preprocess the input text
    cleaned_text = cleaning(text)

    # Tokenize the input text
    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Perform inference using the BERT model
    outputs = model(**inputs)
    predictions = F.softmax(outputs.logits, dim=1)
    sentiment_label = lb.classes_[predictions.argmax(axis=1)][0]
    predicted_probability = predictions.max(axis=1)[0].item() 

    return sentiment_label, predicted_probability
# class Config():
#     seed_val = 17
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     epochs = 5
#     batch_size = 6
#     seq_length = 512
#     lr = 2e-5
#     eps = 1e-8
#     pretrained_model = 'bert-base-uncased'
#     test_size = 0.15
#     random_state = 42
#     add_special_tokens = True
#     return_attention_mask = True
#     pad_to_max_length = True
#     do_lower_case = False
#     return_tensors = 'pt'

# config = Config()

# @st.cache_data
# def bert_predict(text):
#     # Load BERT tokenizer and model
#     tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
#     model = BertForSequenceClassification.from_pretrained(config.pretrained_model, num_labels=3)
    
#     # Move the model to the specified device
#     model.to(config.device)
    
#     # Define and fit LabelBinarizer
#     lb = LabelBinarizer()
#     lb.fit(['Positive', 'Neutral', 'Negative'])

#     # Preprocess the input text
#     cleaned_text = cleaning(text)

#     # Tokenize the input text
#     inputs = tokenizer(
#         cleaned_text,
#         add_special_tokens=config.add_special_tokens,
#         return_attention_mask=config.return_attention_mask,
#         pad_to_max_length=config.pad_to_max_length,
#         do_lower_case=config.do_lower_case,
#         return_tensors=config.return_tensors
#     )

#     # Perform inference using the BERT model
#     with torch.no_grad():
#         model.eval()
#         inputs = {key: val.to(config.device) for key, val in inputs.items()}
#         outputs = model(**inputs)
    
#     predictions = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()

#     # Convert predictions to sentiment labels
#     sentiment_label = lb.classes_[predictions][0]

#     # Get the predicted probabilities
#     probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
#     print(probabilities)

#     # Get the probability corresponding to the predicted sentiment label
#     predicted_probability = probabilities[0, predictions[0]]

#     return sentiment_label, predicted_probability

# @st.cache_data
# def bert_predict(text):
#     # Load BERT tokenizer and model
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

#     # Define and fit LabelBinarizer
#     lb = LabelBinarizer()
#     lb.fit(['Positive', 'Neutral', 'Negative'])

#     # Preprocess the input text
#     cleaned_text = cleaning(text)

#     # Tokenize the input text
#     inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True)

#     # Perform inference using the BERT model
#     outputs = model(**inputs)
#     predictions = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()

#     # Convert predictions to sentiment labels
#     sentiment_label = lb.classes_[predictions][0]

#     probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
#     predicted_probability = probabilities[predictions[0]]

#     return sentiment_label, predicted_probability

@st.cache_data
def ml_prep():
    X_train, _, y_train, _ = data()

    # ML
    tfid = TfidfVectorizer()
    train_tfid_matrix = tfid.fit_transform(X_train)

    log = LogisticRegression(max_iter=1000)
    log.fit(train_tfid_matrix, y_train)

    return tfid, log

@st.cache_data(hash_funcs={tf.keras.models.Sequential: id})
def dl_prep(model_path='/home/mirela/Documents/transport/tripadvisor/dl_model_25.h5'):
    X_train, _, y_train, _ = data()

    # DL 
    tokenizer = Tokenizer(num_words=50000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    total_word = len(tokenizer.word_index) + 1

    if model_path:
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.Sequential([
            Embedding(total_word, 8),
            Bidirectional(LSTM(16)),
            Dropout(0.5),
            Dense(8, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])

    model.summary()
    
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(y_train)

    return tokenizer, model, lb

# def cleaning(text):
#     #remove punctuations and uppercase
#     clean_text = text.translate(str.maketrans('','',string.punctuation)).lower()
    
#     #remove stopwords
#     clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
    
#     #lemmatize the word
#     sentence = []
#     for word in clean_text:
#         lemmatizer = WordNetLemmatizer()
#         sentence.append(lemmatizer.lemmatize(word, 'v'))

#     return ' '.join(sentence)

def cleaning(text):
    text = text.lower()
    text = text.replace('’', "'")

    text = expand_contractions(text)
    text = replace_emojis(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text, './gist_stopwords.txt')
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = remove_words_with_numbers(text)
    text = stem_and_lemmatize(text)

    return text

# Logistic Regression
def ml_predict(text):
    tfid, log = ml_prep()

    clean_text = cleaning(text)
    tfid_matrix = tfid.transform([text])
    pred_proba = log.predict_proba(tfid_matrix)
    idx = np.argmax(pred_proba)
    pred = log.classes_[idx]
    
    return pred, pred_proba[0][idx]

# Deep Neural Network
def dl_predict(text):
    tokenizer, model, lb = dl_prep()

    # clean_text = cleaning(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq)

    pred = model.predict(padded)
    idx = np.argmax(pred)
    # Get the label name back
    result = lb.inverse_transform(pred)[0]
    
    return result, pred[0][idx]

def main():
    html_temp = '''
    <h1 style="font-family: Trebuchet MS; padding: 12px; font-size: 30px; color: #c9533e; text-align: center;
    line-height: 1.25;">Sentiment Analysis<br>
    <span style="color: #f6f1e4; font-size: 30px">on</b></span><br>
    <span style="color: #f6f1e4; font-size: 36px"><b>TripAdvisor Hotel Reviews</b></span><br>
    </h1>
    '''

    st.markdown(html_temp, unsafe_allow_html=True)

    models = st.radio('Select Model', ('Logistic Regression', 'Deep Neural Network', 'BERT'))

    text = st.text_area('Write a Review', '')

    if st.button('Analyze'):
        if models == 'Logistic Regression':
            label, probability = ml_predict(text)
            print(label)
            print(probability)
        elif models == 'Deep Neural Network':
            label, probability = dl_predict(text)
            print(label)
            print(probability)
        else:
            label, probability = bert_predict(text)
            print(label)
            print(probability)

        if label == 'Positive' or label == 'P':
            st.info('Probability: {:.2f}'.format(probability))
            st.success('The sentiment for this particular review is Good')
        elif label == 'Negative' or label == 'N':
            st.info('Probability: {:.2f}'.format(probability))
            st.error('The sentiment for this particular review is Bad')
        else:
            st.info('Probability: {:.2f}'.format(probability))
            st.warning('The sentiment for this particular review is neither Good nor Bad')

# @st.cache_data
# def load_data():
#     # Load your dataset here (replace 'your_dataset.csv' with your actual file path)
#     df = pd.read_csv('./all_data.csv')
#     #df['Raw_Rating'] = df['Raw_Rating'].apply(lambda x: rating(x))
#     return df

# def generate_word_cloud(text):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot()

# def main():
#     st.title('TripAdvisor Comments Sentiment Analysis')

#     # Load TripAdvisor comments
#     file_path = 'tripadvisor_comments.csv'
#     df = load_data()

#     # Sidebar filters
#     st.sidebar.title('Filters')
#     min_rating = st.sidebar.slider('Minimum Rating', min_value=1, max_value=5, value=1)
#     max_rating = st.sidebar.slider('Maximum Rating', min_value=1, max_value=5, value=5)

#     # Filter comments based on rating
#     filtered_comments = df[(df['Rating'] >= min_rating) & (df['Rating'] <= max_rating)]

#     # Display filtered comments
#     st.subheader('Filtered Comments')
#     st.write(filtered_comments)

#     # Sentiment analysis
#     st.subheader('Sentiment Analysis')

#     # Analyze sentiment of individual comments
#     comment_index = st.number_input('Enter comment index', min_value=0, max_value=len(filtered_comments)-1, value=0)
#     comment = filtered_comments.iloc[comment_index]['Comment']
#     st.write('Selected Comment:', comment)

#     # Perform sentiment analysis using TextBlob
#     blob = TextBlob(comment)
#     sentiment_score = blob.sentiment.polarity
#     if sentiment_score > 0:
#         sentiment = 'Positive'
#     elif sentiment_score < 0:
#         sentiment = 'Negative'
#     else:
#         sentiment = 'Neutral'

#     st.write('Sentiment:', sentiment)
#     st.write('Sentiment Score:', sentiment_score)

if __name__ == '__main__':
    main()
