from sqlalchemy import create_engine
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from flask import Flask, request, jsonify
import re
import numpy as np
from joblib import load
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch} starting")

    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch} finished")
        self.epoch += 1

app = Flask(__name__)

server = "localhost"
database = "martfury_shop_temp"
username = "root"
password = "123456"

engine = create_engine(f"mysql+pymysql://{username}:{password}@{server}/{database}")

try:
    query = 'SELECT * FROM products'
    df_products = pd.read_sql(query, engine)
except Exception as e:
    print(e)

features = ['name', 'author', 'publisher']

def combine_features(row):
    return ' '.join([str(row[feature]) for feature in features])

df_products['combined_features'] = df_products.apply(combine_features, axis=1)

# Tạo ma trận TF-IDF
tfidf_vectorizer_recommend = TfidfVectorizer()
tfidf_vectorizer_recommend.fit(df_products['combined_features'])
tfidf_matrix = tfidf_vectorizer_recommend.transform(df_products['combined_features']) 

# Tính ma trận độ tương tự cosine
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Load mô hình FastText
model = FastText.load('fasttext_model.bin') 

# Hàm tiền xử lý dữ liệu văn bản
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text

# Hàm tìm kiếm sản phẩm sử dụng mô hình FastText và TF-IDF
def search_book(query):
    query = preprocess_text(query)
    query_vec = np.mean([model.wv[word] for word in query.split() if word in model.wv], axis=0).reshape(1, -1)

    tfidf_vectorizer_search = TfidfVectorizer(max_features=300)  
    tfidf_matrix_search = tfidf_vectorizer_search.fit_transform(df_products['name'])

    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix_search)

    filtered_books_indices = [i for i, score in enumerate(cosine_similarities[0]) if score > 0.1]

    return df_products.iloc[filtered_books_indices]['product_id'].tolist()

# Endpoint lấy các sản phẩm được đề xuất
@app.route('/recommend', methods=['GET'])
def recommend():
    result = []

    product_id = request.args.get('product_id')
    number = 12
    if product_id is None:
        return "Yêu cầu nhập mã sản phẩm", 400

    product_id = int(product_id)
    if product_id not in df_products['product_id'].values:
        return "Không tìm thấy sản phẩm", 404

    similar_products = list(enumerate(cosine_sim_matrix[product_id]))
    sorted_similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)

    for i in range(1, number + 1):
        result.append(int(sorted_similar_products[i][0]))

    response = app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

    return response

# Endpoint tìm kiếm sản phẩm theo tên
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if query is None:
        return "Yêu cầu nhập chuỗi truy vấn", 400

    result = search_book(query)

    response = app.response_class(
        response=json.dumps(result, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

    return response

if __name__ == '__main__':
    app.run(port=5555, debug=True)
