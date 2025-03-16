from flask import Flask, render_template, request
from google.cloud import storage
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64
import datetime
import hashlib
from scipy.cluster import hierarchy

app = Flask(__name__)

def generate_unique_filename(original_filename):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    hash_object = hashlib.md5(original_filename.encode())
    hash_str = hash_object.hexdigest()[:8]
    unique_filename = f"{timestamp}_{hash_str}_{original_filename}"
    return unique_filename

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading index.html: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'fpkm_data' not in request.files:
            return "No file part"
        file = request.files['fpkm_data']
        if file.filename == '':
            return "No selected file"
        bucket_name = 'new-rnaseq-analysis'
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(generate_unique_filename(file.filename))
        print("Uploading file:", file.filename)
        blob.upload_from_file(file)
        fpkm_data = blob.download_as_text()
        print("Raw data:", fpkm_data[:1000])  # Limit to first 1000 chars for readability
        fpkm_df = pd.read_csv(StringIO(fpkm_data), sep=',', index_col=0)
        print("fpkm_df shape:", fpkm_df.shape)
        print("fpkm_df head:", fpkm_df.head())
        if fpkm_df.empty or fpkm_df.shape[0] < 2 or fpkm_df.shape[1] < 1:
            return "Error: Uploaded file is empty or has insufficient data (need at least 2 rows and 1 column)."
        fpkm_df_log2 = fpkm_df.apply(lambda x: np.log2(x + 1))
        print("fpkm_df_log2 shape:", fpkm_df_log2.shape)
        row_linkage = hierarchy.linkage(fpkm_df_log2, method='ward')
        row_order = hierarchy.leaves_list(row_linkage)
        col_linkage = hierarchy.linkage(fpkm_df_log2.T, method='ward')
        col_order = hierarchy.leaves_list(col_linkage)
        fpkm_df_clustered = fpkm_df_log2.iloc[row_order, col_order]
        fig, ax = plt.subplots()
        sns.heatmap(fpkm_df_clustered, cmap='viridis', ax=ax)
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return render_template('result.html', img_base64=img_base64)
    except Exception as e:
        return f"Error processing upload: {str(e)}"

if __name__ == '__main__':
    app.run()

