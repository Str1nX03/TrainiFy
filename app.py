from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '.'

MODEL_PATH = './models/trained_model.pkl'

# Utility: Train and return best model
def train_model(df, task):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1] if task != 'clustering' else None

    models = []
    if task == 'regression':
        models = [
            ('LinearRegression', LinearRegression()),
            ('RandomForestRegressor', RandomForestRegressor())
        ]
    elif task == 'classification':
        models = [
            ('LogisticRegression', LogisticRegression(max_iter=1000)),
            ('RandomForestClassifier', RandomForestClassifier())
        ]
    elif task == 'clustering':
        model = make_pipeline(StandardScaler(), KMeans(n_clusters=3))
        model.fit(X)
        return model

    best_score = -1
    best_model = None
    for name, model in models:
        try:
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy' if task == 'classification' else 'r2')
            avg_score = scores.mean()
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        except Exception as e:
            print(f"Model {name} failed: {e}")

    if best_model:
        best_model.fit(X, y)
        return best_model
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded file as dataset.csv
        file = request.files['dataset']
        task = request.form['task']

        if file:
            df = pd.read_csv(file)
            df.to_csv('dataset.csv', index=False)

            # Train model
            model = train_model(df, task)
            if model:
                os.makedirs('models', exist_ok=True)
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(model, f)
                return render_template('index.html', download=True)

    return render_template('index.html', download=False)

@app.route('/download')
def download_model():
    return send_file(MODEL_PATH, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
