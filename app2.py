from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

from utils.preprocessing import preprocess_diabetes_data
from utils.dl_model_loader import load_dl_model_and_scaler
from utils.dl_model_trainer import train_and_save_dl_model

from utils.qml import load_qml_model, train_and_save_qml_model, predict_qml
from flask import redirect, request
import os

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv('diabetes.csv')

# Precompute median values by outcome
median_by_outcome = df.groupby('Outcome').median().round(2)
median_table = median_by_outcome.transpose().reset_index()
median_table.columns = ['Feature', 'Non-Diabetic (0)', 'Diabetic (1)']

X = df.drop(columns='Outcome')
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test= train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Dictionary of classifiers
models = {
    'SVM': SVC(kernel='linear'),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB()
}

# Train all traditional ML models
trained_models = {}
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    preds = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, preds)

# --- Load or Train Deep Learning model ---
try:
    dl_model, dl_scaler = load_dl_model_and_scaler()
    print("Loaded Deep Learning model successfully.")
except (FileNotFoundError, OSError) as e:
    print(f"Model not found. Training new Deep Learning model. Reason: {e}")
    train_and_save_dl_model()  # Train and save model
    dl_model, dl_scaler = load_dl_model_and_scaler()  # Load after training

# Evaluate DL model
X_train_dl, X_test_dl, y_train_dl, y_test_dl, scaler = preprocess_diabetes_data("diabetes.csv")

X_train_dl_scaled = dl_scaler.transform(X_train_dl)
X_test_dl_scaled = dl_scaler.transform(X_test_dl)

dl_loss, dl_acc = dl_model.evaluate(X_test_dl_scaled, y_test_dl, verbose=0)

models['Deep Learning'] = dl_model
trained_models['Deep Learning'] = dl_model
accuracies['Deep Learning'] = dl_acc

try:
    qml_model = load_qml_model()
    print("Loaded Quantum Machine Learning model successfully.")
except (FileNotFoundError, OSError) as e:
    print(f"QML model not found. Training new Quantum ML model. Reason: {e}")
    train_and_save_qml_model()  # Train and save QML model
    qml_model = load_qml_model()  # Load after training

 # Evaluate QML model
qml_preds = predict_qml(qml_model, X_test)
qml_acc = accuracy_score(y_test, qml_preds)

models['Quantum ML'] = qml_model
trained_models['Quantum ML'] = qml_model
accuracies['Quantum ML'] = qml_acc 


TOKEN_FILE = "auth_token.txt"

def get_valid_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return f.read().strip()
    return None

@app.before_request
def check_auth_token():
    # Allow static files and favicon.ico
    if request.path.startswith('/static') or request.path == '/favicon.ico':
        return  # Let Flask serve static content normally

    valid_token = get_valid_token()
    incoming_token = request.args.get('auth_token')

    if not valid_token or incoming_token != valid_token:
        return redirect('http://localhost:8501')  # Redirect to Streamlit login


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model = None
    model_accuracy = None
    accuracy_plot = None

    # Helper function for model prediction
    def get_prediction(model, selected_model, input_data):
        if selected_model == 'Deep Learning':
            return int(model.predict(input_data)[0][0] > 0.5)
        elif selected_model == 'Quantum ML':
            return predict_qml(model, input_data)[0]
            #return predict_qml_qiskit(model, input_data)[0]
        else:
            return model.predict(input_data)[0]

    if request.method == 'POST':
        try:
            action = request.form.get('action')

            if action == 'dl_predict':
                selected_model = 'Deep Learning'
            elif action == 'qml_predict':
                selected_model = 'Quantum ML'
            else:
                selected_model = request.form.get('model')
    

            if selected_model == 'Accuracy Stats':
                # Plotting accuracy
                plt.figure(figsize=(7, 3))
                bars = plt.barh(list(accuracies.keys()), list(accuracies.values()), color='skyblue')
                plt.xlabel('Accuracy')
                plt.title('Accuracy of ML Models')
                plt.xlim(0, 1.1)

                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                             f"{width * 100:.2f}%", va='center', fontsize=12)

                plot_path = os.path.join('static', 'accuracy_plot.png')
                plt.tight_layout(pad=2.0)
                plt.savefig(plot_path)
                plt.close()
                accuracy_plot = plot_path
                prediction = "Accuracy Stats selected - See comparison below"

            else:
                # Collect and scale input features
                input_features = [float(request.form[f]) for f in [
                    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
                ]]
                input_data = scaler.transform([input_features])

                model = trained_models[selected_model]
                result = get_prediction(model, selected_model, input_data)

                prediction = "Diabetic" if result == 1 else "Not Diabetic"
                model_accuracy = f"{accuracies[selected_model] * 100:.2f}%"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html',
                           prediction=prediction,
                           selected_model=selected_model,
                           model_accuracy=model_accuracy,
                           accuracy_plot=accuracy_plot,
                           median_table=median_table.to_dict(orient='records'),
                           models=list(models.keys()))


if __name__ == '__main__':
    app.run(debug=True)
