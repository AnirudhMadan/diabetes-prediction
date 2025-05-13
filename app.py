from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_diabetes_data
import os
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Input
    from keras.optimizers import Adam
    from keras.regularizer import l2
    from keras.callbacks import EarlyStopping

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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Dictionary of classifiers
models = {
    'SVM': SVC(kernel='linear'),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Train all models once
trained_models = {}
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    preds = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, preds)

X_train, X_test, y_train, y_test = preprocess_diabetes_data("diabetes.csv")

# Step 2: Apply Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define the Deep Learning model
def create_dl_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),  # Explicit Input layer (cleaner)
        Dense(128, activation='relu', kernel_regularizer=l2(0.002)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.002)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Step 4: Create and train the model using scaled inputs
dl_model = create_dl_model(X_train_scaled.shape[1])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

dl_model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=16,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Step 5: Evaluate the model on the scaled test set
dl_loss, dl_acc = dl_model.evaluate(X_test_scaled, y_test, verbose=0)

models['Deep Learning'] = dl_model
trained_models['Deep Learning'] = dl_model
accuracies['Deep Learning'] = dl_acc

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model = None
    accuracy_plot = None
    model_accuracy = None

    if request.method == 'POST':
        try:
            action = request.form.get('action')
            selected_model = 'Deep Learning' if action == 'dl_predict' else request.form.get('model')


            if selected_model == 'Accuracy Stats':
                plt.figure(figsize=(5, 2))
                bars = plt.barh(list(accuracies.keys()), list(accuracies.values()), color='skyblue')
                plt.xlabel('Accuracy')
                plt.title('Accuracy of ML Models')
                plt.xlim(0, 1.1)

                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                             f"{width * 100:.2f}%", va='center', fontsize=12)

                plot_path = os.path.join('static', 'accuracy_plot.png')
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                accuracy_plot = plot_path
                prediction = "Accuracy Stats selected - See comparison below"

            else:
                input_features = [float(request.form[f]) for f in [
                    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
                ]]
                input_data = scaler.transform([input_features])
                model = trained_models[selected_model]
                if selected_model == 'Deep Learning':
                    result = int(model.predict(input_data)[0][0] > 0.5)
                else:
                    result = model.predict(input_data)[0]
                prediction = "Diabetic" if result == 1 else "Not Diabetic"
                model_accuracy = f"{accuracies[selected_model] * 100:.2f}%"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html',
                           prediction=prediction,
                           selected_model=selected_model,
                           model_accuracy=model_accuracy,
                           accuracy_plot=accuracy_plot,
                           models=list(models.keys()),
                           median_table=median_table.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)