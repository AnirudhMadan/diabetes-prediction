import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import joblib






""" def train_and_save_dl_model():
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome')
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

   # Define a simpler model
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    checkpoint = ModelCheckpoint('saved_models/best_dl_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

    model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=8,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )

    joblib.dump(scaler, 'saved_models/scaler.pkl')

    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print("Deep Learning model and scaler trained and saved successfully!")


    os.makedirs('saved_models', exist_ok=True)
    model.save('saved_models/diabetes_dl_model.h5')
    joblib.dump(scaler, 'saved_models/scaler.pkl') """


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score
from .preprocessing import preprocess_diabetes_data

def train_and_save_dl_model():
    # Load and clean data
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome')
    y = df['Outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2)
    
    # Scaled data
    X_train, X_test, y_train, y_test, scaler = preprocess_diabetes_data('diabetes.csv')

    

    # Define model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),

        Dense(64, kernel_regularizer=l2(0.0005)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    # Compile
    model.compile(
        optimizer=Nadam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    os.makedirs('saved_models', exist_ok=True)
    checkpoint = ModelCheckpoint('saved_models/best_dl_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

    # Train
    model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

    # Save scaler
    joblib.dump(scaler, 'saved_models/scaler.pkl')

    print("Model and scaler saved successfully.")

