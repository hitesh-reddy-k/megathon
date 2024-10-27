import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load and preprocess the dataset
data = pd.read_csv('mumbai2.csv')

def label_alert(value):
    if value > 150:
        return 2  # High alert
    elif value > 100:
        return 1  # Moderate alert
    else:
        return 0  # Low alert

monthly_data = []
monthly_labels = []

# Use correct column names as per the dataset
for _, row in data.iterrows():
    for month in ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']:
        monthly_data.append([row[month]])
        monthly_labels.append(label_alert(row[month]))

# Convert to numpy arrays
X = np.array(monthly_data)
y = np.array(monthly_labels)

# Normalize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert labels to categorical for TensorFlow
y_cat = to_categorical(y, num_classes=3)

# Split the dataset
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Build the model with regularization and dropout
model = Sequential([
    Dense(64, input_shape=(1,), activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes: High, Moderate, Low
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
model.fit(X_train, y_train_cat, epochs=100, batch_size=16, validation_data=(X_test, y_test_cat), callbacks=[early_stopping])


loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f'Test Accuracy: {accuracy:.2f}')


model.save('trained_alert_model2.h5')

def predict_alert(input_value, model, scaler):
    input_data = np.array([input_value]).reshape(-1, 1)
    input_data = scaler.transform(input_data)
    predictions = model.predict(input_data)
    alert_level = np.argmax(predictions, axis=1)[0]
    alert_mapping = {0: "Low Alert", 1: "Moderate Alert", 2: "High Alert"}
    return alert_mapping[alert_level]

model = load_model('trained_alert_model.h5')


next_month_rainfall = 120  
monthly_alert = predict_alert(next_month_rainfall, model, scaler)
print(f'Predicted Alert for Next Month: {monthly_alert}')

average_daily_rainfall = next_month_rainfall / 30  
daily_alert = predict_alert(average_daily_rainfall, model, scaler)
print(f'Simulated Alert for Next Day: {daily_alert}')
