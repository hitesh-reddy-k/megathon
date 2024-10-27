import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load the test dataset
test_data = pd.read_csv('hyderabad_rainfall_2014_2023.csv')  # Replace with the path to your test dataset

# Define thresholds for alert labels (same as in training)
def label_alert(value):
    if value > 150:
        return 2  # Red alert
    elif value > 100:
        return 1  # Yellow alert
    else:
        return 0  # Green (no alert)

# Convert numeric label to string for display
def get_alert_label(label):
    if label == 2:
        return "Red Alert"
    elif label == 1:
        return "Yellow Alert"
    else:
        return "Green"

# Reshape test data for monthly prediction
monthly_test_data = []
monthly_test_labels = []

for _, row in test_data.iterrows():
    for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
        monthly_test_data.append([row[month]])
        monthly_test_labels.append(label_alert(row[month]))

# Convert to numpy arrays
X_test = np.array(monthly_test_data)
y_test = np.array(monthly_test_labels)

# Normalize the feature data using the same scaler as the training data
scaler = StandardScaler()
# You should load the scaler that was fit on the training data instead of fitting it here
# For demonstration, we are fitting it here again, but it’s recommended to use the same scaler
X_test = scaler.fit_transform(X_test)

# Convert labels to categorical
y_test_cat = to_categorical(y_test, num_classes=3)

# Load the trained model
try:
    model = load_model('trained_alert_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")

# Evaluate the model on the test data
try:
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f'Test Accuracy: {accuracy:.2f}')
except Exception as e:
    print(f"Error during evaluation: {e}")

# Make predictions
predictions = model.predict(X_test)

# Convert predicted probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Print actual and predicted alert zones
for i in range(len(predicted_labels)):
    actual_label = get_alert_label(y_test[i])
    predicted_label = get_alert_label(predicted_labels[i])
    print(f'Month {i+1}: Actual - {actual_label}, Predicted - {predicted_label}')
