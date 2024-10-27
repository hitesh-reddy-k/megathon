import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk  # Import ttk for combobox

# Load and preprocess the dataset
data = pd.read_csv('mumbai.csv')

def label_alert(value):
    if value > 150:
        return 2  # High alert
    elif value > 100:
        return 1  # Moderate alert
    else:
        return 0  # Low alert

# Prepare the data for predictions
monthly_data = []
labels = []

for _, row in data.iterrows():
    for month in ['Jan', 'Feb', 'Mar', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']:
        monthly_data.append([row[month]])
        labels.append(label_alert(row[month]))

# Convert to numpy arrays
X = np.array(monthly_data)
y = np.array(labels)

# Normalize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Load the trained model
model = load_model('trained_alert_model2.h5')

# Function to predict alerts
def predict_alert(input_value):
    input_data = np.array([input_value]).reshape(-1, 1)
    input_data = scaler.transform(input_data)
    predictions = model.predict(input_data)
    alert_level = np.argmax(predictions, axis=1)[0]
    alert_mapping = {0: "Low Alert", 1: "Moderate Alert", 2: "High Alert"}
    return alert_mapping[alert_level]

# Function to display alerts for the selected area
def display_alerts():
    selected_area = area_combobox.get()
    
    # Assuming we always have the same monthly rainfall prediction for Mumbai in this example
    next_month_rainfall = 120  # This would be replaced with actual predicted value logic
    monthly_alert = predict_alert(next_month_rainfall)
    average_daily_rainfall = next_month_rainfall / 30
    daily_alert = predict_alert(average_daily_rainfall)

    # Clear previous alerts
    for widget in prediction_frame.winfo_children():
        widget.destroy()

    # Display the results for the selected area
    location_label = tk.Label(prediction_frame, text=f"Location: {selected_area}", font=("Arial", 16))
    location_label.pack(pady=5)
    monthly_alert_label = tk.Label(prediction_frame, text=f"Predicted Alert for Next Month: {monthly_alert}", font=("Arial", 14))
    monthly_alert_label.pack(pady=5)
    daily_alert_label = tk.Label(prediction_frame, text=f"Predicted Alert for Next Day: {daily_alert}", font=("Arial", 14))
    daily_alert_label.pack(pady=5)

# Create the GUI
root = tk.Tk()
root.title("Rainfall Alert Prediction")

# Create a label and combobox for area selection
area_label = tk.Label(root, text="Select a Location:", font=("Arial", 14))
area_label.pack(pady=10)

# Create a combobox for area selection with "Mumbai"
area_combobox = ttk.Combobox(root, values=["Mumbai"], width=30)
area_combobox.pack(pady=5)
area_combobox.set("Mumbai")  # Set default text

# Button to get predictions for the selected area
predict_button = tk.Button(root, text="Get Prediction", command=display_alerts)
predict_button.pack(pady=10)

# Frame to hold prediction results
prediction_frame = tk.Frame(root)
prediction_frame.pack(pady=10)

# Evaluate the model to get accuracy
loss, accuracy = model.evaluate(X, tf.keras.utils.to_categorical(y, num_classes=3), verbose=0)

# Display model accuracy
accuracy_label = tk.Label(root, text=f"Model Accuracy: {accuracy:.2f}", font=("Arial", 14))
accuracy_label.pack(pady=10)

# Run the app
root.mainloop()
