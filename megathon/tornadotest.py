import numpy as np
import pandas as pd
import joblib  # Import joblib for loading scaler
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk  # Import ttk for combobox

# Load the saved model and scaler
model = load_model('trained_tornado_alert_model.h5')
scaler = joblib.load('scaler.joblib')

# Load tornado data
data = pd.read_csv('tornado.csv')  # Ensure this is the path to your test data

# Function to map alert level to descriptive text
def get_alert_label(alert_level):
    alert_mapping = {0: "Low Tornado Risk", 1: "Moderate Tornado Risk", 2: "High Tornado Risk"}
    return alert_mapping.get(alert_level, "Unknown")

# Create the GUI
root = tk.Tk()
root.title("Tornado Alert Prediction")

# Create a label and combobox for state selection
state_label = tk.Label(root, text="Select a State:", font=("Arial", 14))
state_label.pack(pady=10)

# Create a combobox for state selection
state_combobox = ttk.Combobox(root, values=data['States'].unique().tolist(), width=30)
state_combobox.pack(pady=5)
state_combobox.set("Select a State")  # Set default text

# Function to display prediction for the selected state
def display_prediction():
    selected_state = state_combobox.get()
    if selected_state:
        # Find the row for the selected state
        row = data[data['States'] == selected_state].iloc[0]
        
        recent_5_years_data = row[1:6].values.astype(float) 
        input_data = np.array(recent_5_years_data).reshape(1, -1)

        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)
        alert_level = np.argmax(prediction, axis=1)[0]
        accuracy = np.max(prediction) * 100

        alert_message = get_alert_label(alert_level)

        # Clear previous labels and display new prediction
        for widget in prediction_frame.winfo_children():
            widget.destroy()  # Clear previous predictions

        prediction_label = tk.Label(prediction_frame, text=f'Predicted Tornado Alert for {selected_state}: {alert_message} (Confidence: {accuracy:.2f}%)', font=("Arial", 12))
        prediction_label.pack(pady=5)

# Button to trigger prediction display
predict_button = tk.Button(root, text="Get Prediction", command=display_prediction)
predict_button.pack(pady=10)

# Frame to hold prediction results
prediction_frame = tk.Frame(root)
prediction_frame.pack(pady=10)

# Run the app
root.mainloop()
