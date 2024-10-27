import pandas as pd
import numpy as np
import joblib  # Import joblib for saving scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv('tornado.csv')

# Function to label tornado alert levels
def label_alert(value):
    if value > 150:
        return 2  
    elif value > 100:
        return 1
    else:
        return 0 
tornado_data = data.drop(columns=['States'])

# Prepare sequences for model training
X = []
y = []

for _, row in tornado_data.iterrows():
    state_data = row.values
    alert_labels = [label_alert(val) for val in state_data]
    X.extend([state_data[i:i+5] for i in range(len(state_data)-5)])
    y.extend(alert_labels[5:])

# Convert to numpy arrays and normalize
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_cat = to_categorical(y, num_classes=3)

# Split dataset
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# Build and compile the model
model = Sequential([
    Dense(128, input_shape=(5,), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping and adaptive learning rate
model.fit(
    X_train, y_train_cat,
    epochs=100,  # Increased to allow callbacks to work effectively
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save the trained model and scaler
model.save('trained_tornado_alert_model_optimized.h5')
joblib.dump(scaler, 'scaler.joblib')

# Evaluate and print accuracy
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f'Test Accuracy: {accuracy:.2f}')
print("Optimized model and scaler saved successfully.")
