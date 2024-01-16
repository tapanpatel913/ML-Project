import numpy as np
import pandas as pd
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load the dataset
df = pd.read_csv("seattle-weather.csv")

# Select features and target
features = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
target = df['weather']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Naive Bayes model
NB = GaussianNB()
NB.fit(x_train, y_train)

# Function to predict weather based on user input
def predict():
    precipitation = float(title_entry1.get())
    max_temp = float(title_entry2.get())
    min_temp = float(title_entry3.get())
    wind = float(title_entry4.get())

    # Create a NumPy array for the input data
    data = np.array([[precipitation, max_temp, min_temp, wind]])

    # Make the prediction
    prediction = NB.predict(data)

    # Update the output label with the predicted weather
    output_label.config(text="Predicted Weather is: " + '\n'.join(prediction))

# Create the GUI window
window = tk.Tk()
window.title("Weather Predictor")
window.geometry("500x400")

# Create and pack input fields
def create_entry_label(label_text):
    label = tk.Label(window, text=label_text)
    label.config(font=("Arial", 16))
    label.pack(pady=5)
    entry = tk.Entry(window)
    entry.config(font=("Arial", 14))
    entry.pack(pady=5)
    return entry

title_entry1 = create_entry_label("Enter precipitation:")
title_entry2 = create_entry_label("Enter Max Temp:")
title_entry3 = create_entry_label("Enter Min Temp:")
title_entry4 = create_entry_label("Enter Wind Speed:")

# Create and pack output label
output_label = tk.Label(window, wraplength=400)
output_label.config(font=("Arial", 14))
output_label.pack(pady=5)

# Create and pack predict button
predict_button = tk.Button(window, text="Predict Weather", command=predict)
predict_button.config(font=("Arial", 14))
predict_button.pack(pady=5)

# Start the GUI loop
window.mainloop()
