import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("seattle-weather.csv")
features = df[['precipitation', 'temp_max', 'temp_min', 'wind']]     #creating of last features
target = df['weather']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features, target , test_size= 0.2,random_state= 42)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train,y_train)
predict_NB = NB.predict(x_test)
from sklearn import metrics
x_NB = metrics.accuracy_score(y_test,predict_NB)
print(x_NB)
accuracy = []
accuracy.append(x_NB)
import tkinter as tk
# Define the GUI window and input fields
window = tk.Tk()
window.title("Weather Predicter")
window.geometry("500x400")
title_label = tk.Label(window, text="Enter precipitation:")
title_label.config(font=("Arial", 16))
title_label.pack(pady=5)
title_entry1 = tk.Entry(window)
title_entry1.config(font=("Arial", 14))
title_entry1.pack(pady=5)
title_label2 = tk.Label(window, text="Enter Max Temp:")
title_label2.config(font=("Arial", 16))
title_label2.pack(pady=5)
title_entry2 = tk.Entry(window)
title_entry2.config(font=("Arial", 14))
title_entry2.pack(pady=5)
title_label3 = tk.Label(window, text="Enter Min Temp:")
title_label3.config(font=("Arial", 16))
title_label3.pack(pady=5)
title_entry3 = tk.Entry(window)
title_entry3.config(font=("Arial", 14))
title_entry3.pack(pady=5)
title_label4 = tk.Label(window, text="Enter Wind Speed:")
title_label4.config(font=("Arial", 16))
title_label4.pack(pady=5)
title_entry4 = tk.Entry(window)
title_entry4.config(font=("Arial", 14))
title_entry4.pack(pady=5)
def predict():
    precipitation=float(title_entry1.get())
    max_temp=float(title_entry2.get())
    min_temp=float(title_entry3.get())
    wind=float(title_entry4.get())
    data = np.array([[precipitation,max_temp,min_temp,wind]])
    print(data)
    prediction = NB.predict(data)
    print(prediction)
    output_label.config(text="Predicted Weather is: "+'\n'.join(prediction))
output_label = tk.Label(window, wraplength=400)
output_label.config(font=("Arial", 14))
output_label.pack(pady=5)
# Add the input fields and predict button to the window
predict_button = tk.Button(window, text="Predict Weather", command=predict)
predict_button.config(font=("Arial", 14))
predict_button.pack(pady=5)
window.mainloop()