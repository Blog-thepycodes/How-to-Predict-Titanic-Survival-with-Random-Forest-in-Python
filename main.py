import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
 
 
 
 
# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)
 
 
# Preprocess the data
data = data[['Pclass', 'Age', 'Fare', 'Sex', 'Survived']]
data = data.dropna()
 
 
# Convert 'Sex' to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
 
 
# Create 'IsBaby' feature
data['IsBaby'] = (data['Age'] <= 1).astype(int)
 
 
# Features and target variable
X = data[['Pclass', 'Age', 'Fare', 'Sex', 'IsBaby']]
y = data['Survived']
 
 
# Split data into training and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(X, y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]
 
 
# Create and train the random forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
 
 
# Make predictions
y_pred = model.predict(X_test)
 
 
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
 
 
def predict_survival():
  try:
      pclass = int(combo_pclass.get())
      age = float(entry_age.get())
      fare = float(spin_fare.get())
      gender = combo_gender.get()
      is_baby = 1 if age <= 1 else 0
 
 
      # Map gender to numeric
      gender_numeric = 1 if gender == 'Female' else 0
 
 
      # Create a DataFrame with the same feature names as used during training
      input_data = pd.DataFrame([[pclass, age, fare, gender_numeric, is_baby]],
                                columns=['Pclass', 'Age', 'Fare', 'Sex', 'IsBaby'])
 
 
      # Print input data for verification
      print(f"Input Data: {input_data}")
 
 
      # Make the prediction
      prediction = model.predict(input_data)[0]
      print(f"Prediction: {prediction}")
      result = 'Survived' if prediction == 1 else 'Not Survived'
      messagebox.showinfo("Prediction Result", f"Prediction: {result}")
  except ValueError:
      messagebox.showerror("Input Error", "Please enter valid numerical values.")
 
 
def show_tree():
  # Plot the first decision tree in the forest with increased size
  fig, ax = plt.subplots(figsize=(20, 15))  # Increase size for better visibility
  tree.plot_tree(model.estimators_[0],
                 feature_names=['Pclass', 'Age', 'Fare', 'Sex', 'IsBaby'],
                 class_names=['Not Survived', 'Survived'],
                 filled=True, fontsize=10, ax=ax)
  ax.set_title('Decision Tree for Titanic Survival Prediction (Random Forest)')
 
 
  # Save the plot as a PNG file with higher resolution
  plt.savefig("decision_tree.png", dpi=300)
 
 
  # Embed the plot in Tkinter window
  canvas = FigureCanvasTkAgg(fig, master=window)
  canvas.draw()
  canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
 
 
def show_summary():
  summary_text = (
      f"Model Accuracy: {accuracy:.2f}\n\n"
      "Classification Report:\n"
      f"{report}\n\n"
      "Confusion Matrix:\n"
      f"{conf_matrix}"
  )
  messagebox.showinfo("Model Summary", summary_text)
 
 
# Tkinter window
window = tk.Tk()
window.title("Titanic Survival Predictor - The Pycodes")
window.geometry("600x700")
 
 
# Styling
style = ttk.Style()
style.configure('TLabel', font=('Helvetica', 12))
style.configure('TButton', font=('Helvetica', 12))
 
 
# Passenger Class dropdown menu
tk.Label(window, text="Passenger Class:", font=('Helvetica', 12)).pack(pady=5)
combo_pclass = ttk.Combobox(window, values=[1, 2, 3], state="readonly")
combo_pclass.pack(pady=5)
combo_pclass.set("Select Class")
 
 
# Gender dropdown menu
tk.Label(window, text="Gender:", font=('Helvetica', 12)).pack(pady=5)
combo_gender = ttk.Combobox(window, values=['Male', 'Female'], state="readonly")
combo_gender.pack(pady=5)
combo_gender.set("Select Gender")
 
 
# Age input
tk.Label(window, text="Age:", font=('Helvetica', 12)).pack(pady=5)
entry_age = tk.Entry(window)
entry_age.pack(pady=5)
 
 
# Fare dropdown menu (using Spinbox for a range)
tk.Label(window, text="Fare:", font=('Helvetica', 12)).pack(pady=5)
spin_fare = ttk.Spinbox(window, from_=0, to=500, increment=1, format="%.2f")
spin_fare.pack(pady=5)
 
 
# Buttons
btn_predict = ttk.Button(window, text="Predict Survival", command=predict_survival)
btn_predict.pack(pady=10)
 
 
btn_show_tree = ttk.Button(window, text="Show Decision Tree", command=show_tree)
btn_show_tree.pack(pady=10)
 
 
btn_summary = ttk.Button(window, text="Show Model Summary", command=show_summary)
btn_summary.pack(pady=10)
 
 
# Run the Tkinter event loop
window.mainloop()
