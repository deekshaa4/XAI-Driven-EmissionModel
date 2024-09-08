import pandas as pd
import numpy as np
import joblib
import random
import lime
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
from tkinter import *
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from PIL import Image,ImageTk
from warnings import filterwarnings
filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('PROCESSED.csv')
makes=['ACURA', 'ALFA ROMEO', 'ASTON MARTIN', 'AUDI', 'BENTLEY', 'BMW',
       'BUICK', 'CADILLAC', 'CHEVROLET', 'CHRYSLER', 'DODGE', 'FIAT',
       'FORD', 'GMC', 'HONDA', 'HYUNDAI', 'INFINITI', 'JAGUAR', 'JEEP',
       'KIA', 'LAMBORGHINI', 'LAND ROVER', 'LEXUS', 'LINCOLN', 'MASERATI',
       'MAZDA', 'MERCEDES-BENZ', 'MINI', 'MITSUBISHI', 'NISSAN',
       'PORSCHE', 'RAM', 'ROLLS-ROYCE', 'SCION', 'SMART', 'SRT', 'SUBARU',
       'TOYOTA', 'VOLKSWAGEN', 'VOLVO', 'GENESIS', 'BUGATTI']

carr = random.sample(makes, 10)


# Sample 10 different cars
sample_cars = df.sample(n=10, random_state=42)

# SMOTE to generate synthetic data points
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(sample_cars.drop(columns=['CO2_Emissions']), sample_cars['CO2_Emissions'])

# Select the new synthetic data points
synthetic_data = X_resampled[-10:]  # Take the last 10 rows which are synthetic

# Add acceleration and deceleration columns
synthetic_data['acceleration'] = np.random.uniform(0, 100, size=10)
synthetic_data['deceleration'] = np.random.uniform(0, 100, size=10)

# Combine the original and synthetic data for scaling
df['acceleration'] = np.random.uniform(0, 100, size=len(df))
df['deceleration'] = np.random.uniform(0, 100, size=len(df))
combined_data = pd.concat([df.drop(columns=['CO2_Emissions']), synthetic_data])

# Fit the scaler on the combined data
scaler = StandardScaler()
scaler.fit(combined_data)

# Save the scaler
joblib.dump(scaler, 'scaler_with_acc_dec.pkl')

# Scale the synthetic data
synthetic_data_scaled = scaler.transform(synthetic_data)

# Train a new Decision Tree model with the combined data
X_train, y_train = df.drop(columns=['CO2_Emissions']), df['CO2_Emissions']
X_train['acceleration'] = np.random.uniform(0, 100, size=len(X_train))
X_train['deceleration'] = np.random.uniform(0, 100, size=len(X_train))

X_train_scaled = scaler.transform(X_train)
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(dt_model, 'dtt_with_acc_dec.pkl')

# Predict using the new Decision Tree model
dt_predictions = dt_model.predict(synthetic_data_scaled)
synthetic_data['CO2_Emissions_Prediction'] = dt_predictions

# Define the alert function
def speed_alert(acceleration, deceleration, acc_threshold=80, dec_threshold=20):
    if acceleration > acc_threshold:
        alert_message = f"Your acceleration ({acceleration:.2f}) is exceeding the threshold of {acc_threshold}!"
    elif deceleration < dec_threshold:
        alert_message = f"Your deceleration ({deceleration:.2f}) is below the threshold of {dec_threshold}!"
    else:
        alert_message = "Your speed is within the safe range."
    return alert_message

# Create the Tkinter window
root = Tk()
root.title("Car Speed and Emissions Alerts")
root.geometry("1200x600")

img=Image.open("bg.jpg")
img=img.resize((1200,600))
bgg=ImageTk.PhotoImage(img)

lbl=Label(root,image=bgg)
lbl.place(x=0,y=0)


count=0
xx=20
yy=100
# Display alert messages and predictions in the Tkinter window
for i, row in synthetic_data.iterrows():
    print(i)
    if count == 5 :
        xx=750
        yy=100
    alert_message = speed_alert(row['acceleration'], row['deceleration'])
    label_text = f"{carr[i]} Car no : {i+1}:\n{alert_message}\nCO2 Emissions Prediction: {row['CO2_Emissions_Prediction']:.2f} g/km"
    label = Label(root, text=label_text,width=50,bg="black",fg="white",font=("times",12,"bold italic"))
    label.place(x=xx,y=yy)
    count+=1
    yy+=100
root.mainloop()  

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scaled, feature_names=list(X_train.columns), class_names=['CO2_Emissions'], verbose=True, mode='regression')

# Explain the Decision Tree prediction for each synthetic data point
for i in range(10):
    exp_dt = explainer.explain_instance(synthetic_data_scaled[i], dt_model.predict, num_features=10)
    exp_dt.show_in_notebook(show_table=True)


root.mainloop()
