XAI-Driven EmissionPredictor

A machine learning project for predicting vehicle CO2 emissions with explainable AI (XAI) integration for smart city solutions.

# Features

CO2 Emission Prediction: Forecast vehicle emissions based on various factors like engine size, fuel consumption, etc.
Explainable AI (XAI): Uses LIME to provide interpretable predictions for policymakers.
Data Imbalance Handling: Incorporates SMOTE to balance the dataset.
Multiple Models: Trained using Decision Tree, Random Forest, and Gradient Boosting models.

# Installation
# Clone the repository
git clone https://github.com/deekshaa4/XAI-Driven-EmissionModel.git

# Navigate to the project directory
cd XAI-Driven-EmissionModel

# Create a virtual environment for Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Create a virtual environment for Windows
python -m venv venv
venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt

# Usage
# Run the main script to train and test the model
python main.py

# View Explainable AI (LIME) results
After training, LIME will generate interpretable results for each prediction.
Results will be displayed in the console or as visual plots.

# Evaluate Model Performance
You can check the performance of the model using the R² score and MAPE.

# Technologies

Python 3.8+
Scikit-learn: For machine learning models.
Pandas, NumPy: For data handling.
LIME: For model explainability.
SMOTE: For handling imbalanced datasets.
