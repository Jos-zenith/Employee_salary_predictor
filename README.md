# Employee Salary Prediction Web Application

This project is an employee salary prediction web application that uses a Vue 3 front end and a Flask prediction API to estimate whether an employee's salary exceeds $50,000 based on various features.

## Project Structure

```
employee-salary-prediction-app
├── data
│   └── employees.csv          # Dataset used for training the model
├── models
│   └── salary_prediction_model.joblib  # Trained machine learning model
├── src
│   ├── app.py                 # Flask app that serves the Vue portal and prediction API
│   ├── templates
│   │   └── index.html         # Vue 3 interface with v-model bindings
│   └── train_model.py         # Script for training the machine learning model
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

## Dataset

The dataset `employees.csv` contains various features related to employees, such as age, work class, education, marital status, occupation, relationship, race, gender, capital gain, capital loss, hours worked per week, and native country. This data is used to train the machine learning model.

## Model Training

The `train_model.py` script is responsible for preprocessing the data, selecting relevant features, training the model, and saving it as `salary_prediction_model.joblib`. The model can then be loaded for predictions in the web application.

## Web Application

The `src/app.py` file now serves a Vue-powered portal. Users can edit employee details in a polished dashboard layout, submit them to `/api/predict`, and view the salary prediction with confidence. The backend applies the same label encoding used during training so the input schema matches the model.

## Requirements

To run this project, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

## Running the Application

To start the web application, navigate to the `employee-salary-prediction-app` directory and run:

```
python src/app.py
```

Then open the local Flask URL shown in the terminal.

## Objectives

- To provide a web-based tool for predicting employee salaries based on various features.
- To demonstrate the use of machine learning in a practical application.
- To enhance skills in data preprocessing, model training, and web application development using Python.