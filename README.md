# AI-Based Student Performance Prediction & Early Support System

## 1. Project Overview

The AI-Based Student Performance Prediction & Early Support System is a machine learning project designed to identify students who may be at risk of failing or withdrawing from a course.

The system analyzes early-course academic and learning engagement information to estimate student risk and suggest appropriate support actions.

## 2. Objectives

* Analyze historical student data.
* Preprocess and clean the dataset.
* Train machine learning classification models.
* Compare Logistic Regression and Random Forest.
* Evaluate model performance using classification metrics.
* Predict student risk using a trained model.
* Provide early-support recommendations through a web application.

## 3. Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

## 4. Dataset

The project uses a student performance dataset containing 32,593 records and 24 columns.

The target variable is `at_risk`.

* 0: Low risk
* 1: At risk

The dataset contains academic and learning engagement information.

## 5. Machine Learning Models

The project compares two classification algorithms:

1. Logistic Regression
2. Random Forest Classifier

The models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and average precision.

Actual evaluation results should be added after running the training notebook.

## 6. Project Structure

```text
student-performance-prediction/
├── app.py
├── student_risk_model.joblib
├── requirements.txt
├── README.md
├── data/
└── notebooks/
```

## 7. Installation

Install Python and open a terminal in the project directory.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 8. Run the Application

Run the following command:

```bash
streamlit run app.py
```

Open the local address displayed in the terminal to access the application.

## 9. Model Training

The machine learning model is trained and evaluated in Google Colab.

After training, the selected model pipeline is saved as:

`student_risk_model.joblib`

Place the downloaded model file in the same directory as `app.py`.

## 10. Important Limitations

The system estimates risk based on patterns in historical data. It cannot guarantee a student's future academic outcome.

Predictions should be reviewed by educators and used to offer appropriate support, not as the sole basis for consequential decisions.

## 11. Future Enhancements

* Interactive student analytics dashboard
* Explainable AI
* Database integration
* Secure user authentication
* Student support tracking
* Improved evaluation on unseen students and future course presentations
