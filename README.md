# AI_Powered_Customer_Retention_Prediction_System

<p align="center">
  <b>Machine Learning | Python | Flask | Scikit-learn</b>
</p>

<p align="center">
  A machine learning-based web application that predicts whether a telecom customer is likely to leave a service, helping businesses identify customers at risk of churn.
</p>

<p align="center">
  <a href="YOUR_DEPLOYMENT_URL">Live Demo</a> •
  <a href="YOUR_GITHUB_REPOSITORY_URL">GitHub Repository</a> •
  <a href="YOUR_LINKEDIN_URL">LinkedIn</a>
</p>

---

## Project Overview

Customer churn is a major challenge for telecom companies. Losing customers can affect revenue and business growth.

This project uses machine learning classification algorithms to predict whether a customer is likely to leave a telecom service based on their demographic information, account details, service subscriptions, and billing information.

A Flask-based web application allows users to enter customer details and receive a churn prediction along with the estimated probability of churn.

## Key Features

- Customer churn prediction using machine learning.
- Interactive and responsive web interface built with Flask.
- Data preprocessing and missing-value handling.
- Categorical feature encoding.
- Class imbalance handling using SMOTE.
- Feature scaling using StandardScaler.
- Comparison of multiple machine learning algorithms.
- Hyperparameter tuning using GridSearchCV.
- Churn probability displayed with the prediction.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Imbalanced Data Handling | Imbalanced-learn (SMOTE) |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Model Storage | Pickle |

## Machine Learning Workflow

1. **Data Collection** – Used the Telecom Customer Churn dataset.
2. **Data Preprocessing** – Removed unnecessary columns and handled missing values.
3. **Exploratory Data Analysis** – Analyzed customer information and churn patterns.
4. **Feature Engineering** – Applied categorical encoding and prepared model features.
5. **Train-Test Split** – Divided the dataset into training and testing sets.
6. **Imbalance Handling** – Applied SMOTE to the training data.
7. **Feature Scaling** – Used StandardScaler to standardize numerical features.
8. **Model Training** – Trained and evaluated multiple classification algorithms.
9. **Hyperparameter Tuning** – Used GridSearchCV to tune the Gaussian Naive Bayes model.
10. **Model Deployment** – Integrated the trained model into a Flask web application.

## Algorithms Implemented

The following machine learning algorithms were trained and compared:

- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

## Model Performance

The models were evaluated using test accuracy.

| Machine Learning Model | Accuracy |
|---|---:|
| K-Nearest Neighbors | 69.69% |
| Gaussian Naive Bayes | 66.29% |
| Logistic Regression | 74.59% |
| Decision Tree | 73.53% |
| Random Forest | 79.28% |
| AdaBoost | 73.53% |
| Gradient Boosting | 75.16% |
| XGBoost | 78.78% |

**Best Accuracy:** Random Forest – 79.28%

**Note:** Accuracy is not the only evaluation metric for churn prediction. Recall, precision, F1-score, and ROC-AUC are also important because identifying customers who may leave is a key objective.

## Web Application

The application provides a simple interface where users can enter customer information and receive a prediction.

**Application functionality:**

- Enter customer demographic details.
- Select subscribed telecom services.
- Provide contract and billing information.
- Submit the customer details for prediction.
- View the predicted churn status.
- View the estimated churn probability.

### Application Screenshots

Add your project screenshots here after uploading them to your GitHub repository.

```markdown
![Home Page](screenshots/home-page.png)

![Prediction Result](screenshots/prediction-result.png)
```

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── Model.pkl
├── scaledown.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── Dataset/
│   └── Data.csv
│
├── screenshots/
│   ├── home-page.png
│   └── prediction-result.png
│
└── README.md
```

## Installation and Execution

Follow these steps to run the project locally.

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project directory**

```bash
cd Customer-Churn-Prediction
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the virtual environment**

Windows:

```bash
venv\Scripts\activate
```

**5. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the Flask application**

```bash
python app.py
```

**7. Open the application**

```text
http://127.0.0.1:5000/
```

## Future Enhancements

- Improve model performance through advanced feature engineering.
- Perform threshold tuning to improve churn detection.
- Add customer-level churn insights and visual analytics.
- Improve the user interface and dashboard.
- Deploy the application to a cloud platform.

## Conclusion

This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model development to building a web application using Flask.

It provides a practical approach to identifying customers who are likely to churn and demonstrates how machine learning can support customer retention strategies.

---

## Author

**Likhith Naga Sai Tadikonda**

- **LinkedIn:** [Add LinkedIn Profile URL](YOUR_LINKEDIN_URL)
- **GitHub:** (https://github.com/likhith22-bot/AI_Powered_Customer_Retention_Prediction_System/tree/main)
- **Live Deployment:** [View Live Project](YOUR_DEPLOYMENT_URL)

---

<p align="center">
  If you find this project useful, consider giving the repository a ⭐
</p>
