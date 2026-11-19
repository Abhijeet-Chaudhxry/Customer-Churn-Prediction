CHURNWISE
Customer Churn Prediction & Analytics

CHURNWISE is a machine learning-based application designed to predict customer churn and help businesses identify customers who may be at risk of leaving their services.

The project uses historical customer data to generate predictions and provide insights that can support better customer retention decisions.

Features

- **Single Customer Prediction:** Predict churn for an individual customer using manual input or a CSV/Excel file.
- **Multiple Customer Prediction:** Upload a dataset to analyse multiple customers at once.
- **Customer Analytics:** Explore customer-level insights and churn patterns across uploaded datasets.
- **Data Validation:** Checks uploaded datasets for required columns and compatible data types.
- **Preprocessing:** Handles missing values and prepares data for machine learning predictions.
- **Churn Risk Identification:** Uses a probability threshold to flag customers who may require attention.

Machine Learning Approach

CHURNWISE uses **Logistic Regression** as its primary classification algorithm.

The model development process includes:

1. Data cleaning and preprocessing.
2. Handling missing numerical and categorical values.
3. Feature selection and target separation.
4. Model training and evaluation.
5. K-Fold Cross-Validation.
6. Classification performance evaluation using accuracy, precision, recall, F1-score and ROC-AUC.
7. Probability threshold selection for identifying potential churn customers.

A probability threshold of **35%** is used to flag customers for further review. This does not mean that a customer is certain to leave; it indicates a potentially higher churn risk.

Technology Stack

- **Python** – Core programming language
- **Pandas & NumPy** – Data manipulation and numerical operations
- **Scikit-learn** – Machine learning and model evaluation
- **Streamlit** – Interactive web application
- **Plotly** – Data visualization
- **Joblib** – Model serialization and loading

Project Structure

```text
CHURNWISE/
│
├── app.py
├── churnwise_model.pkl
├── requirements.txt
├── README.md
└── data/
```

*The structure above is illustrative. Adjust it to match the actual files in your repository.*

Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

**2. Navigate to the project folder**

```bash
cd YOUR-REPOSITORY
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch the application**

```bash
streamlit run app.py
```

The application will open in your browser.

Project Objective

The objective of CHURNWISE is to demonstrate how machine learning can be applied to customer behaviour analysis and churn prediction.

It aims to turn customer data into understandable predictions and insights, making potential retention opportunities easier to identify.

Developer

**Abhijeet Chaudhary**

Developed as an academic Machine Learning project.

---

*CHURNWISE — Turning customer data into clearer decisions.*