# ChurnSense AI

### Explainable Customer Churn Intelligence Platform

**ChurnSense AI** is an end-to-end customer churn prediction platform that combines **Machine Learning, Explainable AI, and LLM-powered business recommendations** to help businesses identify customers at risk of churn and take actionable retention measures.

The platform uses an optimized **XGBoost classifier** to estimate churn probability, **SHAP** to explain the factors influencing each prediction, and **GPT-5 mini** to generate customer-specific retention strategies for higher-risk customers.

---

## Live Demo

**[ChurnSense AI — Live Application](https://churnsense-ai-qpy7.onrender.com)**

> The application is deployed using Render. The free instance may take a few seconds to wake up after inactivity.

---

## Key Features

### Analytics Dashboard

Explore customer and model-level insights through interactive visualizations:

* Customer churn distribution
* Contract-wise churn analysis
* Customer feature distributions
* Box plot analysis
* Correlation analysis
* Model feature importance
* ROC-AUC curve
* Confusion matrix
* Model performance metrics

### Customer Analysis

Analyze an existing customer from the dataset and view:

* Churn probability
* Risk level
* Customer profile
* Major churn drivers
* Protective factors
* SHAP-based explanations

### Predict Customer

Predict churn for a **new individual customer without requiring a CSV file**.

The user fills out an interactive form containing customer information such as:

* Gender
* Senior citizen status
* Partner and dependents
* Tenure
* Monthly and total charges
* CLTV
* Internet service
* Online security and backup
* Device protection
* Tech support
* Streaming services
* Contract type
* Payment method
* Paperless billing

The form is transformed into the same feature representation used during model training and passed directly to the trained XGBoost model.

### AI Retention Strategy

The recommendation system adapts its response according to churn risk.

* **Low risk:** No unnecessary LLM call; the customer is considered suitable for normal engagement.
* **Moderate risk:** Generates a concise retention recommendation.
* **Higher risk:** Generates increasingly detailed, customer-specific retention strategies.

This avoids unnecessarily consuming LLM tokens for customers who have very little churn risk.

### Batch Prediction

Upload a processed customer dataset to:

* Generate churn predictions for multiple customers
* Identify high-risk customers
* Analyze churn probabilities at scale
* Export prediction results

---

## How It Works

```text
                    Customer Data
                         |
                         v
               Data Preprocessing
                         |
                         v
                Feature Engineering
                         |
                         v
                  XGBoost Model
                         |
              +----------+----------+
              |                     |
              v                     v
       Churn Probability       SHAP Analysis
              |                     |
              +----------+----------+
                         |
                         v
                  Risk Assessment
                         |
             +-----------+-----------+
             |                       |
          Low Risk             Higher Risk
             |                       |
             v                       v
       No LLM required       GPT-5 mini
                                     |
                                     v
                           Retention Strategy
                                     |
                                     v
                              Streamlit App
```

---

## Explainability with SHAP

A major goal of ChurnSense AI is not only to predict **whether** a customer may churn, but also to understand **why**.

SHAP values are used to identify:

### Risk Factors

Features that increase the predicted likelihood of churn.

### Protective Factors

Features that reduce the predicted likelihood of churn.

These numerical model explanations are converted into business-friendly descriptions before being displayed to the user.

For example:

> Customer has been with the company for 10 months.

instead of exposing raw encoded values such as:

```text
Tenure Months: 10
```

This makes the predictions easier for non-technical business users to understand.

---

## AI Recommendation Engine

For customers who require intervention, the platform generates a personalized retention strategy.

The LLM receives:

* Customer profile
* Churn probability
* Risk level
* Major risk factors
* Protective factors

The recommendation system is instructed to:

* Use only the supplied customer information
* Avoid inventing customer details
* Produce customer-specific recommendations
* Use professional business language
* Avoid exposing technical ML/SHAP terminology
* Provide practical retention actions

The system also uses **risk-based recommendation depth**, reducing unnecessary API usage for low-risk customers.

---

## Model Performance

The XGBoost model achieved approximately:

| Metric   |     Score |
| -------- | --------: |
| ROC-AUC  | **0.851** |
| Accuracy | **77.7%** |

The model was selected as the primary prediction model because of its strong classification performance and compatibility with tree-based SHAP explainability.

---

## Technology Stack

### Machine Learning

* Python
* XGBoost
* Scikit-learn
* SHAP

### Data Processing

* Pandas
* NumPy
* SciPy

### Visualization

* Streamlit
* Plotly
* Matplotlib
* Seaborn

### Generative AI

* OpenAI API
* GPT-5 mini

### Application

* Streamlit
* Python
* Joblib

### Deployment

* GitHub
* Render

---

## Project Structure

```text
ChurnSense-AI/
|
├── assets/
│   ├── style.css
│   └── workflow.png
|
├── data/
│   ├── raw/
│   └── processed/
|
├── llm/
│   ├── pipeline.py
│   ├── prompts.py
│   ├── recommender.py
│   └── shap_parser.py
|
├── models/
│   └── xgboost_best.pkl
|
├── notebooks/
│   └── ...
|
├── pages/
│   ├── 1_home.py
│   ├── 2_Analysis_dashboard.py
│   ├── 3_customer_analysis.py
│   ├── 4_AI_recommender.py
│   ├── 5_Batch_Prediction.py
│   └── 06_predict_customer.py
|
├── results/
│   ├── model_comparison.csv
│   ├── y_test.csv
│   ├── y_pred.csv
│   └── y_prob.csv
|
├── utils/
│   ├── customer_decoder.py
│   ├── form_encoder.py
│   └── plots.py
|
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/chitrangda07/ChurnSense-AI.git
```

### 2. Navigate to the project

```bash
cd ChurnSense-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5-mini
```

**Never commit your `.env` file to GitHub.**

### 7. Run the application

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## Environment Variables

| Variable         | Description                                       |
| ---------------- | ------------------------------------------------- |
| `OPENAI_API_KEY` | OpenAI API key used for retention recommendations |
| `OPENAI_MODEL`   | LLM used by the recommendation engine             |

For deployment, these variables should be configured through the hosting provider's environment-variable settings rather than committed to the repository.

---

## Deployment

The application is deployed using **Render**.

The deployment runs the Streamlit application using:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

The OpenAI API key is configured securely as a deployment environment variable.

---

## Recommendation Risk Logic

ChurnSense AI does not send every customer to the LLM.

The recommendation depth is adjusted based on churn probability:

```text
Churn Probability

< 30%
   |
Low Risk
   |
No LLM recommendation required

30% - 49%
   |
Moderate Risk
   |
Short recommendation

50% - 69%
   |
Medium/High Risk
   |
Standard recommendation

>= 70%
   |
High/Critical Risk
   |
Detailed retention strategy
```

This design reduces unnecessary LLM calls and makes the recommendation system more cost-efficient.

---

## Project Objective

Traditional churn prediction systems often stop at:

> "This customer is likely to churn."

ChurnSense AI attempts to answer the next and more useful business question:

> **"What should the business do about it?"**

The platform therefore combines prediction, explanation, and action into a single workflow.

```text
Predict
   |
Explain
   |
Prioritize
   |
Recommend
   |
Retain
```

---

## Future Improvements

Potential future enhancements include:

* Customer database integration
* Persistent customer history
* Authentication and role-based access
* Automated email/SMS retention campaigns
* REST API for external applications
* Customer segmentation
* Real-time monitoring
* PDF business reports
* Docker containerization
* Cloud database integration
* Automated model retraining
* Model monitoring and drift detection

---

## Author

**Chitrangda Dubey**

Computer Science Engineering Student

Interested in:

* Machine Learning
* Data Science
* Explainable AI
* Generative AI
* NLP
* Full-Stack Development

---

## Acknowledgements

Built with Python, XGBoost, SHAP, Streamlit, and GPT-5 mini.
