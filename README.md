
# Dengue Risk Prediction Web App (2025)
![Homepage](https://github.com/shahin5646/dengue_risk_checker_Website_using_ML/blob/5dcfcf3d6121af932714c66da25e7990088b38ff/Result.png)
![Result](https://github.com/shahin5646/dengue_risk_checker_Website_using_ML/blob/5dcfcf3d6121af932714c66da25e7990088b38ff/Homepage.png)


A modern, production-ready Flask web application for predicting dengue risk in Dhaka, Bangladesh, using machine learning. This project leverages XGBoost, scikit-learn, and advanced feature engineering to provide both individual risk predictions and public health insights.

---

## 🚀 Features

- **Dengue Risk Prediction**: Predicts the probability of dengue infection based on user input (age, gender, area, house type, etc.).
- **Geographic Hotspot Analysis**: Highlights high-risk areas in Dhaka using historical data.
- **Risk Factor Explanation**: Explains top features influencing risk (model interpretability).
- **REST API**: Programmatic access for integration and automation.
- **Production-Ready**: Gunicorn/WSGI support, health checks, error handling, and secure configuration.

---

## 🏗️ Project Structure

```
├── app_flask_predict.py            # Main Flask app
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version (for deployment)
├── .python-version                 # Python version hint (for Render)
├── Procfile                        # Gunicorn start command
├── templates/                      # HTML templates (form, result, about)
├── static/                         # Static assets (CSS, JS, images)
├── models/                         # (Optional) Model files
├── assets/                         # (Optional) Additional assets
├── data/                           # Input data (CSV, JSON)
├── best_dengue_risk_model.pkl      # Main ML model
├── logistic_regression_model.pkl   # Backup model
├── risk_preprocessor.pkl           # Preprocessing pipeline
├── feature_info.pkl                # Feature importance info
├── area_stats.pkl                  # Area statistics
```

## 📊 Model & Data

- **ML Model**: XGBoost (main), Logistic Regression (backup)
- **Preprocessing**: ColumnTransformer (OneHotEncoder, StandardScaler)
- **Data**: Historical dengue data for Dhaka (see `data/`)
- **Feature Engineering**: Age binning, area/house type flags, etc.

---

## 🛡️ Security & Best Practices

- No secrets or credentials in code
- Input validation and error handling
- Health check endpoint for monitoring
- Compatible with modern Python (3.11+)
- Ready for containerization and cloud deployment

---

## 👨‍🔬 Authors & Credits

- Developed by: Shahin
- Data sources: [Kaggle - Dengue Dataset Bangladesh](https://www.kaggle.com/datasets/kawsarahmad/dengue-dataset-bangladesh?resource=download)
- Libraries: Flask, scikit-learn, XGBoost, pandas, numpy, joblib, etc.

---

