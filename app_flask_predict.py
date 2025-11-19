
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
import warnings

# Ensure UTF-8 encoding for console output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dengue_risk_prediction_2024'

# ==========================================
# 1. LOAD MODELS AND PREPROCESSORS
# ==========================================

def load_models():
    """Load trained models and preprocessing objects"""
    try:
        # Load the main XGBoost model
        model = joblib.load('best_dengue_risk_model.pkl')
        
        # Load backup logistic regression model
        backup_model = joblib.load('logistic_regression_model.pkl')
        
        # Load the preprocessor (ColumnTransformer with OneHotEncoder, StandardScaler)
        preprocessor = joblib.load('risk_preprocessor.pkl')
        
        # Load feature information and top risk factors
        feature_info = joblib.load('feature_info.pkl')
        
        # Load area statistics for context
        area_stats = joblib.load('area_stats.pkl')
        
        return {
            'model': model,
            'backup_model': backup_model,
            'preprocessor': preprocessor,
            'feature_info': feature_info,
            'area_stats': area_stats
        }
    except FileNotFoundError as e:
        print(f"⚠️  Error loading models: {e}")
        return None


# Load all models at startup
MODELS = load_models()

if MODELS is None:
    print("❌ ERROR: Could not load required model files!")
    print("Expected files: best_dengue_risk_model.pkl, risk_preprocessor.pkl, etc.")
else:
    print("✅ Models loaded successfully!")


# ==========================================
# 2. AREA RISK ASSESSMENT (Geographic Risk)
# ==========================================

# Define Dhaka dengue hotspot areas based on historical data
HOTSPOT_AREAS = {
    'Jatrabari': {'risk_level': 'Very High', 'dengue_cases': 'High'},
    'Tejgaon': {'risk_level': 'Very High', 'dengue_cases': 'High'},
    'Mohammadpur': {'risk_level': 'High', 'dengue_cases': 'Moderate-High'},
    'Bangshal': {'risk_level': 'High', 'dengue_cases': 'Moderate-High'},
    'Mirpur': {'risk_level': 'High', 'dengue_cases': 'Moderate'},
    'Gulshan': {'risk_level': 'Moderate', 'dengue_cases': 'Low-Moderate'},
    'Dhanmondi': {'risk_level': 'Moderate', 'dengue_cases': 'Low-Moderate'},
}

def assess_geographic_risk(area, area_type):
    """
    Assess dengue risk based on geographic location (Area).
    Goal 2: Understand that Area is the #1 risk predictor.
    
    Args:
        area (str): Neighborhood/area name
        area_type (str): 'Developed' or 'Undeveloped'
    
    Returns:
        dict: Area context and risk information
    """
    area_normalized = area.strip().title()
    
    context = {
        'area': area_normalized,
        'area_type': area_type,
        'is_hotspot': False,
        'area_message': ''
    }
    
    # Check if area is a known hotspot
    if area_normalized in HOTSPOT_AREAS:
        hotspot_info = HOTSPOT_AREAS[area_normalized]
        context['is_hotspot'] = True
        context['area_message'] = (
            f"📍 {area_normalized} is a known dengue hotspot. "
            f"Risk Level: {hotspot_info['risk_level']}. "
            f"Historical dengue cases: {hotspot_info['dengue_cases']}. "
            f"Extra precautions recommended."
        )
    else:
        # Default message for non-hotspot areas
        risk_by_type = "higher risk of dengue transmission" if area_type == "Undeveloped" else "relatively controlled dengue transmission"
        context['area_message'] = (
            f"📍 {area_normalized} ({area_type} area) has {risk_by_type}. "
            f"Risk depends on local environmental factors and population density."
        )
    
    return context


# ==========================================
# 3. FEATURE ENGINEERING FOR PREDICTION
# ==========================================

def create_feature_vector(age, gender, area_type, house_type, area):
    # Create a DataFrame with ALL features )
    user_data = pd.DataFrame({
        'Age': [int(age)],
        'Gender': [gender.strip()],
        'AreaType': [area_type.strip()],
        'HouseType': [house_type.strip()],
        'Area': [area.strip()]
    })
    # Feature Engineering 
    age_bins_edges = [0, 10, 20, 30, 40, 50, 100]
    user_data['age_bin'] = np.digitize(user_data['Age'], age_bins_edges, right=True) - 1
    user_data['is_tinshed'] = (user_data['HouseType'].str.lower().str.contains('tin', na=False)).astype(int)
    user_data['is_undeveloped'] = (user_data['AreaType'] == 'Undeveloped').astype(int)
    # Ensure column order matches model training
    ordered_cols = ['Age', 'age_bin', 'is_tinshed', 'is_undeveloped', 'Gender', 'AreaType', 'HouseType', 'Area']
    user_data = user_data[ordered_cols]
    print("\n[DEBUG] Feature vector columns and values sent to preprocessor:")
    print(user_data)
    return user_data


def prepare_prediction_input(age, gender, area_type, house_type, area):
    try:
        # Validate age
        age = int(age)
        if age < 1 or age > 120:
            return None, "Age must be between 1 and 120 years."
        
        # Create feature vector
        features = create_feature_vector(age, gender, area_type, house_type, area)
        
        return features, None
    
    except ValueError as e:
        return None, f"Invalid input: {str(e)}"



# 4. PREDICTION LOGIC (Goal 1: Predict Dengue Risk)
def make_prediction(features_df):
    if MODELS is None:
        return {
            'success': False,
            'error': 'Models not loaded. Please check system files.'
        }
    
    try:
        # Use the model pipeline directly (it includes the preprocessor)
        prediction = MODELS['model'].predict(features_df)[0]
        prediction_proba = MODELS['model'].predict_proba(features_df)[0]

        # Risk probability as percentage (probability of dengue positive)
        risk_probability = prediction_proba[1] * 100  # Class 1 = Dengue Positive

        # Prediction text
        prediction_text = "⚠️ DENGUE POSITIVE (High Risk)" if prediction == 1 else "✅ DENGUE NEGATIVE (Low Risk)"

        # Determine risk level and recommendation
        if risk_probability >= 70:
            risk_level = "Very High"
            recommendation = "🚨 Urgent medical consultation recommended. Seek immediate medical attention for testing and evaluation."
        elif risk_probability >= 50:
            risk_level = "High"
            recommendation = "⚠️ Medical evaluation recommended. Please consult a healthcare provider soon for proper assessment."
        elif risk_probability >= 30:
            risk_level = "Moderate"
            recommendation = "⚡ Monitor symptoms carefully. Seek medical advice if you develop fever, headache, or joint pain."
        else:
            risk_level = "Low"
            recommendation = "✅ Continue regular health practices. Maintain good hygiene and use mosquito repellent during monsoon season."

        # Color indicator for UI
        if risk_probability >= 70:
            risk_color = "🔴"
        elif risk_probability >= 50:
            risk_color = "🟠"
        elif risk_probability >= 30:
            risk_color = "🟡"
        else:
            risk_color = "🟢"

        return {
            'success': True,
            'prediction': prediction,
            'probability_pct': risk_probability,
            'prediction_text': prediction_text,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'risk_color': risk_color,
            'class_0_prob': prediction_proba[0] * 100,  # No dengue probability
            'class_1_prob': prediction_proba[1] * 100   # Dengue probability
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Prediction error: {str(e)}"
        }


# ==========================================
# 5. RISK FACTOR EXPLANATION (Goal 2: Feature Importance)
# ==========================================

def get_risk_factors():
    """
    Get top risk factors from the trained model.
    Goal 2: Explain what factors most influence dengue risk.
    
    Returns:
        list: Top risk factors with importance scores
    """
    if MODELS is None:
        return []
    
    try:
        feature_info = MODELS['feature_info']
        
        # Get top 10 risk factors from XGBoost feature importance
        top_factors = feature_info.get('top_risk_factors', {})
        
        if not top_factors:
            return []
        
        # Convert to list format for template
        risk_factors = []
        max_importance = max(top_factors.values()) if top_factors else 1
        
        for rank, (factor_name, importance) in enumerate(top_factors.items(), 1):
            percentage = (importance / max_importance) * 100 if max_importance > 0 else 0
            risk_factors.append({
                'rank': rank,
                'name': factor_name,
                'importance': importance,
                'percentage': percentage
            })
        
        return risk_factors
    
    except Exception as e:
        print(f"⚠️  Error getting risk factors: {e}")
        return []


# ==========================================
# 6. FLASK ROUTES
# ==========================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home page: Display prediction form (GET) or process prediction (POST).
    """
    if request.method == 'GET':
        # Display the form
        return render_template('form.html')
    
    elif request.method == 'POST':
        # Process form submission
        try:
            # Extract form data
            age = request.form.get('age')
            gender = request.form.get('gender')
            area_type = request.form.get('area_type')
            house_type = request.form.get('house_type')
            area = request.form.get('area')
            
            # Validate and prepare input
            features_df, error = prepare_prediction_input(age, gender, area_type, house_type, area)
            
            if error:
                return render_template('form.html', error=error)
            
            # Make prediction (Goal 1)
            prediction_result = make_prediction(features_df)
            
            if not prediction_result['success']:
                error_msg = prediction_result.get('error', 'Unknown prediction error')
                return render_template('form.html', error=error_msg)
            
            # Get geographic risk context (Goal 2)
            area_context = assess_geographic_risk(area, area_type)
            
            # Get risk factors explanation (Goal 2)
            risk_factors = get_risk_factors()
            
            # Prepare result data for template
            result_data = {
                'age': age,
                'gender': gender,
                'area_type': area_type,
                'house_type': house_type,
                'area': area,
                'area_context': area_context,
                'prediction_text': prediction_result['prediction_text'],
                'probability_pct': prediction_result['probability_pct'],
                'risk_level': prediction_result['risk_level'],
                'recommendation': prediction_result['recommendation'],
                'risk_color': prediction_result['risk_color'],
                'risk_factors': risk_factors,
                'class_0_prob': prediction_result['class_0_prob'],
                'class_1_prob': prediction_result['class_1_prob']
            }
            print("\n==== FLASK DEBUG: result_data to result.html ====")
            for k, v in result_data.items():
                print(f"{k}: {v}")
            print("==== END FLASK DEBUG ====")
            # Render result page
            return render_template('result.html', **result_data)
        
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"
            return render_template('form.html', error=error)


@app.route('/about', methods=['GET'])
def about():
    """
    About page: Display project information and developer details.
    """
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for programmatic predictions.
    Expects JSON: {'age': int, 'gender': str, 'area_type': str, 'house_type': str, 'area': str}
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'area_type', 'house_type', 'area']
        if not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'error': f'Missing required fields. Required: {required_fields}'
            }), 400
        
        # Prepare input
        features_df, error = prepare_prediction_input(
            data['age'],
            data['gender'],
            data['area_type'],
            data['house_type'],
            data['area']
        )
        
        if error:
            return jsonify({'success': False, 'error': error}), 400
        
        # Make prediction
        prediction_result = make_prediction(features_df)
        
        if not prediction_result['success']:
            return jsonify(prediction_result), 500
        
        # Get geographic and risk factor information
        area_context = assess_geographic_risk(data['area'], data['area_type'])
        risk_factors = get_risk_factors()
        
        return jsonify({
            'success': True,
            'input': {
                'age': int(data['age']),
                'gender': data['gender'],
                'area': data['area'],
                'area_type': data['area_type'],
                'house_type': data['house_type']
            },
            'prediction': {
                'class': 'Dengue Positive' if prediction_result['prediction'] == 1 else 'Dengue Negative',
                'probability_percent': round(prediction_result['probability_pct'], 2),
                'risk_level': prediction_result['risk_level'],
                'recommendation': prediction_result['recommendation']
            },
            'area_context': area_context,
            'top_risk_factors': risk_factors[:5]
        }), 200
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/risk-factors', methods=['GET'])
def api_risk_factors():
    """
    API endpoint to get top risk factors (Goal 2).
    """
    try:
        risk_factors = get_risk_factors()
        return jsonify({
            'success': True,
            'risk_factors': risk_factors,
            'note': 'Area/Geographic location is the #1 predictor of dengue risk in Dhaka region'
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/hotspots', methods=['GET'])
def api_hotspots():
    """
    API endpoint to get known dengue hotspot areas.
    Useful for public health teams and awareness campaigns.
    """
    try:
        hotspots = []
        for area_name, info in HOTSPOT_AREAS.items():
            hotspots.append({
                'area': area_name,
                'risk_level': info['risk_level'],
                'dengue_cases': info['dengue_cases']
            })
        
        return jsonify({
            'success': True,
            'hotspots': hotspots,
            'total_hotspot_areas': len(hotspots)
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint. Verifies that models are loaded.
    """
    models_loaded = MODELS is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'endpoints': {
            'form': '/',
            'api_predict': '/api/predict (POST)',
            'risk_factors': '/api/risk-factors (GET)',
            'hotspots': '/api/hotspots (GET)',
            'health': '/health (GET)'
        }
    }), 200 if models_loaded else 503


# ==========================================
# 7. ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('form.html', error='Page not found.'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('form.html', error='Internal server error. Please try again.'), 500


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" DENGUE RISK PREDICTION - FLASK APPLICATION")
    print("=" * 70)
    print("\n Application Configuration:")
    print(f"   - Models Loaded: {MODELS is not None}")
    print(f"   - Form Route: http://localhost:5000/")
    print(f"   - API Endpoint: http://localhost:5000/api/predict")
    print(f"   - Risk Factors: http://localhost:5000/api/risk-factors")
    print(f"   - Hotspots: http://localhost:5000/api/hotspots")
    print(f"   - Health Check: http://localhost:5000/health")
    print("\n Goal 1: Predict dengue risk probability for individuals")
    print(" Goal 2: Explain top risk factors influencing dengue in Dhaka")
    print("\n" + "=" * 70 + "\n")
    
    # Run Flask app
    app.run(
        debug=True,           # Enable debug mode for development
        host='127.0.0.1',     # Listen on localhost
        port=5000,            # Port 5000
        use_reloader=True     # Auto-reload on code changes
    )
