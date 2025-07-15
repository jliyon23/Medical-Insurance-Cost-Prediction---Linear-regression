from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store models and data
models = {}
data = {}
label_encoders = {}

def load_and_preprocess_data():
    """Load and preprocess the insurance dataset"""
    global data, label_encoders
    
    # Load the dataset
    df = pd.read_csv('insurance.csv')
    
    # Store original data for display
    data['original'] = df.copy()
    
    # Convert categorical columns
    df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')
    
    # Label encode categorical variables
    label_encoders['sex'] = LabelEncoder()
    label_encoders['smoker'] = LabelEncoder()
    label_encoders['region'] = LabelEncoder()
    
    df['sex'] = label_encoders['sex'].fit_transform(df['sex'])
    df['smoker'] = label_encoders['smoker'].fit_transform(df['smoker'])
    df['region'] = label_encoders['region'].fit_transform(df['region'])
    
    data['processed'] = df
    return df

def train_models():
    """Train all machine learning models"""
    global models
    
    df = data['processed']
    
    # Prepare data for basic models
    x = df.drop(['charges'], axis=1)
    y = df['charges']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    models['linear'] = {
        'model': lin_reg,
        'score': lin_reg.score(x_test, y_test),
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Ridge Regression
    ridge = Ridge(alpha=0.5)
    ridge.fit(x_train, y_train)
    models['ridge'] = {
        'model': ridge,
        'score': ridge.score(x_test, y_test),
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Lasso Regression
    lasso = Lasso(alpha=0.2, fit_intercept=True, precompute=False, max_iter=1000,
                  tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
    lasso.fit(x_train, y_train)
    models['lasso'] = {
        'model': lasso,
        'score': lasso.score(x_test, y_test),
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Random Forest
    rfr = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                               random_state=1, n_jobs=-1)
    rfr.fit(x_train, y_train)
    x_train_pred = rfr.predict(x_train)
    x_test_pred = rfr.predict(x_test)
    
    models['random_forest'] = {
        'model': rfr,
        'score': metrics.r2_score(y_test, x_test_pred),
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_pred': x_train_pred,
        'test_pred': x_test_pred
    }
    
    # Polynomial Regression
    x_poly = df.drop(['charges', 'sex', 'region'], axis=1)
    pol = PolynomialFeatures(degree=2)
    x_pol = pol.fit_transform(x_poly)
    x_train_pol, x_test_pol, y_train_pol, y_test_pol = train_test_split(x_pol, y, test_size=0.2, random_state=0)
    
    pol_reg = LinearRegression()
    pol_reg.fit(x_train_pol, y_train_pol)
    
    models['polynomial'] = {
        'model': pol_reg,
        'poly_features': pol,
        'score': pol_reg.score(x_test_pol, y_test_pol),
        'x_train': x_train_pol,
        'x_test': x_test_pol,
        'y_train': y_train_pol,
        'y_test': y_test_pol
    }

def create_correlation_heatmap():
    """Create correlation heatmap"""
    df = data['processed']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_feature_importance_plot():
    """Create feature importance plot for Random Forest"""
    rfr_model = models['random_forest']['model']
    
    importances = rfr_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    variables = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(len(variables)), importances[indices], color="skyblue")
    plt.xticks(range(len(variables)), [variables[i] for i in indices], rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def create_residual_plot():
    """Create residual plot for Random Forest"""
    model_data = models['random_forest']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(model_data['train_pred'], model_data['train_pred'] - model_data['y_train'],
                c='gray', marker='o', s=35, alpha=0.5, label='Train data')
    plt.scatter(model_data['test_pred'], model_data['test_pred'] - model_data['y_test'],
                c='blue', marker='o', s=35, alpha=0.7, label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper right')
    plt.hlines(y=0, xmin=0, xmax=60000, lw=2, color='red')
    plt.title('Residual Plot (Random Forest)')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    """Main page with dataset overview and model comparison"""
    df = data['original']
    
    # Dataset statistics
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'description': df.describe().to_html(classes='table table-striped table-hover')
    }
    
    # Model scores
    model_scores = {
        'Linear Regression': round(models['linear']['score'], 4),
        'Ridge Regression': round(models['ridge']['score'], 4),
        'Lasso Regression': round(models['lasso']['score'], 4),
        'Random Forest': round(models['random_forest']['score'], 4),
        'Polynomial Regression': round(models['polynomial']['score'], 4)
    }
    
    # Create correlation heatmap
    correlation_plot = create_correlation_heatmap()
    
    return render_template('index.html', 
                         stats=stats, 
                         model_scores=model_scores,
                         correlation_plot=correlation_plot,
                         sample_data=df.head(10).to_html(classes='table table-striped table-hover'))

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        model_type = request.form['model']
        
        # Encode categorical variables
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
        region_encoded = label_encoders['region'].transform([region])[0]
        
        # Prepare input data
        if model_type == 'polynomial':
            # For polynomial, we don't use sex and region
            input_data = np.array([[age, bmi, children, smoker_encoded]])
            input_data = models['polynomial']['poly_features'].transform(input_data)
        else:
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        
        # Make prediction
        model = models[model_type]['model']
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'model_used': model_type.replace('_', ' ').title()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/analysis')
def analysis():
    """Analysis page with visualizations"""
    # Create plots
    feature_importance_plot = create_feature_importance_plot()
    residual_plot = create_residual_plot()
    
    # Random Forest metrics
    rf_model = models['random_forest']
    rf_metrics = {
        'train_mse': round(metrics.mean_squared_error(rf_model['y_train'], rf_model['train_pred']), 3),
        'test_mse': round(metrics.mean_squared_error(rf_model['y_test'], rf_model['test_pred']), 3),
        'train_r2': round(metrics.r2_score(rf_model['y_train'], rf_model['train_pred']), 3),
        'test_r2': round(metrics.r2_score(rf_model['y_test'], rf_model['test_pred']), 3)
    }
    
    # Polynomial metrics
    poly_model = models['polynomial']
    poly_pred = poly_model['model'].predict(poly_model['x_test'])
    poly_metrics = {
        'mae': round(metrics.mean_absolute_error(poly_model['y_test'], poly_pred), 3),
        'mse': round(metrics.mean_squared_error(poly_model['y_test'], poly_pred), 3),
        'rmse': round(np.sqrt(metrics.mean_squared_error(poly_model['y_test'], poly_pred)), 3)
    }
    
    return render_template('analysis.html',
                         feature_importance_plot=feature_importance_plot,
                         residual_plot=residual_plot,
                         rf_metrics=rf_metrics,
                         poly_metrics=poly_metrics)

@app.route('/model_details/<model_name>')
def model_details(model_name):
    """Show detailed information about a specific model"""
    if model_name not in models:
        return "Model not found", 404
    
    model_data = models[model_name]
    model = model_data['model']
    
    details = {
        'name': model_name.replace('_', ' ').title(),
        'score': round(model_data['score'], 4),
        'intercept': getattr(model, 'intercept_', 'N/A'),
        'coefficients': getattr(model, 'coef_', 'N/A')
    }
    
    # Convert numpy arrays to lists for JSON serialization
    if hasattr(details['coefficients'], 'tolist'):
        details['coefficients'] = details['coefficients'].tolist()
    
    return render_template('model_details.html', details=details, model_name=model_name)

if __name__ == '__main__':
    # Initialize the application
    print("Loading and preprocessing data...")
    load_and_preprocess_data()
    
    print("Training models...")
    train_models()
    
    print("Starting Flask application...")
    app.run(debug=True)