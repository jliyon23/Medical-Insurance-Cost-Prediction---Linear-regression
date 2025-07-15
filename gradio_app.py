import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

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
    models['Linear Regression'] = {
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
    models['Ridge Regression'] = {
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
    models['Lasso Regression'] = {
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
    
    models['Random Forest'] = {
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
    
    models['Polynomial Regression'] = {
        'model': pol_reg,
        'poly_features': pol,
        'score': pol_reg.score(x_test_pol, y_test_pol),
        'x_train': x_train_pol,
        'x_test': x_test_pol,
        'y_train': y_train_pol,
        'y_test': y_test_pol
    }

def predict_insurance_cost(age, sex, bmi, children, smoker, region, model_type):
    """Make prediction using the selected model"""
    try:
        # Encode categorical variables
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
        region_encoded = label_encoders['region'].transform([region])[0]
        
        # Prepare input data
        if model_type == 'Polynomial Regression':
            # For polynomial, we don't use sex and region
            input_data = np.array([[age, bmi, children, smoker_encoded]])
            input_data = models['Polynomial Regression']['poly_features'].transform(input_data)
        else:
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
        
        # Make prediction
        model = models[model_type]['model']
        prediction = model.predict(input_data)[0]
        model_score = models[model_type]['score']
        
        # Format result
        result = f"""
        **Predicted Insurance Cost: ${prediction:,.2f}**
        
        **Model Details:**
        - Model Used: {model_type}
        - Model R² Score: {model_score:.4f}
        - Prediction Confidence: {'High' if model_score > 0.8 else 'Medium' if model_score > 0.6 else 'Low'}
        
        **Input Summary:**
        - Age: {age} years
        - Gender: {sex}
        - BMI: {bmi}
        - Children: {children}
        - Smoker: {smoker}
        - Region: {region}
        """
        
        return result
        
    except Exception as e:
        return f"Error making prediction: {str(e)}"

def create_correlation_heatmap():
    """Create correlation heatmap"""
    df = data['processed']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return plt

def create_feature_importance_plot():
    """Create feature importance plot for Random Forest"""
    rfr_model = models['Random Forest']['model']
    
    importances = rfr_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    variables = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)", fontsize=16, fontweight='bold')
    bars = plt.bar(range(len(variables)), importances[indices], 
                   color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'])
    plt.xticks(range(len(variables)), [variables[i] for i in indices], rotation=45)
    plt.ylabel('Importance')
    plt.xlabel('Features')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        importance_value = importances[indices[i]]
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{importance_value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def get_model_comparison():
    """Get model comparison data"""
    comparison_data = []
    for model_name, model_info in models.items():
        comparison_data.append([
            model_name,
            f"{model_info['score']:.4f}"
        ])
    
    # Sort by score (descending)
    comparison_data.sort(key=lambda x: float(x[1]), reverse=True)
    
    return comparison_data

def get_dataset_info():
    """Get dataset information"""
    df = data['original']
    
    info = f"""
    **Dataset Overview:**
    - Total Records: {df.shape[0]:,}
    - Features: {df.shape[1]}
    - Columns: {', '.join(df.columns.tolist())}
    
    **Statistical Summary:**
    """
    
    return info, df.describe()

# Initialize data and models
print("Loading and preprocessing data...")
load_and_preprocess_data()

print("Training models...")
train_models()

print("Creating Gradio interface...")

# Create Gradio interface
with gr.Blocks(title="Medical Insurance Cost Prediction", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Medical Insurance Cost Prediction
    
    Use machine learning models to predict insurance costs based on personal and demographic information.
    """)
    
    with gr.Tabs():
        # Prediction Tab
        with gr.TabItem("🔮 Make Prediction"):
            gr.Markdown("### Enter your information to get an insurance cost prediction")
            
            with gr.Row():
                with gr.Column():
                    age = gr.Slider(18, 100, value=30, label="Age", info="Your age in years")
                    sex = gr.Dropdown(["male", "female"], label="Gender", info="Select your gender")
                    bmi = gr.Slider(10, 60, value=25, step=0.1, label="BMI", info="Body Mass Index")
                
                with gr.Column():
                    children = gr.Slider(0, 10, value=0, step=1, label="Children", info="Number of children")
                    smoker = gr.Dropdown(["yes", "no"], label="Smoker", info="Do you smoke?")
                    region = gr.Dropdown(["northeast", "northwest", "southeast", "southwest"], 
                                       label="Region", info="Your region")
            
            model_type = gr.Dropdown(
                list(models.keys()), 
                value="Random Forest",
                label="Prediction Model", 
                info="Choose the machine learning model"
            )
            
            predict_btn = gr.Button("🎯 Predict Insurance Cost", variant="primary", size="lg")
            prediction_output = gr.Markdown()
            
            predict_btn.click(
                predict_insurance_cost,
                inputs=[age, sex, bmi, children, smoker, region, model_type],
                outputs=prediction_output
            )
        
        # Dataset Analysis Tab
        with gr.TabItem("📊 Dataset Analysis"):
            gr.Markdown("### Explore the dataset and model performance")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Dataset Information")
                    dataset_info, dataset_stats = get_dataset_info()
                    gr.Markdown(dataset_info)
                    gr.Dataframe(dataset_stats, label="Statistical Summary")
                    
                with gr.Column():
                    gr.Markdown("#### Model Performance Comparison")
                    comparison_data = get_model_comparison()
                    gr.Dataframe(
                        comparison_data, 
                        headers=["Model", "R² Score"],
                        label="Model Comparison (sorted by performance)"
                    )
        
        # Visualizations Tab
        with gr.TabItem("📈 Visualizations"):
            gr.Markdown("### Data Visualizations and Model Insights")
            
            with gr.Row():
                correlation_plot = gr.Plot(create_correlation_heatmap(), label="Feature Correlation Heatmap")
                feature_importance_plot = gr.Plot(create_feature_importance_plot(), label="Feature Importance")
            
            gr.Markdown("""
            #### Key Insights:
            - **Smoking status** is the most significant predictor of insurance charges
            - **Age** shows strong correlation with insurance costs  
            - **BMI** is an important health indicator for pricing
            - **Gender and region** have relatively lower impact
            """)
        
        # Sample Data Tab
        with gr.TabItem("🔍 Sample Data"):
            gr.Markdown("### Sample data from the insurance dataset")
            sample_data = data['original'].head(20)
            gr.Dataframe(sample_data, label="First 20 records from the dataset")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
