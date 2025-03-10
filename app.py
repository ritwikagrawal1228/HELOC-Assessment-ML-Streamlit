import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import base64
from fpdf import FPDF
import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time
import re

# Set page config
st.set_page_config(
    page_title="HELOC Application Assessment System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = "models/xgboost_model_optimized.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_NAMES_PATH = "models/feature_names.json"
METADATA_PATH = "models/model_metadata.json"
THRESHOLD_PATH = "models/threshold_analysis.csv"

# Initialize session state
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'custom_threshold' not in st.session_state:
    st.session_state.custom_threshold = 0.5
if 'applications' not in st.session_state:
    st.session_state.applications = []
if 'assessments' not in st.session_state:
    st.session_state.assessments = []
if 'form_prefill' not in st.session_state:
    st.session_state.form_prefill = {}

# Helper functions
def load_model_resources():
    """Load the model, scaler, and metadata files"""
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open(FEATURE_NAMES_PATH, 'r') as f:
            feature_names = json.load(f)
        
        # Load model metadata
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        # Load threshold analysis
        threshold_df = pd.read_csv(THRESHOLD_PATH)
        
        # Get recommended threshold (best F1 score)
        recommended_threshold = threshold_df.sort_values('f1', ascending=False).iloc[0]['threshold']
        
        return model, scaler, feature_names, metadata, threshold_df, recommended_threshold
    
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        return None, None, None, None, None, None

def predict_application(model, scaler, features_df, threshold):
    """Make predictions for loan applications"""
    try:
        # Ensure features are in the correct order
        if hasattr(scaler, 'feature_names_in_'):
            # Sklearn 1.0+ stores feature names used during fit
            expected_features = scaler.feature_names_in_
            # Reorder columns to match expected order
            features_df = features_df.reindex(columns=expected_features)
        elif 'expected_features' in feature_names:
            # Use our saved feature names as fallback
            expected_features = feature_names['expected_features']
            features_df = features_df.reindex(columns=expected_features)
            
        # Scale features
        scaled_features = scaler.transform(features_df)
        
        # Get probability predictions
        probabilities = model.predict_proba(scaled_features)[:, 1]
        
        # Make decisions based on threshold
        decisions = (probabilities >= threshold).astype(int)
        
        return probabilities, decisions
    
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None, None

def validate_input_data(df, expected_features):
    """Validate that the input dataframe has all expected features"""
    missing_cols = set(expected_features) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_features)
    
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"
    
    # Check for null values
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        return False, f"Null values found in columns: {', '.join(null_cols)}"
    
    return True, "Data validated successfully"

def get_explanation(feature_importance_dict, probability, features_df, idx=0):
    """Generate an explanation for the prediction"""
    # Sort features by importance
    top_features = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    explanation = []
    for feature, importance in top_features:
        value = features_df.iloc[idx][feature]
        contribution = importance * value
        
        if contribution > 0:
            direction = "increases"
        else:
            direction = "decreases"
        
        explanation.append(f"- {feature}: Value {value:.2f} {direction} rejection probability by {abs(contribution):.4f}")
    
    return explanation

def generate_pdf_report(application_data, probability, decision, explanation, applicant_name="Applicant"):
    """Generate a PDF report for the loan application assessment"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, 'HELOC Application Assessment Report', 0, 1, 'C')
    
    # Applicant info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, f'Applicant: {applicant_name}', 0, 1)
    pdf.cell(190, 10, f'Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    
    # Decision
    pdf.set_font('Arial', 'B', 14)
    if decision:
        pdf.set_text_color(255, 0, 0)
        result_text = "REJECTED"
    else:
        pdf.set_text_color(0, 128, 0)
        result_text = "APPROVED"
    
    pdf.cell(190, 15, f'Result: {result_text}', 0, 1)
    pdf.set_text_color(0, 0, 0)
    
    # Probability
    pdf.set_font('Arial', '', 12)
    pdf.cell(190, 10, f'Rejection Probability: {probability:.2%}', 0, 1)
    
    # Explanation
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'Key Factors:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for line in explanation:
        pdf.multi_cell(190, 7, line)
    
    # Application details
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'Application Details:', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Get top 5 features by importance
    if feature_importance_dict:
        top_5_features = [f[0] for f in sorted(feature_importance_dict.items(), 
                                              key=lambda x: abs(x[1]), reverse=True)[:5]]
    else:
        top_5_features = []
    
    for key, value in application_data.items():
        if key in top_5_features:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(90, 7, key, 0)
            pdf.cell(100, 7, f"{value}", 0, 1)
            pdf.set_font('Arial', '', 10)
        else:
            pdf.cell(90, 7, key, 0)
            pdf.cell(100, 7, f"{value}", 0, 1)
    
    # Footer
    pdf.set_y(-30)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, 'This is an automated assessment report generated by the HELOC Assessment System.', 0, 1, 'C')
    pdf.cell(0, 10, 'Please contact the lending department for any questions.', 0, 1, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

def create_download_link(file_bytes, filename):
    """Create a download link for a file"""
    b64 = base64.b64encode(file_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/000000/cottage.png", width=100)
st.sidebar.title("HELOC Assessment System")

# Sidebar Options
app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Run Assessment", "Batch Processing", "Model Info", "Debug Mode"])

# Load model resources
model, scaler, feature_names, metadata, threshold_df, recommended_threshold = load_model_resources()

# Initialize feature importance dictionary for PDF generation
feature_importance_dict = {}
if model is not None and metadata is not None and 'feature_list' in metadata:
    feature_importance_dict = dict(zip(metadata['feature_list'], model.feature_importances_))

# Set default threshold to recommended value
if 'custom_threshold' not in st.session_state or st.session_state.custom_threshold == 0.5:
    st.session_state.custom_threshold = recommended_threshold

# Sidebar controls for threshold
st.sidebar.subheader("Assessment Controls")
threshold_option = st.sidebar.radio(
    "Rejection Threshold Selection",
    ["Use Recommended (Optimal F1)", "Custom Threshold"]
)

if threshold_option == "Use Recommended (Optimal F1)":
    threshold = recommended_threshold
    st.sidebar.info(f"Using recommended threshold: {recommended_threshold:.2f}")
else:
    threshold = st.sidebar.slider("Custom Rejection Threshold", 0.0, 1.0, float(st.session_state.custom_threshold), 0.01)
    st.session_state.custom_threshold = threshold

# Debug Mode Toggle
debug_toggle = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
if debug_toggle != st.session_state.debug_mode:
    st.session_state.debug_mode = debug_toggle
    st.experimental_rerun()

# About page
if app_mode == "About":
    st.title("HELOC Application Assessment System")
    
    st.markdown("""
    ## Overview
    This application automates the assessment of Home Equity Line of Credit (HELOC) applications using a machine learning model.
    
    The system analyzes applicant data and provides an immediate decision on whether the loan application should be automatically
    rejected based on risk factors.
    
    ## Features
    - **Individual Assessment**: Evaluate a single loan application with detailed explanations
    - **Batch Processing**: Upload a CSV file with multiple applications for bulk assessment
    - **Custom Thresholds**: Adjust the rejection threshold based on business needs
    - **PDF Reports**: Generate downloadable assessment reports
    - **Model Insights**: View model performance metrics and key predictors
    
    ## How It Works
    The system uses an XGBoost classifier model trained on historical loan data to predict the likelihood
    of loan default. Applications with high default probability are automatically rejected.
    
    ## Getting Started
    Select "Run Assessment" from the sidebar to begin evaluating loan applications.
    """)
    
    st.info("This system is intended to be used as a decision support tool and should be used in conjunction with expert judgment.")

# Run Assessment page
elif app_mode == "Run Assessment":
    st.title("HELOC Application Assessment")
    
    if model is None or scaler is None or feature_names is None:
        st.error("Failed to load model resources. Please check the model files.")
    else:
        st.write("Enter applicant information to assess HELOC application risk.")
        
        # Quick Input Section
        with st.expander("Quick Input (Comma-separated values)"):
            st.write("Enter comma-separated values for quick input. Values should match the order of features shown below.")
            
            # Get expected features to show the order
            expected_features = feature_names["expected_features"]
            st.write("Features (in order):")
            st.code(", ".join(expected_features))
            
            # Example test cases
            st.subheader("Sample Test Cases:")
            
            # High risk/rejection example
            st.markdown("**Likely Rejection Example:**")
            rejection_example = "0.95, 0.98, 0.9, 25000, 450000, 0.95, 7, 12, 3, 0.98, 580, 0.85, 0.7, 3, 0.9"
            if len(expected_features) > 15:
                rejection_example += ", " + ", ".join(["0.95"] * (len(expected_features) - 15))
            st.code(rejection_example)
            st.markdown("*This example has very high debt ratios, numerous delinquencies, poor credit score, and inadequate income relative to loan amount - extremely likely to be rejected.*")
            
            # Low risk/approval example
            st.markdown("**Likely Approval Example:**")
            approval_example = "0.2, 0.3, 0.25, 125000, 200000, 0.3, 0, 0, 0, 0.2, 780, 0.15, 0.1, 0, 0.2"
            if len(expected_features) > 15:
                approval_example += ", " + ", ".join(["0.25"] * (len(expected_features) - 15))
            st.code(approval_example)
            st.markdown("*This example has low debt ratios, no delinquencies, good income - likely to be approved.*")
            
            # Input area
            csv_input = st.text_area("Paste comma-separated values here:")
            
            # Button to parse input
            if st.button("Fill Form with Values"):
                if csv_input:
                    try:
                        # Parse the CSV input
                        values = [float(x.strip()) for x in csv_input.split(',')]
                        
                        # Check if we have enough values
                        if len(values) < len(expected_features):
                            st.error(f"Not enough values provided. Expected {len(expected_features)} values, got {len(values)}.")
                        else:
                            # Store the values in session state to fill the form
                            st.session_state.form_prefill = dict(zip(expected_features, values[:len(expected_features)]))
                            st.success("Values parsed successfully! The form below has been filled with these values.")
                    except Exception as e:
                        st.error(f"Error parsing values: {e}")
        
        # Create columns for the form
        col1, col2 = st.columns(2)
        
        # Get expected features
        expected_features = feature_names["expected_features"]
        
        # Organize features into logical groups
        financial_features = [f for f in expected_features if any(x in f.lower() for x in ["income", "debt", "payment", "balance", "amount", "value"])]
        credit_features = [f for f in expected_features if any(x in f.lower() for x in ["credit", "loan", "delinq", "late", "default"])]
        other_features = [f for f in expected_features if f not in financial_features and f not in credit_features]
        
        # Create a dictionary to store form inputs
        form_data = {}
        
        # Display form
        with st.form("application_form"):
            st.subheader("Applicant Information")
            
            # Basic info (non-model features but useful for identification)
            applicant_name = st.text_input("Applicant Name (for reporting only)")
            
            st.subheader("Financial Information")
            for feature in financial_features:
                # Get default value from form_prefill if available
                default_value = st.session_state.form_prefill.get(feature, 0.0)
                form_data[feature] = st.number_input(
                    f"{feature}", 
                    value=default_value,
                    help=f"Enter value for {feature}"
                )
            
            st.subheader("Credit Information")
            for feature in credit_features:
                # Get default value from form_prefill if available
                default_value = st.session_state.form_prefill.get(feature, 0.0)
                form_data[feature] = st.number_input(
                    f"{feature}", 
                    value=default_value,
                    help=f"Enter value for {feature}"
                )
            
            st.subheader("Other Information")
            for feature in other_features:
                # Get default value from form_prefill if available
                default_value = st.session_state.form_prefill.get(feature, 0.0)
                form_data[feature] = st.number_input(
                    f"{feature}", 
                    value=default_value,
                    help=f"Enter value for {feature}"
                )
            
            submitted = st.form_submit_button("Assess Application")
        
        if submitted:
            # Create a DataFrame from form data
            application_df = pd.DataFrame([form_data])
            
            # Debug information
            if st.session_state.debug_mode:
                st.subheader("Debug Information")
                st.write("Application Data:")
                st.write(application_df)
                
                # Validate input
                valid, message = validate_input_data(application_df, expected_features)
                st.write(f"Validation: {message}")
                
                # Show scaled features
                probabilities, decisions = predict_application(model, scaler, application_df, threshold)
                scaled_features = scaler.transform(application_df)
                st.write("Scaled Features:")
                st.write(pd.DataFrame(scaled_features, columns=expected_features))
            
            # Make prediction
            probabilities, decisions = predict_application(model, scaler, application_df, threshold)
            
            if probabilities is not None and decisions is not None:
                # Store the application and assessment
                st.session_state.applications.append({
                    'name': applicant_name if applicant_name else f"Applicant {len(st.session_state.applications) + 1}",
                    'data': form_data,
                    'probability': probabilities[0],
                    'decision': decisions[0],
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Display results
                st.subheader("Assessment Result")
                
                # Result with color coding
                result_col1, result_col2 = st.columns([1, 3])
                with result_col1:
                    if decisions[0] == 1:
                        st.error("REJECTED")
                    else:
                        st.success("APPROVED")
                
                with result_col2:
                    st.write(f"Rejection Probability: {probabilities[0]:.2%}")
                    st.write(f"Threshold: {threshold:.2f}")
                    st.progress(float(probabilities[0]))
                
                # Generate explanation
                explanation = get_explanation(feature_importance_dict, probabilities[0], application_df)
                
                st.subheader("Key Factors")
                for line in explanation:
                    st.write(line)
                
                # Generate PDF report
                pdf_bytes = generate_pdf_report(
                    form_data, 
                    probabilities[0], 
                    decisions[0], 
                    explanation, 
                    applicant_name if applicant_name else "Applicant"
                )
                
                # Create download link
                st.markdown(
                    create_download_link(
                        pdf_bytes, 
                        f"assessment_report_{applicant_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
                    ), 
                    unsafe_allow_html=True
                )
                
                # Visual explanation with bar chart
                st.subheader("Feature Contribution Analysis")
                
                # Calculate feature contributions
                feature_contribution = {}
                for feature in expected_features:
                    value = application_df.iloc[0][feature]
                    importance = feature_importance_dict.get(feature, 0)
                    contribution = value * importance
                    feature_contribution[feature] = contribution
                
                # Get top contributing features (both positive and negative)
                sorted_contributions = sorted(feature_contribution.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                
                # Create DataFrame for plotting
                contrib_df = pd.DataFrame(sorted_contributions, columns=['Feature', 'Contribution'])
                
                # Create bar chart
                fig = px.bar(
                    contrib_df, 
                    x='Contribution', 
                    y='Feature',
                    orientation='h',
                    color='Contribution',
                    color_continuous_scale=['blue', 'gray', 'red'],
                    title='Top Features Impact on Rejection Probability'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

# Batch Processing page
elif app_mode == "Batch Processing":
    st.title("Batch Application Assessment")
    
    if model is None or scaler is None or feature_names is None:
        st.error("Failed to load model resources. Please check the model files.")
    else:
        st.write("Upload a CSV file with multiple applications for batch assessment.")
        
        # Get expected features
        expected_features = feature_names["expected_features"]
        
        # Upload form
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Load CSV data
                batch_df = pd.read_csv(uploaded_file)
                
                st.subheader("Uploaded Data Preview")
                st.write(batch_df.head())
                
                # Check for required columns
                valid, message = validate_input_data(batch_df, expected_features)
                
                if not valid:
                    st.error(message)
                    
                    # Show missing columns
                    missing_cols = set(expected_features) - set(batch_df.columns)
                    if missing_cols:
                        st.write("Missing columns:")
                        st.write(list(missing_cols))
                    
                    # In debug mode, show detailed comparison
                    if st.session_state.debug_mode:
                        st.subheader("Columns Comparison")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Expected Columns:")
                            st.write(expected_features)
                        
                        with col2:
                            st.write("Actual Columns:")
                            st.write(list(batch_df.columns))
                else:
                    if st.button("Run Batch Assessment"):
                        # Track start time for performance monitoring
                        start_time = time.time()
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Use predict_application instead of direct transformation
                        probabilities, decisions = predict_application(model, scaler, batch_df, threshold)
                        
                        if probabilities is not None and decisions is not None:
                            # Add results to dataframe
                            results_df = batch_df.copy()
                            results_df['rejection_probability'] = probabilities
                            results_df['decision'] = decisions
                            results_df['decision'] = results_df['decision'].map({1: 'REJECTED', 0: 'APPROVED'})
                            
                            # Calculate processing time
                            processing_time = time.time() - start_time
                            
                            # Update progress
                            progress_bar.progress(1.0)
                            status_text.success(f"Completed processing {len(batch_df)} applications in {processing_time:.2f} seconds")
                            
                            # Display results
                            st.subheader("Assessment Results")
                            st.write(results_df)
                            
                            # Summary statistics
                            st.subheader("Summary")
                            rejection_rate = (decisions.sum() / len(decisions)) * 100
                            st.write(f"Total Applications: {len(batch_df)}")
                            st.write(f"Rejected: {decisions.sum()} ({rejection_rate:.1f}%)")
                            st.write(f"Approved: {len(decisions) - decisions.sum()} ({100 - rejection_rate:.1f}%)")
                            
                            # Create a distribution chart
                            fig = px.histogram(
                                results_df, 
                                x='rejection_probability',
                                color='decision',
                                marginal='box',
                                title='Distribution of Rejection Probabilities',
                                labels={'rejection_probability': 'Rejection Probability', 'count': 'Number of Applications'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download links
                            csv = results_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            
                            st.markdown(
                                f'<a href="data:file/csv;base64,{b64}" download="batch_assessment_results.csv">Download Results CSV</a>',
                                unsafe_allow_html=True
                            )
                            
                            # Offer to generate individual PDFs for rejected applications
                            st.subheader("Generate Reports for Rejected Applications")
                            
                            if st.button("Generate PDF Reports"):
                                # Create a ZIP file containing individual PDF reports
                                from io import BytesIO
                                import zipfile
                                
                                # Create a buffer for the zip file
                                zip_buffer = BytesIO()
                                
                                # Create the ZIP file
                                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                    # Filter rejected applications
                                    rejected_df = results_df[results_df['decision'] == 'REJECTED']
                                    
                                    # Create progress bar
                                    pdf_progress = st.progress(0)
                                    pdf_status = st.empty()
                                    
                                    # Generate PDF for each rejected application
                                    for i, (idx, row) in enumerate(rejected_df.iterrows()):
                                        # Update progress
                                        pdf_progress.progress((i + 1) / len(rejected_df))
                                        pdf_status.text(f"Generating PDF {i + 1} of {len(rejected_df)}")
                                        
                                        # Extract application data
                                        app_data = row[expected_features].to_dict()
                                        
                                        # Generate explanation
                                        explanation = get_explanation(
                                            feature_importance_dict, 
                                            row['rejection_probability'], 
                                            batch_df.iloc[[idx]], 
                                            0
                                        )
                                        
                                        # Generate PDF
                                        pdf_bytes = generate_pdf_report(
                                            app_data,
                                            row['rejection_probability'],
                                            row['decision'] == 'REJECTED',
                                            explanation,
                                            f"Application_{idx}"
                                        )
                                        
                                        # Add to ZIP file
                                        zip_file.writestr(
                                            f"assessment_report_Application_{idx}.pdf",
                                            pdf_bytes
                                        )
                                    
                                    # Update status
                                    pdf_status.success(f"Generated {len(rejected_df)} PDF reports")
                                
                                # Download link for the ZIP file
                                zip_b64 = base64.b64encode(zip_buffer.getvalue()).decode()
                                zip_filename = f"rejection_reports_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
                                
                                st.markdown(
                                    f'<a href="data:application/zip;base64,{zip_b64}" download="{zip_filename}">Download All PDF Reports (ZIP)</a>',
                                    unsafe_allow_html=True
                                )
            
            except Exception as e:
                st.error(f"Error processing batch file: {e}")
                
                if st.session_state.debug_mode:
                    st.write("Detailed error:")
                    st.exception(e)

# Model Info page
elif app_mode == "Model Info":
    st.title("Model Information")
    
    if metadata is None or threshold_df is None:
        st.error("Failed to load model metadata. Please check the model files.")
    else:
        # Model overview
        st.subheader("Model Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", metadata["model_type"])
        with col2:
            st.metric("Model Version", metadata["model_version"])
        with col3:
            st.metric("Training Date", metadata["training_date"])
        
        # Model performance
        st.subheader("Model Performance")
        
        metrics = metadata["metrics"]
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with m_col2:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        with m_col3:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with m_col4:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        
        # Second row of metrics
        m2_col1, m2_col2 = st.columns(2)
        with m2_col1:
            st.metric("Specificity", f"{metrics['specificity']:.4f}")
        with m2_col2:
            st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        
        # Threshold analysis
        st.subheader("Threshold Analysis")
        
        # Interactive threshold plot
        fig = go.Figure()
        
        # Add lines for each metric
        for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1']:
            fig.add_trace(go.Scatter(
                x=threshold_df['threshold'],
                y=threshold_df[metric],
                mode='lines',
                name=metric.capitalize(),
                line=dict(width=2),
                hovertemplate=f'Threshold: %{{x:.2f}}<br>{metric.capitalize()}: %{{y:.4f}}'
            ))
        
        # Add vertical line for recommended threshold
        fig.add_vline(
            x=recommended_threshold,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Recommended: {recommended_threshold:.2f}",
            annotation_position="top right"
        )
        
        # Add vertical line for current threshold
        fig.add_vline(
            x=threshold,
            line_width=2,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Current: {threshold:.2f}",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title='Metrics Performance Across Different Thresholds',
            xaxis_title='Threshold',
            yaxis_title='Metric Value',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        
        # Get feature importances
        feature_importances = dict(zip(
            metadata['feature_list'],
            model.feature_importances_
        ))
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create DataFrame for top features
        top_features_df = pd.DataFrame(
            sorted_features[:15],
            columns=['Feature', 'Importance']
        )
        
        # Plot feature importance
        fig = px.bar(
            top_features_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Most Important Features',
            labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model parameters
        with st.expander("Model Parameters"):
            st.json(metadata["best_parameters"])
        
        # Dataset information
        with st.expander("Dataset Information"):
            st.write(f"Dataset Shape: {metadata['dataset_shape'][0]} rows, {metadata['dataset_shape'][1]} columns")
            st.write(f"Training Samples: {metadata['training_samples']}")
            st.write(f"Test Samples: {metadata['test_samples']}")
            st.write(f"Features Count: {metadata['features_count']}")
            st.write(f"Training Time: {metadata['training_time_seconds']:.2f} seconds")

# Debug Mode page
elif app_mode == "Debug Mode" and st.session_state.debug_mode:
    st.title("Debug Mode")
    
    st.warning("This mode is intended for developers and model administrators only.")
    
    # System information
    st.subheader("System Information")
    
    # Check if files exist
    model_file_exists = os.path.exists(MODEL_PATH)
    scaler_file_exists = os.path.exists(SCALER_PATH)
    feature_names_file_exists = os.path.exists(FEATURE_NAMES_PATH)
    metadata_file_exists = os.path.exists(METADATA_PATH)
    threshold_file_exists = os.path.exists(THRESHOLD_PATH)
    
    # Display status
    sys_col1, sys_col2 = st.columns(2)
    
    with sys_col1:
        st.write("File Status:")
        st.write(f"Model File: {'✅' if model_file_exists else '❌'}")
        st.write(f"Scaler File: {'✅' if scaler_file_exists else '❌'}")
        st.write(f"Feature Names File: {'✅' if feature_names_file_exists else '❌'}")
        st.write(f"Metadata File: {'✅' if metadata_file_exists else '❌'}")
        st.write(f"Threshold File: {'✅' if threshold_file_exists else '❌'}")
    
    with sys_col2:
        st.write("Resource Status:")
        st.write(f"Model Loaded: {'✅' if model is not None else '❌'}")
        st.write(f"Scaler Loaded: {'✅' if scaler is not None else '❌'}")
        st.write(f"Feature Names Loaded: {'✅' if feature_names is not None else '❌'}")
        st.write(f"Metadata Loaded: {'✅' if metadata is not None else '❌'}")
        st.write(f"Threshold Data Loaded: {'✅' if threshold_df is not None else '❌'}")
    
    # Model inspection
    if model is not None:
        st.subheader("Model Inspection")
        
        # Display model type and parameters
        st.write(f"Model Type: {type(model).__name__}")
        
        if hasattr(model, 'get_params'):
            st.json(model.get_params())
        
        # Try to display feature importances
        if hasattr(model, 'feature_importances_'):
            st.write("Feature Importances:")
            importances = model.feature_importances_
            
            if feature_names is not None:
                importance_df = pd.DataFrame({
                    'Feature': feature_names['expected_features'],
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(importance_df)
    
    # Scaler inspection
    if scaler is not None:
        st.subheader("Scaler Inspection")
        
        # Display scaler type and parameters
        st.write(f"Scaler Type: {type(scaler).__name__}")
        
        # Show mean and scale
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            scaler_df = pd.DataFrame({
                'Feature': feature_names['expected_features'] if feature_names is not None else [f"Feature_{i}" for i in range(len(scaler.mean_))],
                'Mean': scaler.mean_,
                'Scale': scaler.scale_
            })
            
            st.dataframe(scaler_df)
    
    # Application history
    if len(st.session_state.applications) > 0:
        st.subheader("Application History")
        
        history_df = pd.DataFrame([
            {
                'Name': app['name'],
                'Timestamp': app['timestamp'],
                'Probability': app['probability'],
                'Decision': 'REJECTED' if app['decision'] == 1 else 'APPROVED'
            }
            for app in st.session_state.applications
        ])
        
        st.dataframe(history_df)
        
        # Option to clear history
        if st.button("Clear Application History"):
            st.session_state.applications = []
            st.experimental_rerun()
    
    # Test prediction
    st.subheader("Test Prediction")
    
    if model is not None and scaler is not None and feature_names is not None:
        # Generate a random application for testing
        if st.button("Generate Random Test Application"):
            # Get expected features
            expected_features = feature_names["expected_features"]
            
            # Create random data within reasonable bounds
            random_data = {}
            for feature in expected_features:
                # Set different ranges based on feature names
                if "ratio" in feature.lower():
                    random_data[feature] = np.random.uniform(0, 1)
                elif "amount" in feature.lower() or "value" in feature.lower() or "balance" in feature.lower():
                    random_data[feature] = np.random.uniform(1000, 500000)
                elif "number" in feature.lower() or "count" in feature.lower():
                    random_data[feature] = np.random.randint(0, 10)
                else:
                    random_data[feature] = np.random.uniform(-2, 2)
            
            # Create DataFrame
            test_df = pd.DataFrame([random_data])
            
            # Use predict_application function
            probabilities, decisions = predict_application(model, scaler, test_df, threshold)
            
            if probabilities is not None and decisions is not None:
                # Display test data
                st.write("Test Application Data:")
                st.dataframe(test_df)
                
                # Display prediction
                st.write(f"Rejection Probability: {probabilities[0]:.4f}")
                st.write(f"Decision: {'REJECTED' if decisions[0] == 1 else 'APPROVED'}")
                
                # Show scaled data
                scaled_data = pd.DataFrame(
                    scaler.transform(test_df),
                    columns=expected_features
                )
                
                with st.expander("View Scaled Features"):
                    st.dataframe(scaled_data)

# If debug mode is disabled but user tries to access the debug page
elif app_mode == "Debug Mode" and not st.session_state.debug_mode:
    st.error("Debug Mode is not enabled. Please enable it in the sidebar.")
