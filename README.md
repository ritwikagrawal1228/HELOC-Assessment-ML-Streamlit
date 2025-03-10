• Developed and deployed a machine learning-powered HELOC Application Assessment System that automates credit decisioning by integrating advanced data cleaning, feature engineering, and categorical encoding with Python, Pandas, and NumPy to transform special symbolic values and handle missing data, thereby reducing manual review workload and boosting processing efficiency.

• Engineered a high-performance XGBoostClassifier—leveraging hyperparameter tuning (Max Depth = 9, 100 Decision Trees), threshold optimization, and SMOTE for imbalanced data—to auto-reject high-risk applications, which achieved precision of 94.06% and specificity of 99.40%, ensuring robust risk classification.

• Leveraged Azure AutoML to benchmark ensemble models (including logistic regression, decision trees, and random forests), ultimately selecting XGBoost for its superior AUC (0.8041), and integrated Scikit-learn’s StandardScaler for feature standardization to improve model performance.

• Implemented comprehensive model evaluation using ROC Curve analysis, Confusion Matrix metrics, and threshold sensitivity analysis to balance operational cost savings with risk minimization, optimizing auto-rejection thresholds that yielded a $30.6K annual net benefit at a 0.65 F1-score while mitigating potential lost loan profit of $2.02M at a 0.95 threshold.

• Developed an interactive Streamlit-based web application for real-time risk assessment, featuring batch processing of bulk HELOC applications, customizable rejection thresholds, and automated PDF report generation via FPDF, thereby streamlining underwriting workflows in financial services.

• Established robust logging and metadata tracking with Python’s logging module to document training parameters, system performance, and model metrics, ensuring reproducibility, regulatory compliance, and transparency through detailed feature importance analysis and decision report generation.

• Optimized data transformation by standardizing raw numeric inputs with StandardScaler and engineering domain-specific features—such as Boolean indicators for delinquency and target encoding for “RiskPerformance”—to enhance the predictive power of the automated credit decisioning pipeline.

• Enhanced model interpretability and stakeholder communication by generating interactive visualizations (using Plotly) and clear feature contribution analyses, which support dynamic business strategy alignment and regulatory transparency in the evolving HELOC lending landscape.


------------------------------------------------------
TO DO -

1. gitignore
2. Data Processing Pipeline
------------------------------------------------------
