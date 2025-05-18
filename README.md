# Forecasting-student-admission-with-deep-learning-regression-model

# Goal
The primary goal of this project is to develop a deep learning regression model that predicts the likelihood of a student’s admission to graduate school based on a variety of application features. By leveraging historical admissions data, the project aims to accurately estimate each applicant's chance of being accepted, providing valuable insights for both applicants and admissions officers.
\
Data taken from [Graduate Admission Kaggle dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions).

# Description
This Jupyter Notebook presents a step-by-step workflow for forecasting graduate school admissions using a neural network regression approach. The dataset used is sourced from Kaggle’s "Graduate Admissions" collection, which includes features such as GRE Score, TOEFL Score, University Rating, SOP and LOR strengths, CGPA, and Research experience.

The workflow is as follows:

- Data Loading and Preprocessing: The admissions dataset is imported and relevant features are standardized to ensure optimal neural network performance.
- Model Construction: A sequential deep learning model is built using TensorFlow/Keras. The architecture comprises an input layer, a hidden layer with ReLU activation, and an output layer suited for regression tasks.
- Model Training: The model is trained using mean squared error loss and incorporates early stopping to prevent overfitting. Training and validation losses are visualized for analysis.
- Evaluation: Model performance is assessed on test data using metrics such as Mean Absolute Error (MAE) and the coefficient of determination (R² score), providing a quantitative measure of prediction accuracy.
This notebook serves as a practical example of applying deep learning to educational data analytics, demonstrating all key steps from preprocessing to model evaluation.

# Tools
- Python (Jupyter Notebook): For interactive, reproducible data science and machine learning workflows.
- Pandas & NumPy: For data manipulation and numerical operations.
- Matplotlib: For data visualization and plotting training results.
- scikit-learn: For data preprocessing (standardization, train-test split) and evaluation metrics (R² score).
- TensorFlow & Keras: For building, training, and evaluating the deep learning regression model.
