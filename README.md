# ML Classification Visualizer

Deployed demo: https://ml-classification-visualizer.streamlit.app/

## Overview

The ML Classification Visualizer is a Streamlit application designed to help users visualize and evaluate machine learning classification models. This application supports various machine learning algorithms, enabling users to upload their datasets, preprocess the data, train models, and visualize key performance metrics.

The aim with this project is to provide a lightweight tool that helps the user understand their dataset and the affects of standard model parameters better.

## Features

- Upload and explore datasets in CSV format.
- Visualize missing data through bar charts.
- Select features and target variables for model training.
- Handle missing values with various options (fill with mean/median or drop).
- Train and evaluate models using different algorithms:
  - Decision Tree
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- Visualize model performance metrics, including:
  - Accuracy and confusion matrix
  - Classification report
  - Feature importance
  - ROC curve and AUC for binary classification
  - SHAP values for tree-based models
  - Learning curves
- Interactive and user-friendly interface built with Streamlit.

## Installation

To set up the project, follow these steps:

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Graphviz
- SHAP

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ml-classification-visualizer.git
cd ml-classification-visualizer
```

### Step 2: Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Step 3: Install Required Packages
You can install the required packages using pip:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn graphviz shap
```
Or you can run the below command to install dependencies using requirements.txt (ensure to cd into the repo folder):

```bash
pip install -r requirements.txt
```

Step 4: Run the Application
To start the Streamlit application, run the following command:

```bash
streamlit run app.py
```

The application will open in your default web browser, typically at http://localhost:8501.

### Project Structure
The project is organized into the following modules:

```graphql
ml-classification-visualizer/
│
├── main.py             # The main Streamlit application logic.
├── README.md           # Project documentation.
|__ dataset.csv         # Sample dataset to try the tool with
```

### Usage
 - Upload Dataset: In the application, upload your CSV file containing the dataset you wish to analyze.
 - Explore Dataset: Preview the dataset and view basic statistics.
 - Missing Data: Visualize missing data and choose how to handle it.
 - Feature Selection: Select the features and target variable for model training.
 - Choose Algorithm: Select a machine learning algorithm and configure model parameters.
 - Train Model: Train the selected model on the dataset.
 - View Results: The dashboard will display:
   - Model accuracy.
   - Confusion matrix.
   - Classification report.
   - Feature importance (for Decision Tree and Random Forest).
   - ROC curve and AUC (for binary classification).
   - SHAP values (for tree-based models).
   - Learning curve.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and create a pull request.

### Contact
For any questions or feedback, please reach out to me at anshumancos3@gmail.com


Feel free to modify any sections to better fit your project's specific details or add any additional features you might be able to include!
