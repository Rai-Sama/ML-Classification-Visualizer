import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from io import StringIO
import shap

# Title of the app
st.title("ML Classification Visualizer")

# Sidebar for dataset upload
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load dataset function
def load_data():
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        # Default to Iris dataset
        from sklearn import datasets
        iris = datasets.load_iris(as_frame=True)
        data = pd.concat([iris.data, iris.target.rename("class")], axis=1)
    return data

# Load dataset and display
df = load_data()
st.write("### Dataset Preview")
st.dataframe(df.head())

# Data summary
st.write("### Data Summary")
st.write(df.describe(include='all'))

# Missing Data Visualizer
with st.expander("Missing Data"):
    missing_values = df.isnull().sum() / len(df) * 100
    st.bar_chart(missing_values)

# Sidebar for feature selection
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect("Select features", df.columns[:-1], default=list(df.columns[:-1]))
target = st.sidebar.selectbox("Select target", df.columns, index=len(df.columns) - 1)

# Handle missing values
handle_missing = st.sidebar.selectbox("Handle Missing Data", ["None", "Fill with Mean", "Fill with Median", "Drop Missing"])
if handle_missing == "Fill with Mean":
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
elif handle_missing == "Fill with Median":
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
elif handle_missing == "Drop Missing":
    df.dropna(inplace=True)

# Automatically detect categorical features and apply encoding
categorical_cols = df[features].select_dtypes(include=['object']).columns.tolist()

# Apply one-hot encoding to categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Label encode the target if it's categorical
if df[target].dtype == 'object':
    le = LabelEncoder()
    df_encoded[target] = le.fit_transform(df[target])

# Select the updated features and target after encoding
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# Sidebar to select the ML algorithm
st.sidebar.header("Choose Algorithm")
algorithm = st.sidebar.selectbox("Select ML Algorithm", ["Decision Tree", "Logistic Regression", "Random Forest", "SVM"])

# Sidebar for model parameters depending on the selected algorithm
st.sidebar.header("Model Parameters")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifier based on selection
if algorithm == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)
    max_features = st.sidebar.selectbox("Max Features", [None, "sqrt", "log2"], index=0)
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"], index=0)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, 
                                 criterion=criterion, min_samples_leaf=min_samples_leaf, max_features=max_features)

elif algorithm == "Logistic Regression":
    penalty = st.sidebar.selectbox("Penalty", ["l2", "none"])
    C = st.sidebar.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
    solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
    clf = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000)

elif algorithm == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

elif algorithm == "SVM":
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
    C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

# Train the selected classifier
clf.fit(X_train, y_train)

# Evaluate performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Cross-validation
cross_val = st.sidebar.checkbox("Cross-validation", value=False)
if cross_val:
    cv_scores = cross_val_score(clf, X, y, cv=5)
    st.write(f"### Cross-validation Accuracy: {cv_scores.mean():.2f}")

# Confusion Matrix
with st.expander("Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# Classification Report
with st.expander("Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# Feature Importance (only for Decision Tree and Random Forest)
if algorithm in ["Decision Tree", "Random Forest"]:
    with st.expander("Feature Importance"):
        importance = pd.Series(clf.feature_importances_, index=X.columns)
        fig2, ax2 = plt.subplots()
        importance.plot(kind='bar', ax=ax2)
        st.pyplot(fig2)

# Visualize the decision tree (only for Decision Tree)
if algorithm == "Decision Tree":
    with st.expander("Decision Tree Visualization"):
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data, feature_names=X.columns, 
                        class_names=[str(c) for c in set(y)], filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data.getvalue())
        st.graphviz_chart(graph)

# Download tree as DOT format (only for Decision Tree)
if algorithm == "Decision Tree":
    st.sidebar.download_button(label="Download Tree (DOT format)", data=dot_data.getvalue(), file_name="decision_tree.dot", mime="text/plain")

# ROC Curve and AUC
if len(set(y_test)) == 2:  # Only for binary classification
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        y_prob = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    st.write(f"### AUC: {auc_score:.2f}")
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    ax3.plot([0, 1], [0, 1], 'k--')
    st.pyplot(fig3)

# SHAP Values (only for tree-based models)
if algorithm in ["Decision Tree", "Random Forest"]:
    with st.expander("SHAP Values"):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        st.pyplot(bbox_inches='tight')

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5)
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
fig4, ax4 = plt.subplots()
ax4.plot(train_sizes, train_mean, label="Training score")
ax4.plot(train_sizes, test_mean, label="Cross-validation score")
with st.expander("Learning Curve"):
    st.pyplot(fig4)