# Import the required libraries for your requirement
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model App title
st.title("Automated Machine Learning Model APP")
# Here you can upload the Dataset ot file
uploaded_file = st.file_uploader("Choose the CSV file or Excel file for the Predictions.", type="csv")


if uploaded_file is not None:
    # Now you can load your dataset here
    df = pd.read_csv(uploaded_file)
    st.write("Your Data is under review process. So, please wait for some time.")
    st.dataframe(df.head())

    # Here you can able to see your dataset infom
    st.write("Dataset Information")
    st.write(df.info())

    # Check for null values
    st.write("Null Values in the Dataset")
    st.write(df.isnull().sum())

    # Handling missing values
    if st.checkbox("Handle Null Values (Remove or Fill)", True):
        st.write("Filling Null Values with Mean (for numeric columns)")
        df = df.fillna(df.mean())
        st.write("Null values have been filled.")

        
        # Check for categorical columns and apply Label Encoding or One-Hot Encoding
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            st.write("Categorical Columns Found")
            st.write(categorical_columns)


            # Option for Label Encoding or One-Hot Encoding
            encoding_method = st.radio("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
            if encoding_method == "Label Encoding":
                label_encoders = {}
                for col in categorical_columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    label_encoders[col] = le
                    st.write("Label Encoding applied to categorical columns.")
            elif encoding_method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=categorical_columns)
                st.write("One-Hot Encoding applied to categorical columns.")


                # Data visualization
                if st.checkbox("Show Correlation Heatmap"):
                    st.write("Correlation Heatmap")
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                    st.pyplot(plt)
                    if st.checkbox("Show Pair Plot"):
                        st.write("Pair Plot of the Data")
                        sns.pairplot(df)
                        st.pyplot(plt)


                        # Select target column
                        target_column = st.selectbox("Select the Target Column", df.columns)
                        
                        # Split data into training and testing sets
                        X = df.drop(target_column, axis=1)
                        y = df[target_column]
                        test_size = st.slider("Select Test Size", 0.1, 0.5, 0.3)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        st.write("Data Split")
                        st.write(f"Training set size: {len(X_train)}")
                        st.write(f"Test set size: {len(X_test)}")

                        # Feature Scaling
                        if st.checkbox("Apply Feature Scaling"):
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                            st.write("Feature scaling applied.")
                            
                            # Model training
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            
                            
                            # Make Predictions
                            y_pred = model.predict(X_test)
                            
                            # Display accuracy and performance metrics
                            st.write("Model Performance by using Classification report")
                            accuracy = accuracy_score(y_test, y_pred)
                            st.write(f"Accuracy: {accuracy:.2f}")
                            st.text("Classification Report:")
                            st.text(classification_report(y_test, y_pred))
                            
                            
                            # Display confusion matrix
                            st.write("Confusion Matrix")
                            conf_matrix = confusion_matrix(y_test, y_pred)
                            st.write(conf_matrix)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                            st.pyplot(fig)
                            
                            # Option to download the trained model
                            if st.button("Download Trained Model"):
                                joblib.dump(model, "trained_model.pkl")
                                st.write("Model saved as 'trained_model.pkl'") 