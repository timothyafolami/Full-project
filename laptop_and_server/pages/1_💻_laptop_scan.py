import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
# import catboost
import catboost


st.title("IT preventative maintenance")

# Load the trained laptop model
laptop_model = joblib.load('laptop_model.joblib')

# Load the trained server model
server_model = joblib.load('server_model.joblib')

# laptop preprocessing

class LaptopDataPreprocessor:
    def __init__(self):
        # Define the preprocessing steps
        self.categorical_features = ['Laptop Model', 'Laptop Status', 'Manufacturer', 'Processor Type', 'Graphics Card', 'Bluetooth', 'Wi-Fi', 'Touch Screen']
        self.categorical_transformer = OneHotEncoder(drop='first')

        self.numerical_features = ['Disk Usage (%)', 'CPU Usage (%)', 'Memory Usage (%)', 'Screen Size (inch)', 'Battery Capacity (Wh)', 'Number of USB Ports', 'Weight (kg)']
        self.numerical_transformer = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', self.categorical_transformer, self.categorical_features),
                ('num', self.numerical_transformer, self.numerical_features)
            ])

    def fit_transform(self, df):
        # Apply preprocessing to the data and fit the transformer
        X_preprocessed = self.preprocessor.fit_transform(df)
        return X_preprocessed

    def transform(self, df):
        # Apply preprocessing to the data (without fitting the transformer)
        X_preprocessed = self.preprocessor.transform(df)
        return X_preprocessed

    def save(self, file_path):
        # Save the preprocessor pipeline to a file
        joblib.dump(self.preprocessor, file_path)

    @staticmethod
    def load(file_path):
        # Load the preprocessor pipeline from a file
        preprocessor = joblib.load(file_path)
        preprocessor_instance = LaptopDataPreprocessor()
        preprocessor_instance.preprocessor = preprocessor
        return preprocessor_instance

laptop_df = pd.read_csv("Laptop_log_data.csv")
laptop_df_x = laptop_df.drop(columns=['Laptop ID', 'Target'])
laptop_preprocessor = LaptopDataPreprocessor()
X_train = laptop_preprocessor.fit_transform(laptop_df_x)

# Save the preprocessor pipeline to a file
laptop_preprocessor.save('laptop_preprocessor_pipeline.joblib')

def preprocess_data(df):
    preprocessor = LaptopDataPreprocessor.load('laptop_preprocessor_pipeline.joblib')
    return preprocessor.transform(df)

# Load the server model
laptop_model = joblib.load('laptop_model.joblib')

with st.expander("Laptop Model"):
    st.write("This section is for detecting faults in Laprops.")
    st.write("Provide the following information for multiple laptops:")

    # Load your server data into a DataFrame
    laptop_df = pd.read_csv("Laptop_log_data.csv")
    laptop_data = laptop_df.copy()  # Use a copy of your server data
    st.dataframe(laptop_data.drop(columns=['Target']))

    # 
    laptop_df_x = laptop_df.drop(columns=['Laptop ID', 'Target'])

with st.expander("Laptop Scan"):
    if st.button("Scan"):
        # Check if there is any data to process
        if not laptop_data.empty:
            X_preprocessed = preprocess_data(laptop_df_x)
            server_predictions = laptop_model.predict(X_preprocessed)

            # Create a DataFrame to display predictions
            result_df = laptop_df[['Laptop ID']].copy()
            result_df['Prediction'] = server_predictions
            result = result_df[result_df['Prediction'] == 1]
            result['Prediction_category'] = "Faulty"
        
            st.dataframe(result)


st.markdown("---")
st.write("Automatic procedure generation, and fault detectors...")

# Optionally, you can add a sidebar for additional options, if needed
st.sidebar.title("Options")
st.sidebar.write("Add any additional options or information here.")

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.header("About")
    st.sidebar.info(
        "This app is split into 3 sections, the first is that it generates automatic procedures of doing a manual process."
        "The other aspects of this app detect faults in laptops and servers using pre-trained models. "
        "Provide the required information for each model and click the respective button to detect faults."
    )