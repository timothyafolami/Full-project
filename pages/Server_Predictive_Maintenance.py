import streamlit as st
import joblib
import pandas as pd
import sklearn
from catboost import CatBoostClassifier


# creating a function to preprocess the data
def data_prep(df):
    # load the preprocessor pipeline
    preprocessor = joblib.load('server_preprocessor_pipeline.joblib')
    # now preprocessing
    X_preprocessed = preprocessor.transform(df)
    # return the preprocessed data
    return X_preprocessed

# Function to load models and data
def load_data_and_models():
    try:
        laptop_data = pd.read_csv("Server_log_data.csv")
        laptop_model = joblib.load('server_model.joblib')
        return laptop_data, laptop_model
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None, None

server_data, server_model = load_data_and_models()


# Streamlit UI
st.title("Predictive Monitoring")

# Sidebar
st.sidebar.title("Options")
st.sidebar.write("Additional informations can come here.")

with st.expander("Server Predictive Maintenance"):

    st.write("This section is for detecting faults in servers.")
    st.write("Provide the following information for multiple servers:")

    if server_data is not None:

        # displaying the data
        st.dataframe(server_data.drop(columns=['Target']))

        if st.button("Scan"):
            # dropping some columns
            server_df_x = server_data.drop(columns=['Server ID', 'Server Name', 'Target'])

            # Check if there is any data to process
            X_preprocessed = data_prep(server_df_x)
            # model prediction
            server_predictions = server_model.predict(X_preprocessed)

            # Create a DataFrame to display predictions
            result_df = server_data[['Server ID']].copy()
            result_df['Prediction'] = server_predictions
            result = result_df[result_df['Prediction'] == 1]
            result['Prediction_category'] = "Faulty"
            # preparing the result
            output = result.drop(columns=['Prediction'])
            # displaying the result
            st.dataframe(output)
    else:
            st.write("No data available for analysis.")


st.markdown("---")
st.write("Automatic procedure generation, and fault detectors...")

# Optionally, you can add a sidebar for additional options, if needed
st.sidebar.title("Options")
st.sidebar.write("This is a demo of how predictive maintenance can be used to detect faults in servers.")

# Conclusion and additional notes
st.markdown("## Conclusion")
st.write("This tool aids in IT Predictive maintenance by analyzing and predicting potential faults in servers.")

