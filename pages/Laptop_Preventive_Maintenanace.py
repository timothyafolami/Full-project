import streamlit as st
import joblib
import pandas as pd


# creating a function to preprocess the data
def data_prep(df):
    # load the preprocessor pipeline
    preprocessor = joblib.load('laptop_preprocessor_pipeline.joblib')
    # now preprocessing
    X_preprocessed = preprocessor.transform(df)
    # return the preprocessed data
    return X_preprocessed
        

# Function to load models and data
def load_data_and_models():
    try:
        laptop_data = pd.read_csv("Laptop_log_data.csv")
        laptop_model = joblib.load('laptop_model.joblib')
        return laptop_data, laptop_model
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None, None

laptop_data, laptop_model = load_data_and_models()

# Streamlit UI
st.title("Laptop Preventative Maintenance")

# Sidebar
st.sidebar.title("Options")
st.sidebar.write("Add any additional options or information here.")

# Laptop Fault Detection
with st.expander("Laptop Preventative Maintenance"):

    st.write("This section is for detecting faults in laptops.")

    if laptop_data is not None:
        # displaying the data
        st.dataframe(laptop_data.drop(columns=['Target']))

        if st.button("Scan Laptops"):

            # dropping id and target columns
            laptop_df_x = laptop_data.drop(columns=['Laptop ID', 'Target'])
            # data preprocessing
            X_preprocessed = data_prep(laptop_df_x)

            # model prediction
            predictions = laptop_model.predict(X_preprocessed)

            # Create a DataFrame to display predictions
            result_df = laptop_data[['Laptop ID']].copy()
            result_df['Prediction'] = predictions
            result_df['Prediction_category'] = result_df['Prediction'].apply(lambda x: "Faulty" if x == 1 else "Normal")
            output = result_df[result_df['Prediction']==1]

            # displaying the result
            st.dataframe(output.drop(columns=['Prediction']))
    else:
        st.write("No data available for analysis.")

# Additional Features
st.markdown("---")
st.write("This is a demo of how preventive maintenance can be used to detect faults in laptops. ")

# Sidebar additional information
st.sidebar.header("About")
st.sidebar.info(
    "This app is designed to detect faults in laptops and servers using trained models. "
    # "Upload or enter your data for analysis."
)

# Conclusion and additional notes
st.markdown("## Conclusion")
st.write("This tool aids in IT preventative maintenance by analyzing and predicting potential faults in laptops.")


