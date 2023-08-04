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
import catboost
from apikey import OPENAI_API_KEY
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

################################################
#######Automatic Process##############
################################################

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAI(openai_api_key=API_KEY)


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

# Function to make predictions using the laptop model
def predict_laptop_fault(features):
    # loading the preprocessing pipeline

    preprocessor = LaptopDataPreprocessor.load('laptop_preprocessor_pipeline.joblib')
    preprocess = preprocessor.transform(features)
    laptop_model = joblib.load('laptop_model.joblib')
    return laptop_model.predict(preprocess)


# -------------------------------------------------------
class ServerDataPreprocessor:
    def __init__(self):
        # Define the preprocessing steps
        self.categorical_features = ['Server Status', 'Operating System', 'Server Location']
        self.categorical_transformer = OneHotEncoder(drop='first')

        self.numerical_features = ['Disk Usage (%)', 'CPU Usage (%)', 'Memory Usage (%)', 'Number of CPU Cores', 'RAM Capacity (GB)', 'Network Traffic (Mbps)', 'Disk I/O (IOPS)', 'Server Uptime (days)']
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
        preprocessor_instance = ServerDataPreprocessor()
        preprocessor_instance.preprocessor = preprocessor
        return preprocessor_instance

# Load the data
server_df = pd.read_csv("Server_log_data.csv")
server_df_x = server_df.drop(columns=['Server ID', 'Server Name', 'Target'])
# Create an instance of the ServerDataPreprocessor and fit_transform the data
server_preprocessor = ServerDataPreprocessor()
X_train = server_preprocessor.fit_transform(server_df_x)

# Save the preprocessor pipeline to a file
server_preprocessor.save('server_preprocessor_pipeline.joblib')

# Function to make predictions using the server model
def predict_server_fault(features):
    # loading the preprocessing pipeline
    preprocessor = ServerDataPreprocessor.load('server_preprocessor_pipeline.joblib')
    preprocess = preprocessor.transform(features)
    server_model = joblib.load('server_model.joblib')
    return server_model.predict(preprocess)

# --------------------------
# Streamlit app code
st.set_page_config(page_title='Fault Detection App', page_icon=':computer:', layout='wide')

st.title("Fault Detection App")

prompt_template = PromptTemplate(
    template = "Describe an automatic procedure of doing this: {manual_process}",
    input_variables = ['manual_process']
)

manual_chain = LLMChain(
    llm = llm,
    prompt = prompt_template,
    verbose = True
)


with st.expander("Automatic Procedure"):
    user_prompt = st.text_input("What is the manual process")

    if st.button("Generate") and user_prompt:
        with st.spinner("Generating..."):
            output = manual_chain.run(manual_process=user_prompt)
            st.write(output)

# Section for Laptop Model
with st.expander("Laptop Model"):
    st.write("This section is for detecting faults in laptops.")
    st.write("Provide the following information:")
    
    # Widgets for user input related to laptops
    laptop_data = {}
    laptop_data['Laptop Model'] = st.selectbox("Laptop Model", laptop_df['Laptop Model'].unique())
    laptop_data['Laptop Status'] = st.selectbox("Laptop Status", laptop_df['Laptop Status'].unique())
    laptop_data['Fan Faulty'] = st.slider("Fan Faulty", 0, 1, 0)
    laptop_data['Disk Usage (%)'] = st.slider("Disk Usage (%)", 10.01, 95.0, 50.0)
    laptop_data['CPU Usage (%)'] = st.slider("CPU Usage (%)", 10.01, 95.0, 50.0)
    laptop_data['Memory Usage (%)'] = st.slider("Memory Usage (%)", 10.0, 95.0, 50.0)
    laptop_data['Manufacturer'] = st.selectbox("Manufacturer", laptop_df['Manufacturer'].unique())
    laptop_data['Processor Type'] = st.selectbox("Processor Type", laptop_df['Processor Type'].unique())
    laptop_data['Screen Size (inch)'] = st.slider("Screen Size (inch)", 13, 17, 15)
    laptop_data['Battery Capacity (Wh)'] = st.slider("Battery Capacity (Wh)", 40, 70, 50)
    laptop_data['Number of USB Ports'] = st.slider("Number of USB Ports", 2, 4, 3)
    laptop_data['Graphics Card'] = st.selectbox("Graphics Card", laptop_df['Graphics Card'].unique())
    laptop_data['Bluetooth'] = st.selectbox("Bluetooth", laptop_df['Bluetooth'].unique())
    laptop_data['Wi-Fi'] = st.selectbox("Wi-Fi", laptop_df['Wi-Fi'].unique())
    laptop_data['Touch Screen'] = st.selectbox("Touch Screen", laptop_df['Touch Screen'].unique())
    laptop_data['Weight (kg)'] = st.slider("Weight (kg)", 1.0, 2.5, 1.5)
    
    if st.button("Detect Laptop Fault"):
        laptop_features = pd.DataFrame([laptop_data])
        laptop_prediction = predict_laptop_fault(laptop_features)
        if laptop_prediction[0] == 1:
            # Condition for "Faulty" prediction (1)
            st.write('<div style="background-color: red; padding: 10px; border-radius: 5px; color: white;">Faulty</div>', unsafe_allow_html=True)
        else:
            # Condition for "Good" prediction (0)
            st.write('<div style="background-color: green; padding: 10px; border-radius: 5px; color: white;">Good!</div>', unsafe_allow_html=True)
        
# Section for Server Model
with st.expander("Server Model"):
    st.write("This section is for detecting faults in servers.")
    st.write("Provide the following information:")
    
    # Widgets for user input related to servers
    server_data = {}
    # server_data['Server ID'] = st.text_input("Server ID")
    # server_data['Server Name'] = st.selectbox("Server Name", server_df['Server Name'].unique())
    server_data['Server Status'] = st.selectbox("Server Status", server_df['Server Status'].unique())
    server_data['Disk Usage (%)'] = st.slider("Disk Usage (%)", 10.01, 94.99, 50.0, key="server_disk_usage")
    server_data['CPU Usage (%)'] = st.slider("CPU Usage (%)", 10.01, 94.99, 50.0, key="server_cpu_usage")
    server_data['Memory Usage (%)'] = st.slider("Memory Usage (%)", 10.0, 95.0, 50.0, key="server_memory_usage")
    server_data['Operating System'] = st.selectbox("Operating System", server_df['Operating System'].unique())
    server_data['Number of CPU Cores'] = st.slider("Number of CPU Cores", 4, 16, 8)
    server_data['RAM Capacity (GB)'] = st.slider("RAM Capacity (GB)", 8, 64, 16)
    server_data['Network Traffic (Mbps)'] = st.slider("Network Traffic (Mbps)", 10.0, 100.0, 50.0)
    server_data['Disk I/O (IOPS)'] = st.slider("Disk I/O (IOPS)", 10.0, 99.97, 50.0)
    server_data['Server Location'] = st.selectbox("Server Location", server_df['Server Location'].unique())
    server_data['Server Uptime (days)'] = st.slider("Server Uptime (days)", 1, 365, 30)
    
    if st.button("Detect Server Fault"):
        server_features = pd.DataFrame([server_data])
        server_prediction = predict_server_fault(server_features)
        if server_prediction[0] == 1:
            # Condition for "Faulty" prediction (1)
            st.write('<div style="background-color: red; padding: 10px; border-radius: 5px; color: white;">Faulty</div>', unsafe_allow_html=True)
        else:
            # Condition for "Good" prediction (0)
            st.write('<div style="background-color: green; padding: 10px; border-radius: 5px; color: white;">Good!</div>', unsafe_allow_html=True)


st.markdown("---")
st.write("Automatic procedure generation, and fault detectors...")

# Optionally, you can add a sidebar for additional options, if needed
st.sidebar.title("Options")
st.sidebar.write("Add any additional options or information here.")

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.header("About")
    st.sidebar.info(
        "This app is splitted into 3 sections, the first is that it generates automatic procedures of doing a manual process."
        "The other aspects oftThis app detects faults in laptops and servers using pre-trained models. "
        "Provide the required information for each model and click the respective button to detect faults."
    )


    
