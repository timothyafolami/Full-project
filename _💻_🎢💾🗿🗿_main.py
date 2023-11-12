import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Tech Fault Detection Suite",
    page_icon="🔍",
)

# Main Page Title
st.title("Welcome to the Tech Fault Detection Suite!")

# Sidebar
st.sidebar.success("Select a page above.")

# Laptop Fault Detection Introduction
st.markdown("""
    ## Laptop Fault Detection

    In the fast-paced world of technology, maintaining your laptop in top condition is crucial. Our Laptop Fault Detection tool is here to assist you in diagnosing and identifying issues with your laptop. Leveraging advanced diagnostics, this tool helps in pinpointing problems based on your laptop's features and symptoms. It's the perfect assistant for quick and accurate fault detection, keeping your device running smoothly.

    ---
""")

# Server Fault Prediction Introduction
st.markdown("""
    ## Server Fault Prediction

    For modern businesses and IT infrastructures, server reliability is a top priority. Our Server Fault Prediction model is designed to anticipate and identify potential server issues before they become critical. By analyzing server parameters and performance data, this tool provides valuable insights into the health of your server, helping to prevent downtime and maintain seamless operations. Stay ahead of server issues with our predictive diagnostics.

    ---
""")

# School Students Bot Introduction
st.markdown("""
    ## Hello, Students! Meet Your Academic Companion!

    Embark on a journey of seamless learning with our School Students Bot, your personal academic assistant. Designed to cater to curious minds, this bot is here to answer your study-related queries, help with homework, and offer educational support. It's not just about answering questions; it's about making learning interactive and fun. From collecting useful information to aiding in note-taking, this bot is your partner in academic success. Let's make learning an adventure together!

    ---
""")

# Business Automation Bot Introduction
st.markdown("""
    ## Welcome to Your Business Automation Assistant!

    In the dynamic world of business, efficiency and prompt information access are key. That's where our Business Automation Bot steps in – your go-to digital assistant for streamlining operations and enhancing productivity. Whether it's answering frequent queries, automating routine tasks, or collecting vital business insights, our bot is engineered to assist you in real-time. Say goodbye to manual processes and hello to smart automation. Let's transform the way you work, one question at a time.

    ---
""")
