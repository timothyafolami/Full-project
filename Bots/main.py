import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI BOTs",
    page_icon="🤖",
)

# Main Page Title
st.title("Welcome to Our Interactive Bot Platform!")

# Sidebar
st.sidebar.success("Select a page above.")

# Business Automation Bot Introduction
st.markdown("""
    ## Welcome to Your Business Automation Assistant!

    In the dynamic world of business, efficiency and prompt information access are key. That's where our Business Automation Bot steps in – your go-to digital assistant for streamlining operations and enhancing productivity. Whether it's answering frequent queries, automating routine tasks, or collecting vital business insights, our bot is engineered to assist you in real-time. Say goodbye to manual processes and hello to smart automation. Let's transform the way you work, one question at a time.

    ---
""")

# School Students Bot Introduction
st.markdown("""
    ## Hello, Students! Meet Your Academic Companion!

    Embark on a journey of seamless learning with our School Students Bot, your personal academic assistant. Designed to cater to curious minds, this bot is here to answer your study-related queries, help with homework, and offer educational support. It's not just about answering questions; it's about making learning interactive and fun. From collecting useful information to aiding in note-taking, this bot is your partner in academic success. Let's make learning an adventure together!

    ---
""")
