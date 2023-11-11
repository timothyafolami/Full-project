import streamlit as st
import streamlit.components.v1 as components

st.title("Business Automation Bot Integration")

# Adding a brief introduction about the business automation bot
st.markdown("""
    ### Your Digital Assistant for Business Efficiency 🚀
    Welcome to our Business Automation Bot! This intelligent assistant is here to streamline your business operations, answer queries, and automate routine tasks. Click the button below to interact with the bot and experience smart automation solutions that transform your business processes.
    ---
""")

# Custom HTML and JavaScript to open a new window with the bot URL
bot_url = "https://mediafiles.botpress.cloud/3146e631-a448-4838-8fa5-950b8a951611/webchat/bot.html"
html = f"""
    <div style="text-align: center"> <!-- Center alignment for the button -->
        <script type="text/javascript">
            function openBot() {{
                window.open("{bot_url}", "Chatbot", "width=800,height=600"); // Adjusted size
                return false;
            }}
        </script>
        <button onclick="openBot()">Talk to our Business Automation Bot</button>
    </div>
    """

components.html(html, height=100)
