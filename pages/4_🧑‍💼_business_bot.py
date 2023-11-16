import streamlit as st
import streamlit.components.v1 as components

# Custom HTML for Botpress widget
botpress_html = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.botpress.cloud/webchat/v0/inject.js"></script>
</head>
<body>
    <script>
        window.botpressWebChat.init({
            botId: 'your_bot_id',
            hostUrl: 'https://cdn.botpress.cloud/webchat/v1',
            messagingUrl: 'https://messaging.botpress.cloud',
            clientId: 'your_client_id'
        });
    </script>
</body>
</html>
"""

# Replace 'your_bot_id' and 'your_client_id' with actual values

# Streamlit app content
st.title("Botpress Chat Integration")

# Embed the Botpress widget
components.html(botpress_html, height=600)
