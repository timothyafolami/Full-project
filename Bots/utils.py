import openai

def get_initial_message():
    messages=[
            {"role": "system", "content":"""
    You are SchoolBot. Your responsibility to receieve informations from users (students) based on what they want to do.\
    You collect the following name:\
        1) Name
        2) Email
        3) Department
        4) Query
        5) Other necessary information. ...
    """}]
    return messages

def get_chatgpt_response(messages, model="gpt-3.5-turbo", streaming=None):
    # print("model: ", model)
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    stream = streaming
    )
    return  response['choices'][0]['message']['content']

def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages
