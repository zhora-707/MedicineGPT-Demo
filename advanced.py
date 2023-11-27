import os
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = load_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            if "lousy" in inp and "persistent cough" in inp:
                output = 'How long have you had these symptoms?'
            elif "thought" in inp and "regular cold" in inp:
                output = 'Have you been experiencing any fever or difficulty breathing?'
            elif "breathing feels" in inp and "harder" in inp:
                output = 'Considering the tightness in your chest and difficulty breathing, ' \
                         'it is important to monitor these symptoms closely. ' \
                         'Try using a humidifier or taking steamy showers to ease the discomfort in your chest. ' \
                         'Also, consider taking over-the-counter medications suitable for cough and chest congestion,' \
                         'but if the symptoms persist or worsen, seeking medical advice promptly would be advisable.'
            elif "Thanks" in inp and "tips" in inp:
                output = 'Take care of yourself and keep an eye on any changes. ' \
                         'If it does not improve, do not hesitate to consult with a healthcare professional. ' \
                         'Wishing you a speedy recovery!'
            else:
                output = chain.run(input=inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

#block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

# Add custom CSS to style the Gradio interface components

block = gr.Blocks(css=".gradio-interface {background-color: transparent;} .gradio-interface input, .gradio-interface button {border-color: #ccc;}")


with block:
    with gr.Row():
        gr.Markdown("<h3><center>MedicineGPT</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        )
    with gr.Row():
        message = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
                container=False,
            )
        # submit = gr.Button(value="Send", variant="secondary", css_class_name="custom-button")
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
        submit = gr.Button(value="Send", variant="secondary")#.style(full_width=False)

    gr.Examples(
        examples=[
            "Can I get some medicine recomendations?",
            "What should I do tonight?",
            "Is everything serious?",
            "I have a headache!",
            "I feel like I have a fever"
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a MedGPT")

    gr.HTML(
        "<center>Powered by <img src='https://raw.githubusercontent.com/zhora-707/MedicineGPT-Demo/main/logo.jpg' "
        "alt='MedGPT Logo'>MedGPT🔗</img></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    # submit = gr.Button(value="Send", variant="secondary", css_class_name="custom-button")

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )



block.launch(debug=True)

# sirun

# import gradio as gr
# import os
# import openai
# import time
#
# openai.api_key = os.environ.get('OPENAI_API_KEY')  # Replace with your key
#
# def add_text(history, text):
#     history = history + [(text, None)]
#     return history, gr.Textbox(value="", interactive=False)
#
# def add_file(history, file):
#     history = history + [((file.name,), None)]
#     return history
#
# def predict(message, history):
#     history_openai_format = []
#     for human, assistant in history:
#         history_openai_format.append({"role": "user", "content": human})
#         history_openai_format.append({"role": "assistant", "content": assistant})
#     history_openai_format.append({"role": "user", "content": message})
#
#     response = openai.ChatCompletion.create(
#         model='gpt-3.5-turbo',
#         messages=history_openai_format,
#         temperature=1.0,
#         stream=True
#     )
#
#     partial_message = ""
#     for chunk in response:
#         if len(chunk['choices'][0]['delta']) != 0:
#             partial_message = partial_message + chunk['choices'][0]['delta']['content']
#             yield partial_message
#
# def bot(history):
#     response = "**That's cool!**"
#     history[-1][1] = ""
#     for character in response:
#         history[-1][1] += character
#         time.sleep(0.05)
#         yield history
#
#
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(
#         [],
#         likeable=True,
#         elem_id="chatbot",
#         bubble_full_width=False,
#         avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
#         show_share_button=True,
#         show_copy_button=True
#         # submit_button_text="Get Advice",
#         # input_textbox_label="Your Health Query",
#         # show_typing_status=True
#         # enable_voice=True,
#         # theme="light",  # Change to your preferred color theme
#         # header="Health Assistant Chat",
#         # footer="Powered by OpenAI",
#     )
#
#     with gr.Row():
#         txt = gr.Textbox(
#             scale=4,
#             show_label=False,
#             placeholder="Enter text and press enter, or upload an image",
#             container=False,
#         )
#         btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
#
#     txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
#         bot, chatbot, chatbot, api_name="bot_response"
#     )
#     txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
#     file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
#         bot, chatbot, chatbot
#     )
#
# demo.queue()
# demo.launch()



#  Ex 1








# import gradio as gr
# import os
# import time
#
# def add_text(history, text):
#     history = history + [(text, None)]
#     return history, gr.Textbox(value="", interactive=False)
#
#
# def add_file(history, file):
#     history = history + [((file.name,), None)]
#     return history
#
#
# def bot(history):
#
#     response = "**That's cool!**"
#     history[-1][1] = ""
#     for character in response:
#         history[-1][1] += character
#         time.sleep(0.05)
#         yield history
#
#
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(
#         [],
#         elem_id="chatbot",
#         bubble_full_width=False,
#         avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
#     )
#
#     with gr.Row():
#         txt = gr.Textbox(
#             scale=4,
#             show_label=False,
#             placeholder="Enter text and press enter, or upload an image",
#             container=False,
#         )
#         btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
#
#     txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
#         bot, chatbot, chatbot, api_name="bot_response"
#     )
#     txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
#     file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
#         bot, chatbot, chatbot
#     )
#
# demo.queue()
# demo.launch()
