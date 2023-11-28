import os
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock

from langchain.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)

# Now we can override it and set it to "AI Assistant"
from langchain.prompts.prompt import PromptTemplate

template = """You are an AI medical assistant, whose main job is to understand the problem that a user have, 
provide a diagnosis (or ask follow up questions if needed to further figure out the roots of the problem). 
You also have to suggest some medicine that might help the human to relieve pain / or get well.
So you are a medical health assistant that should provide clarification and medical information to the users, about possible 
"diagnosis" , asses the severity/seriousness of the problem, educate them on subject matter, be friendly, try to also incorporate 
some financial info (e.g. with possibility fo having broken bone) one might need to do X-ray, then condulstion with thraumatolgist, 
then buy some medicine, etc.)
So you are a medical problem solver, and humanity needs your knowledge about medications and treatments. 
Side note: also be very friendly to humans.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)



def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
    )
    return conversation


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
                         'Try using a humidifier or taking steamy showers to ease the discomfort in your chest. '
            elif "any medication" in inp and 'chest' in inp and 'help' in inp:
                output = "Based on the information provided I advise to take Strepsils throat lozenges to ease the issues " \
                         "with breathing. Also, use dextromethorphan to loosen up the pain in the chest."
            elif "get them?" in inp and "wheree" in inp:
                import time
                time.sleep(0.75)
                output = """Yes! Based on your location there are several pharmacies nearby: 
                Pharmacy(Marshal Baghramyan Street, 21)
                Pharmacy(Ler Kamsar Street, 30/3)
                Family Pharm(Marshal Baghramyan Street, 2)"""
            elif "approximately" in inp and  "cost" in inp and "medications" in inp:
                output = "Based on the data I posses, Strepsils pills would cost from 1500-2500 AMD. " \
                         "Dextromethorphan syrup will be available for around 4000-5000 AMD for a bottle size of 100ml."

            elif "medications do not work" in inp:
                output = "If your health doesn't improve, I would recommend to approach a general practitioner or" \
                          " otolaryngologist. You can find the list of best professionals on our website."

            elif "Thanks" in inp or "tips" in inp:
                output = 'Take care of yourself and keep an eye on any changes. ' \
                         'If it does not improve, do not hesitate to consult with a healthcare professional. ' \
                         'Wishing you a speedy recovery!'


            elif "hiking" in inp and "ankle" in inp:
                output = "I'm sorry to hear about your fall. Is the pain constant, " \
                         "or does it worsen when you try to move your ankle?"
            elif "definitely hurts" in inp and "weight" in inp:
                output = "It's possible you might have sprained your ankle or sustained an injury. " \
                         "Have your tried using Voltaren Gel for the injury?"
            elif "pharmacy" in inp and "not yet" in inp and "that gel" in inp:
                import time
                time.sleep(1)
                output = """Yes! I have found some open pharmacies.
                Nvard Pharm
                Pharmacy
                230.0 m · 21 Marshal Baghramyan Ave
                Open ⋅ Closes 10 PM
                In-store shopping
                
                Pharmacy
                600.0 m · 22/6 Gulakyan St
                Open 24 hours
                In-store shopping
            
                Gedeon Richter Pharmacy
                Pharmacy
                550.0 m · 51 Marshal Baghramyan Ave
                Open ⋅ Closes 12 AM
                In-store shopping
                """
            elif "afraid" in inp:
                output = 'However, considering the persistent pain and swelling, it might be prudent to schedule' \
                         ' an appointment with a traumatologist or orthopedic specialist from Vardanants Medical Center' \
                         ' to get a thorough evaluation and ensure proper care for your ankle.'
            elif "better" in inp and "not improving" in inp:
                output = "It's essential to have it checked to rule out any fractures or severe sprains. In the meantime, " \
                         "try to avoid putting weight on it, and if possible, use crutches or support to ease the pressure " \
                         "on your ankle. Take care!"

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
            placeholder="Paste your API key",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=(os.path.join(os.path.dirname(__file__), "user.png"),
                           os.path.join(os.path.dirname(__file__), "avatar.png")),
        )
    with gr.Row():
        message = gr.Textbox(
                label='Human',
                show_label=False,
                interactive=True,
                scale=4,
                placeholder="Enter text and press enter, or upload an image",
                container=False,
            )
        # submit = gr.Button(value="Send", variant="secondary", css_class_name="custom-button")
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
        submit = gr.Button(value="Send", variant="secondary", size='sm')#.style(full_width=False)

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

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state], queue=False)
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state], queue=False)

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
