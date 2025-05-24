import gradio as gr
from .api import generate

context: list[str] = []


def chat(
    input: str, chat_history: list, top_k: float, top_p: float, temp: float
) -> tuple[list, list]:
    """
    Chat function that generates a response from the model.
    It takes the input message, the chat history, and the parameters
    and returns the chat history and the updated context.

    Args:
        input (str): The input message from the user
        chat_history (list): The chat history for UI display
        top_k (float): The top_k parameter
        top_p (float): The top_p parameter
        temp (float): The temperature parameter

    Returns:
        tuple: The chat history and the updated context
    """
    global context
    chat_history = chat_history or []
    output, context = generate(input, context, top_k, top_p, temp)
    chat_history.append((input, output))
    return chat_history, chat_history


def build_ui():
    """
    Build the UI

    Returns:
        gr.Blocks: The UI
    """
    with gr.Blocks() as block:
        gr.Markdown("<h1><center> Chatbot </center></h1>")
        chatbot = gr.Chatbot()
        message = gr.Textbox(placeholder="Type here")
        state = gr.State()
        with gr.Row():
            top_k = gr.Slider(0.0, 100.0, label="top_k", value=40)
            top_p = gr.Slider(0.0, 1.0, label="top_p", value=0.9)
            temp = gr.Slider(0.0, 2.0, label="temperature", value=0.8)
        submit = gr.Button("SEND")
        submit.click(
            chat, inputs=[message, state, top_k, top_p, temp], outputs=[chatbot, state]
        )
    return block
