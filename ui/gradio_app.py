import gradio as gr
import uuid
from typing import Callable

def create_gradio_interface(
    answer_fn: Callable,
    clear_fn: Callable,
    export_fn: Callable,
    theme: str = "soft"
) -> gr.Blocks:
    
    with gr.Blocks() as demo:
        
        gr.Markdown("""
        #  Medical RAG Assistant

        Ask medical questions about drug interactions, side effects, dosages, or chat casually. 
        The system uses advanced retrieval-augmented generation (RAG) with FAISS indices and language models.

        **Features:**
        -  Drug interaction analysis
        -  Food-drug interactions
        -  Medical knowledge retrieval
        -  Context-aware chat history
        """)
        
        # Hidden session ID
        session_state = gr.State(value=str(uuid.uuid4()))
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about drug interactions, side effects, or chat casually...",
                        scale=4,
                        lines=2
                    )
                    submit_btn = gr.Button("Send 📨", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button(" Clear Chat")
                    export_btn = gr.Button(" Export Chat")
                    export_status = gr.Textbox(
                        label="Export Status",
                        visible=False,
                        interactive=False
                    )
            
            with gr.Column(scale=1):
                gr.Markdown("### Quick Examples")
                gr.Markdown("*Click on any example to try it:*")

                examples = gr.Examples(
                    examples=[
                        ["Hi"],
                        ["Can I take ibuprofen with paracetamol?"],
                        ["Food interactions with warfarin"],
                        ["Interaction between metformin and alcohol"],
                        ["What is the interaction between lepirudin and antipyrine?"],
                        ["Which foods or supplements may increase bleeding risk with urokinase?"],
                        ["How does alcohol affect blood sugar control with insulin human?"]
                    ],
                    inputs=[msg],
                    label=None
                )
                
                gr.Markdown("""
                ### ℹ Tips

                - Be specific in your questions
                - Mention drug names clearly
                - Ask about food interactions explicitly
                - Use the chat history for follow-up questions

                ###  Disclaimer

                This is an AI assistant for informational purposes only. 
                Always consult healthcare professionals for medical advice.
                """)
        
        # Event handlers
        def respond(message, session_id, history):
            if not message.strip():
                return "", history
            
            answer, _ = answer_fn(message, session_id, history)
            
            # Append in messages format for Gradio 6.x
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            
            return "", history

        def clear_and_reset(session_id):
            clear_fn(session_id)
            new_session = str(uuid.uuid4())
            return [], new_session, ""

        def export_and_notify(session_id):
            filepath = export_fn(session_id)
            return gr.update(value=f"Chat exported to: {filepath}", visible=True)
        
        # Connect events
        msg.submit(respond, [msg, session_state, chatbot], [msg, chatbot])
        submit_btn.click(respond, [msg, session_state, chatbot], [msg, chatbot])
        clear_btn.click(clear_and_reset, [session_state], [chatbot, session_state, export_status])
        export_btn.click(export_and_notify, [session_state], [export_status])
    
    return demo
