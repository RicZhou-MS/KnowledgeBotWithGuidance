import gradio as gr
from gradio.themes.utils import colors, sizes
import os
from dotenv import load_dotenv
import guidance
from Docs_Search import DocVectorSearch
from GRSessionData import *
from GuidanceHelper import *


vSearch = DocVectorSearch()
vGuidanceHelper = GuidanceHelper()

def chat_set_msg(grSessionData,user_message, history):
    grSessionData.user_question = user_message
    grSessionData.add_chat_history_openai_user_msg(user_message)
    grSessionData.add_chat_history_gui_user_msg(user_message)
    # grSessionData.remove_br_from_chat_history_gui()
    return grSessionData, grSessionData.chat_history_gui


def chat_set_bot(grSessionData):
    standalone_question = vGuidanceHelper.rephrase_program(grSessionData.chat_history_openai)
    print(f"[Standalone question]: {standalone_question}")
    rzdocs = vSearch.getDocs(standalone_question)
    print(f"[Vector Search doc count]: {len(rzdocs)} [Content size]: {sum([len(doc.page_content) for doc in rzdocs])}")
    # print("***********************************************************************************")
    # print(rzdocs)
    # print("***********************************************************************************")
    chatAnswer = vGuidanceHelper.chat_program(chat_history=grSessionData.chat_history_openai,doc_list=rzdocs)
    grSessionData.add_chat_history_openai_bot_msg(chatAnswer)
    grSessionData.add_chat_history_gui_bot_msg(chatAnswer)
    print("=====================================================================================")
    print(chatAnswer)
    print("=====================================================================================")
    grSessionData.shrink_chat_history_openai()
    return grSessionData,"", grSessionData.chat_history_gui

def clearHistory(grSessionData):
    grSessionData.clear_all_chat_history()
    return grSessionData, grSessionData.chat_history_gui


# Spin up web GUI
# with gr.Blocks(theme=gr.themes.Glass()) as demo:
theme = gr.themes.Default(text_size=sizes.text_lg,primary_hue=colors.blue,secondary_hue=colors.orange)
swtichdarkscript = """
async () => {
  gradioURL = window.location.href
  if (!gradioURL.endsWith('?__theme=dark')) {
    window.location.replace(gradioURL + '?__theme=dark');
  }
  else{
    window.location.replace(gradioURL.replace("?__theme=dark", ""));
  }
}
"""
with gr.Blocks(theme=theme) as demo:
    grSessionData = gr.State(GRSessionData())
    # chat bot section
    title = gr.Button("Azure OpenAI KB bot via Guidance") #, label="", color="CornflowerBlue")
    chatbot = gr.Chatbot().style(height=600)
    msg = gr.Textbox(label="Type your question at below")
    with gr.Row():
        clear = gr.Button("Clear")

    gr.HTML('<hr size="18" width="100%" color="red">')
    
    # GUI event handlers
    title.click(None,None,None,_js=swtichdarkscript)
    msg.submit(chat_set_msg, [grSessionData, msg, chatbot], [grSessionData,chatbot], queue=False).then(chat_set_bot, grSessionData, [grSessionData, msg, chatbot])
    clear.click(clearHistory, grSessionData, [grSessionData,chatbot], queue=False)

    
#gr.State()
#demo.launch()
# demo.launch(auth=("admin", "pass1234"), share=True)
demo.launch(server_name="0.0.0.0", server_port=80)
# dark theme:  http://localhost/?__theme=dark