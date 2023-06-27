class GRSessionData:
    chat_history_gui = []  # history + [[user_message, None]] # history[-1][1] = assistant_message
    chat_history_openai = [] # chat_history.append({'role': 'assistant', 'content': chatResult["answer"]}) # chat_history.append({'role': 'user', 'content': user_input})
    standalone_question = None
    user_question = None
    doc_search_result = None

    def __init__(self):
        self.chat_history_gui = []
        self.chat_history_openai = []
        self.standalone_question = ""
        self.user_question = ""

    def add_chat_history_gui_user_msg(self, msg):
        self.chat_history_gui.append([msg, None])
    
    def add_chat_history_gui_bot_msg(self, msg):
        self.chat_history_gui[-1][1] = msg

    def add_chat_history_openai_user_msg(self, msg):
        self.chat_history_openai.append({'role': 'user', 'content': msg})

    def add_chat_history_openai_bot_msg(self, msg):
        self.chat_history_openai.append({'role': 'assistant', 'content': msg})

    def shrink_chat_history_openai(self):
        self.chat_history_openai = self.chat_history_openai[-10:] # keep the history within 5 rounds

    def clear_all_chat_history(self):
        self.chat_history_gui = []
        self.chat_history_openai = []

    def remove_br_from_chat_history_gui(self):
        for x in range(0, len(self.chat_history_gui)):
            self.chat_history_gui[x][0] = self.chat_history_gui[x][0].replace("<br>", "")
            self.chat_history_gui[x][1] = self.chat_history_gui[x][1].replace("<br>", "")