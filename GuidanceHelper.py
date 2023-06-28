import guidance
import os
from dotenv import load_dotenv

class GuidanceHelper:
    rephraseprogram = None
    chatprogram = None
    document_language = "English"

    def __init__(self):
        load_dotenv()
        gpt35 = guidance.llms.OpenAI(
            'gpt-3.5-turbo',
            api_type='azure',
            api_key=os.getenv('OPENAI_API_KEY'),
            api_base=os.getenv('OPENAI_API_BASE'),
            api_version='2023-05-15', #'2022-12-01',
            deployment_id=os.getenv('GPT35_DEPLOYMENT_ID') #'chat'
        )

        gpt4 = guidance.llms.OpenAI(
            'gpt-4',
            api_type='azure',
            api_key=os.getenv('OPENAI_API_KEY'),
            api_base=os.getenv('OPENAI_API_BASE'),
            api_version='2023-05-15',
            deployment_id=os.getenv('GPT4_DEPLOYMENT_ID') #'gpt-4'
        )
        
        '''

        
        self.rephraseprogram = guidance("""{{#system~}}
        You are a helpful assistant. User will give you a conversation between EMPLOYEE and HR. You will understand the whole conversation carefully and inference the EMPLOYEE latest intention, then rephrase the intention into proper description which is suitable for embedding search engine to locate most relavant information chunks afterwards.
        Please provide the rephrased information ONLY, do NOT add any extra description or explanation. Please MUST provide your response in {{document_language}} language no matter the language the conversation presents.
        {{~/system}}

        {{#user~}}
        EMPLOYEE: 你好
        {{~/user}}
        {{#assistant~}}
        Greeting
        {{~/assistant}}

        {{#user~}}
        EMPLOYEE: 发票丢失了
        {{~/user}}
        {{#assistant~}}
        Process of handling invoice lost
        {{~/assistant}}

        {{#user~}}
        EMPLOYEE: Good morning!
        HR: Good morning, how can I help you?
        EMPLOYEE: I don't know when the holiday in May is.
        {{~/user}}
        {{#assistant~}}
        The date of holiday in May
        {{~/assistant}}


        {{#user~}}
        EMPLOYEE: what are the traning classes tomorrow
        HR: 1. Math
        2. English
        3. P.E.
        EMPLOYEE: who is the trainer of the first class?
        {{~/user}}
        {{#assistant~}}
        The trainer name of Math class
        {{~/assistant}}

        {{#user~}}
        {{#each chat_history}}
        {{#if this.role == 'user'}}
        EMPLOYEE: {{this.content}}
        {{else}}
        HR: {{this.content}}
        {{/if}}
        {{/each}}
        {{~/user}}

        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=200}}
        {{~/assistant}}
        """
                        ,llm=gpt35)
    
                        
        self.chatprogram = guidance("""{{#system~}}
        You are a smart assistant and professional at answering user question, you answer ONLY with the facts listed in the list of sources below. You will NOT make up answer and you will ONLY reply "no answer found" in case no context provided or the context is disconnected with the question. If asking a clarifying question to the user would help, ask the question. For tabular information return it as an html table. Do not return markdown format.Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [{'source': '01.pdf', 'page': 10}]. Don't combine sources, list each source separately, e.g. [{'source': 'A01.pdf', 'page': 10}][{'source': 'B02.pdf', 'page': 15}].

        Sources:
        {{#each doc_list}}
        {{this.metadata}}: {{this.page_content}}
        {{/each}}
        {{~/system}}

        {{#each chat_history}}
        {{#if this.role == 'user'}}
        {{#user~}}
        {{this.content}}
        {{~/user}}
        {{else}}
        {{#assistant~}}
        {{this.content}}
        {{~/assistant}}
        {{/if}}
        {{/each}}

        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=1000}}
        {{~/assistant}}
        """,llm=gpt4)

        '''

        self.rephraseprogram = guidance("""
        {{#system~}}
        {{llm.default_system_prompt}}
        {{~/system}}

        {{#user~}}
        I got a conversation as following:
        {{#each chat_history}}
        {{#if this.role == 'user'}}
        STUDENT: {{this.content}}
        {{else}}
        TEACHER: {{this.content}}
        {{/if}}
        {{/each}}

        Please understand the conversation carefully and rephrase the last STUDENT question into a single standalone question accordingly. please provide the standalone question ONLY, do NOT add any extra description.
        If what STUDENT said at last is not a question, return its original meaning. Please make sure your response in {{document_language}} language no matter the language that the conversation presents.
        {{~/user}}

        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=200}}
        {{~/assistant}}
        """
                        ,llm=gpt35)

        self.chatprogram = guidance("""{{#system~}}
        You are a smart assistant and professional at answering user question, you answer ONLY with the facts listed in the list of sources below. You will NOT make up answer and should only reply "no answer found" in case no context provided or the context is not relevant to the question. If asking a clarifying question to the user would help, ask the question. For tabular information return it as an html table. Do not return markdown format. Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, e.g. [{'source': '01.pdf', 'page': 10}]. Don't combine sources, list each source separately, e.g. [{'source': 'A01.pdf', 'page': 10}][{'source': 'B02.pdf', 'page': 15}].

        Sources:
        {{#each doc_list}}
        {{this.metadata}}: {{this.page_content}}
        {{/each}}
        {{~/system}}

        {{#each chat_history}}
        {{#if this.role == 'user'}}
        {{#user~}}
        {{this.content}}
        {{~/user}}
        {{else}}
        {{#assistant~}}
        {{this.content}}
        {{~/assistant}}
        {{/if}}
        {{/each}}

        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=1000}}
        {{~/assistant}}
        """,llm=gpt4)

    def rephrase_program(self, chat_history):
        rephraseResult = self.rephraseprogram(chat_history=chat_history, document_language=self.document_language)
        return rephraseResult["answer"]
    
    def chat_program(self, chat_history, doc_list):
        chatResult = self.chatprogram(chat_history=chat_history, doc_list=doc_list)
        return chatResult["answer"]
    
    def chat_program_streaming(self, chat_history, doc_list):
        chatResult = self.chatprogram(chat_history=chat_history, doc_list=doc_list, stream=True, silent=True)
        return chatResult