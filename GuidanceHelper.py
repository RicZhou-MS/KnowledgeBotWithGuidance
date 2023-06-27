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

        Please understand the converation carefully and rephrase the last STUDENT question into a single standalone question accordingly. please provide the standalone question ONLY, do NOT add any extra description.
        And please make sure you provide the standalone question in {{document_language}} language no matter the language that the last STUDENT question present.
        {{~/user}}

        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=200}}
        {{~/assistant}}
        """
                        ,llm=gpt35)

        self.chatprogram = guidance("""{{#system~}}
        You are a smart assistant and professional at answering user question, you answer ONLY with the facts listed in the list of sources below. You will not make up answer and will ONLY reply "no answer found" in case no context provided or the context is disconnected with the question.If asking a clarifying question to the user would help, ask the question. For tabular information return it as an html table. Do not return markdown format.Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [{'source': '01.pdf', 'page': 10}]. Don't combine sources, list each source separately, e.g. [{'source': 'A01.pdf', 'page': 10}][{'source': 'B02.pdf', 'page': 15}].

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