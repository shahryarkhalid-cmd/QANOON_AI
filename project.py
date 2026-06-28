# ====== IMPORTING THE FILES ====================
from langchain_community.document_loaders import WebBaseLoader , PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import START , END , StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel , Field
from typing import TypedDict
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import List , Dict , Literal , Optional , Annotated
from langchain_core.messages import HumanMessage , SystemMessage , AIMessage , BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from sentence_transformers import CrossEncoder
from langgraph.prebuilt import tools_condition , ToolNode
load_dotenv()

class Planner(BaseModel):
  decision : str = Field(description='The word dicision')
  key : Literal['chat' , 'rag'] = Field(description = "The decision made by llm")
class Planner_validator(BaseModel):
  planner_decision : Planner = Field(description = "The Decision of the planner whether chat or rag")
class Raw_data(BaseModel):
  page_content : str = Field("The page content of the data")
  metadata : dict = Field("The metadata of the data")
  id : str = Field(description='unique ID OF the document')

class DocumentScore(BaseModel):
    id: str = Field(description='unique ID OF the document')
    rating: float = Field(description='relevance score 0-1')
class Generator(BaseModel):
  answer : str
  source : Literal['retrieved' , 'web_search' , 'both']
  confidence : Literal['high' , 'medium' , 'low']
class Stripping(BaseModel):
  sentence : str
  score : float
# DIFINING A STATE:
class State(TypedDict):
  query : str
  messages : Annotated[list[BaseMessage] , add_messages]
  planner_decision : dict
  raw_docs : list[Raw_data]
  document_scores : list[DocumentScore]
  router : Literal['Approve' , 'Incorrect' , 'Ambiguous']
  allowed_docs : list[Raw_data]
  ambiguous_docs : Optional[list[Raw_data]]
  incorrect_docs : Optional[list[Raw_data]]
  strip_doc : list[Stripping]
  ambiguous_incorrect : Optional[list[str]] 
  after_stripping_doc : list[str]
  web_search_result : str
  generator : Generator
  rewritten_query:str
  chat_web_search_result : str
  chat_rewritten_query : str
  chat_generator : str


import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')



# Defining a RAG pipeline:
def RAG(state : State):
    try:
      vector_store = FAISS.load_local('Law' , embeddings , allow_dangerous_deserialization = True)
      print(f'The documents loaded')
    except Exception as e:
      print(f'We cannot load because \n {e}')
      url =[
          '\Documents\Pakistan_criminal_code.pdf',
          '\Documents\Pakitan_panel_code.pdf',
          '\Documents\Transfer_Property_Act.pdf',
          '\Documents\constitution_of_1973.pdf',
      ]
      pdf_loader = []
      for i in range (len(url)):
        pdf = PyMuPDFLoader(url[i])
        print(f'Loading document number : {i}\n')
        pdf = pdf.load()
        pdf_loader.extend(pdf)
      print(f'The documents loaded are : \n : {len(pdf_loader)}')

      splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap  = 100
      )
      print('documents are splitted................\n')
      chunks = splitter.split_documents(pdf_loader)
      print(f'Splitted number of chunks : {len(chunks)}\n')
      vector_store = FAISS.from_documents(chunks , embeddings)
      vector_store.save_local('Law')
      print('Document is saved.............\n')
    retriever = vector_store.as_retriever(search_kwargs = {'k' : 3})
    result = retriever.invoke(state['query'])
    for i in range(len(result)):
      result[i] = Raw_data(page_content = result[i].page_content , metadata = result[i].metadata , id = result[i].id)
    print('RAG IS SEETED \n')
    return {'raw_docs' : result}



import json
# Explicit OUTPUT PARSING
class evaluator_validator(BaseModel):
    documents: List[DocumentScore] = Field(description='list of document scores')
def explicit_output_parser(doc : str , model = type(BaseModel)):
  start = doc.find('{')
  end = doc.rfind('}') + 1
  if start !=-1 and end != 0:
    json_str = doc[start:end]
  json_str = re.sub(r'[\x00-\x1f\x7f]', ' ', json_str)
  json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
  json_str = re.sub(r'\s+', ' ', json_str)
  return model.model_validate_json(json_str)
def modified_explicit_output_parser(doc : str , model = type(BaseModel)):
  start = doc.find('{')
  end = doc.rfind('}') + 1
  if start !=-1 and end != 0:
    json_str = doc[start:end]
  json_str = re.sub(r'[\x00-\x1f\x7f]', ' ', json_str)
  json_dict = json.loads(json_str)
  return model.model_validate(json_dict)


# DEFINING THE PLANNER :

def planner(state:State):
  prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a smart planner for a Pakistan Law Assistant.

Your ONLY job is to read the user query and decide where to route it.

Route to "rag" when user asks about:
1. LEGAL QUESTIONS
   - Laws, acts, ordinances, sections, articles
   - "what is the punishment for..."
   - "what are my rights..."
   - "is it legal to..."

2. PAKISTAN SPECIFIC INFORMATION
   - Constitution, Penal Code, Criminal Procedure
   - Family Law, Property Law, Labor Law
   - FIR, bail, arrest, divorce, inheritance, property

3. FACTUAL QUESTIONS ABOUT PAKISTAN
   - Government policies, legal procedures
   - Rights, punishments, court procedures

Route to "chat" when user:
1. GREETS OR DOES SMALL TALK
   - "hello", "hi", "how are you", "thank you"

2. ASKS GENERAL QUESTIONS NOT RELATED TO PAKISTAN LAW
   - "what is machine learning", "tell me a joke"
   - "latest news", "current situation of..."

3. CONVERSATIONAL FOLLOW UPS
   - "can you explain that again", "what do you mean"

EXAMPLES:
Query: "what is the punishment for theft?"        → {{"decision": "rag"}}
Query: "can police arrest me without warrant?"    → {{"decision": "rag"}}
Query: "what are fundamental rights in Pakistan?" → {{"decision": "rag"}}
Query: "talaq ka tariqa kya hai?"                 → {{"decision": "rag"}}
Query: "hello how are you?"                       → {{"decision": "chat"}}
Query: "what is artificial intelligence?"         → {{"decision": "chat"}}
Query: "latest news about something"              → {{"decision": "chat"}}
Query: "thank you for your help"                  → {{"decision": "chat"}}

CRITICAL RULES:
- Reply in JSON format ONLY
- Output MUST be exactly like this: {{"decision": "chat"}} or {{"decision": "rag"}}
- No explanation
- No extra text
- Nothing else outside JSON
"""),
    ('human', 'Query: {query}\n\nDecision:')
])
  model = ChatGroq(model = 'meta-llama/llama-4-scout-17b-16e-instruct')
  chain = prompt | model | StrOutputParser()
  result = chain.invoke({'query' : state['query']})
  if not result or result.strip() == '':
      print('⚠️ Empty result from LLM defaulting to rag')
      return {'planner_decision': {'decision': 'rag'}}
  json_dict = json.loads(result)
  print(f'Planner : {json_dict}\n')
  return {'planner_decision' : json_dict}



# Defining a Evaluator

# Document Evaluator:
def document_evaluator(state:State):
  llm = ChatGroq(model = 'llama-3.1-8b-instant')
  prompt = ChatPromptTemplate([('system' , """You are an evaluator.
         You are given documents and a query.
         Rate each document from 0-1 based on how relevant it is to the query.
         Higher score means more relevant.
         Reply ONLY in this JSON format, nothing else:
         {{
             "documents": [
                 {{"id": "id_0", "rating": 0.8}},
                 {{"id": "id_1", "rating": 0.3}},
                 {{"id": "id_2", "rating": 0.6}}
             ]
         }}"""
                                     ) , ('human' , '{document} , {query}')
  ]
                              )

  chain = prompt | llm | StrOutputParser()
  result = chain.invoke({'document' : state['raw_docs'] , 'query' : state['query']})
  scores = []
  final_result = explicit_output_parser(result , evaluator_validator)
  print('Documents are scored \n')
  return {'document_scores' : final_result.documents}

# DOCUMENT FILTERATOR:
def document_filterator(state : State):
  allowed_doc = []
  ambiguous_doc = []
  incorrect_doc = []
  for i in range(len(state['document_scores'])):
    if state['document_scores'][i].rating > 0.5:
      allowed_doc.append(state['raw_docs'][i])
    elif state['document_scores'][i].rating < 0.3:
      incorrect_doc.append(state['raw_docs'][i])
    else:
      ambiguous_doc.append(state['raw_docs'][i])
  print("documents are Filtered \n")
  return {'allowed_docs' : allowed_doc , 'ambiguous_docs' : ambiguous_doc , 'incorrect_docs':incorrect_doc}


def sentence_stripping(state:State):
    stripping_doc = state['ambiguous_docs'] + state['allowed_docs']
    all_sentences = []
    min_length = 20
    for i in range(len(stripping_doc)):
       sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', stripping_doc[i].page_content) if s.strip()]
       sentences = [s for s in sentences if len(s) >= min_length]
       all_sentences.extend(sentences)
    model = cross_encoder
    pairs = [[state['query'] , sentence] for sentence in all_sentences]
    print(f'Total pairs created: {len(pairs)}\n')
    scores = model.predict(pairs)
    sentenced_scores = list(zip(all_sentences , scores))
    scored_sentences = sorted(sentenced_scores, key=lambda x: x[1], reverse=True)
    strip_doc = [Stripping(sentence=sentence, score=float(score)) for sentence, score in scored_sentences]
    print('Documents are stripped')
    return {'strip_doc' : strip_doc}
def sentence_stripping_2(state:State):
  after_stripping_doc = state.get('after_stripping_doc', [])
  ambiguous_incorrect = state.get('ambiguous_incorrect', [])

  for i in range(len(state['strip_doc'])):
    if state['strip_doc'][i].score > -5:
      after_stripping_doc.append(state['strip_doc'][i].sentence)
    else:
      ambiguous_incorrect.append(state['strip_doc'][i].sentence)
  print('docuemnts sencentences are stripped \n')
  return {'after_stripping_doc' : after_stripping_doc , 'ambiguous_incorrect' : ambiguous_incorrect}

# QUERY REWRITER:
def query_rewriter(state:State):
  prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a legal query rewriting expert for Pakistan Law.

   Your job is to rewrite a user's casual or vague question into a precise legal query
   that will retrieve the most relevant information from a legal database.

   Follow these rules when rewriting:

   1. ADD legal terminology
   2. ADD relevant law name if you know it
   3. ADD specific section or article if possible
   4. EXPAND vague queries with related legal terms
   5. KEEP the original meaning never change intent
   6. For Urdu or mixed queries convert to English legal terms


  Reply ONLY with the rewritten query as plain text.
  No explanation, no extra text, just the rewritten query.
"""),
    ('human', 'Original Query: {query}\n\nRewritten Query:')
])
  model = ChatGroq(model = "groq/compound")
  chain = prompt | model | StrOutputParser()
  result = chain.invoke(state['query'])
  print('Query is rewritten \n')
  return {'rewritten_query' : result}


# WEBSEARCHING
tool = TavilySearchResults(max_results=5)
def Websearching(state:State):
  ambiguous_incorrect_str = state.get('ambiguous_incorrect', [])
  incorrect_docs_content = [doc.page_content for doc in state.get('incorrect_docs', [])]
  context_parts = ambiguous_incorrect_str + incorrect_docs_content
  context_for_llm = '\n\n'.join(context_parts)

  prompt = ChatPromptTemplate(
      [
          ('system' , """You are a helpful AI assistant your job is to use the tool and search for the query answer You are given two things first one is the query
          itself and the other context as a helping agent for helping to find the best match"""),('human' , '{query} , {context}')
      ]
  )
  llm = ChatGroq(model = 'meta-llama/llama-4-scout-17b-16e-instruct')
  llm_with_tool = llm.bind_tools([tool] , tool_choice = 'required')
  chain = prompt | llm_with_tool
  result = chain.invoke({'query' : state['query'] , 'context' : context_for_llm})
  tool_call = result.tool_calls[0]
  search_result = tool.invoke(tool_call['args'])
  search_text = '\n\n'.join([r['content'] for r in search_result])
  print('Web search is done \n')
  return {'web_search_result': search_text}
# Generator:
def generator(state:State):
  prompt = ChatPromptTemplate.from_messages([
    ('system', """You are an expert AI assistant with access to both retrieved document context and live web search results.

    Your job is to generate a precise, accurate, and well-structured answer based on the provided context.

    Follow these strict rules:
    1. Answer ONLY from the provided context, do not use your own knowledge
    2. If retrieved context and web search context contradict each other, prefer the web search result as it is more recent
    3. If the answer is not found in either context, respond with "I do not have enough information to answer this question"
    4. Always structure your answer clearly with proper explanation
    5. Do not hallucinate or make up any facts
    6. Cite whether the answer came from retrieved documents or web search

    Context Structure you will receive:
    - Retrieved Context: information from the document knowledge base
    - Web Search Context: information from live web search results

    Reply in the following JSON format:
    {{
       "answer": "your detailed answer here",
       "source": "retrieved / web_search / both",
       "confidence": "high / medium / low"exp
}} and donot use /n as telling second line keep in mind donot use backsplashes
"""),
    ('human', """Retrieved Context: {retrieved_context}

Web Search Context: {web_search_context}

User Query: {query}

Answer:""")
])
  llm = ChatGroq(model = 'meta-llama/llama-4-scout-17b-16e-instruct')
  explicit_output_parser
  chain = prompt | llm
  result = chain.invoke({'retrieved_context' : '\n\n'.join(state['after_stripping_doc']) , 'web_search_context' : state['web_search_result'] , 'query' : state['query']})
  final_result = explicit_output_parser(result.content , Generator)
  print('Final output is generated \n')
  return {'generator' : final_result , 'messages'  : [AIMessage(content = final_result)]}
# Router
def planner_router(state:State):
  if state['planner_decision']['decision'] == 'chat':
    return 'CHAT_BOT'
  return 'RAG'
def evaluator_router(state: State):
    approve_count = 0
    ambiguous_count = 0
    incorrect_count = 0
    for i in range(len(state['document_scores'])):
        rating = state['document_scores'][i].rating
        if rating > 0.7:
            approve_count += 1
        elif rating < 0.3:
            incorrect_count += 1
        else:
            ambiguous_count += 1
    print(f'Approved: {approve_count} | Ambiguous: {ambiguous_count} | Incorrect: {incorrect_count}')
    if approve_count > 0:
        return 'Approve'
    elif ambiguous_count > 0:
        return 'Ambiguous'
    else:
        return 'Incorrect'
def chat_router(state: State):
    last_message = state['messages'][-1].content.lower()

    # keywords that need web search
    search_keywords = [
        'news', 'latest', 'today', 'current', 'price',
        'rate', 'who is', 'what happened', 'recently',
        'update', 'weather', 'score', '2026', 'now'
    ]

    if any(word in last_message for word in search_keywords):
        print(' Routing to web search')
        return 'search'

    print(' Routing to end directly')
    return 'end'


# DEFINING A SIMPLE CHATTING :
def chat_node(state:State):
  prompt = ChatPromptTemplate.from_messages([
    ('system', """You are "Wakeel Sahab" — Pakistan's most charming AI legal assistant.

You are professional, knowledgeable, and humorous. You balance professionalism with light Desi humor.
Your tone is warm and friendly.

Rules for interaction:
1. Greet warmly with humor.
2. Be helpful and clear.
3. Add a witty remark or light joke where appropriate.
4. Keep professionalism; avoid sarcastic or hurtful humor.
5. Use Desi expressions like "yaar", "arre", "bilkul", "acha" occasionally.
6. Never make fun of Pakistan's legal system disrespectfully.
7. NEVER give legal advice. For legal questions, say: "Arre yaar that sounds like a serious legal matter! Let me look that up properly for you in my legal database!"
8. Keep responses concise and witty.
9. Always end with an invitation to ask more.
"""),
    ('human', '{query}')
])
  model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")
  chain = prompt | model | StrOutputParser()
  result = chain.invoke({'query' : state['query']})
  generator = result
  message = AIMessage(content = result)
  print(f'💬 Chat done: {result[:100]}\n')

  return {
        'messages': [
            HumanMessage(content=state['query']),
            AIMessage(content=result)
        ],
        'chat_generator': result                     }
# CHAIN_WEBSEARCHING
search_tool = TavilySearchResults(max_results=5 , name="tavily_search_results_json")
def Chat_Websearching(state:State):
  prompt = ChatPromptTemplate.from_messages([
        ('system', f"""You are a search assistant.
         You have access to ONLY ONE tool called: {search_tool.name}
         You MUST use ONLY this tool: {search_tool.name}
         Do not use any other tool name."""),
        ('human', '{query}')
    ])
  llm = ChatGroq(model = 'meta-llama/llama-4-scout-17b-16e-instruct')
  llm_with_tool = llm.bind_tools([tool] , tool_choice = 'required')
  chain = prompt | llm_with_tool
  result = chain.invoke({'query' : state['chat_rewritten_query']})
  tool_call = result.tool_calls[0]
  search_result = tool.invoke(tool_call['args'])
  search_text = '\n\n'.join([r['content'][:300] for r in search_result[:3]])
  print('CHAT_Web search is done \n')
  return {'chat_web_search_result': search_text}
def chat_query_rewriter(state:State):
      print("Entered in query rewriter\n")
      chat_rewriter_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a search query optimization expert.
     Rewrite user queries into effective web search queries.

     Rules:
     1. ADD specific keywords and year 2026
     2. ADD location when relevant
     3. ADD time context for events
     CRITICAL: Reply with rewritten query ONLY. No explanation. Under 15 words.
     """),
    ('human', 'Original Query: {query}\n\nRewritten Search Query:')
])
      print("Before rewriting query")
      llm = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct')
      chain = chat_rewriter_prompt | llm | StrOutputParser()
      rewritten = chain.invoke({'query': state['query']})
      print('chat query rewritten\n')
      return {'chat_rewritten_query': rewritten}


# defining a graph
tool_node = ToolNode([tool])
graph = StateGraph(State)
graph.add_node('PLANNER' , planner)
graph.add_node('CHAT_BOT' , chat_node)
graph.add_node('CHAT_SEARCHER' , Chat_Websearching)
graph.add_node('CHAT_QUERY_REWRITER' , chat_query_rewriter)
graph.add_node('RAG' , RAG)
graph.add_node('DOCUMENT_SCORER' , document_evaluator)
graph.add_node('DOCUMENT_FILTERATOR' , document_filterator)
graph.add_node('DOCUMENT_STRIPPER' ,  sentence_stripping)
graph.add_node('SENTENCE_STRIPPER' , sentence_stripping_2)
graph.add_node('QUERY_REWRITER' , query_rewriter)
graph.add_node('WEB_SEARCHER' , Websearching)
graph.add_node('GENERATOR' , generator)
graph.add_edge(START , 'PLANNER')
graph.add_conditional_edges('PLANNER' , planner_router , {'CHAT_BOT' : 'CHAT_BOT' , 'RAG':'RAG'})
graph.add_conditional_edges('CHAT_BOT' , chat_router , {'search' :'CHAT_QUERY_REWRITER' , 'end' :END})
graph.add_edge('CHAT_QUERY_REWRITER' , 'CHAT_SEARCHER')
graph.add_edge('CHAT_SEARCHER' ,END)
graph.add_edge('RAG' , 'DOCUMENT_SCORER')
graph.add_edge('DOCUMENT_SCORER' , 'DOCUMENT_FILTERATOR')
graph.add_conditional_edges('DOCUMENT_FILTERATOR' , evaluator_router , {'Approve' : 'DOCUMENT_STRIPPER' , 'Ambiguous' : 'QUERY_REWRITER' , 'Incorrect' : 'QUERY_REWRITER'})
graph.add_edge('DOCUMENT_STRIPPER' , 'SENTENCE_STRIPPER')
graph.add_edge('SENTENCE_STRIPPER' , 'GENERATOR')
graph.add_edge('QUERY_REWRITER' , 'WEB_SEARCHER')
graph.add_edge('WEB_SEARCHER' , 'GENERATOR')
graph.add_edge('GENERATOR' , END)
connection = sqlite3.connect('CRAG.db' , check_same_thread=False)
checkpointer = SqliteSaver(connection)
workflow = graph.compile(checkpointer = checkpointer)





config = {'configurable' : {'thread_id' : '3'}}
while True:
    user_input = input('user:')
    quitting_statements = ['bye' , 'quit' , 'exit']
    if user_input in quitting_statements:
        print("Thanks for chatting")
        break
    result = workflow.invoke({
    'query' : user_input,
    'raw_docs' : [],
    'document_scores' : [],
    'router' : '',
    'planner_decision' : {},
    'allowed_docs' : [],
    'ambiguous_docs' : [],
    'incorrect_docs' : [],
    'strip_doc' : [],
    'ambiguous_incorrect' : [],
    'after_stripping_doc' : [],
    'web_search_result' : '',
    'generator' : {},
    'rewritten_query' : ''
} , config = config)
    if result['planner_decision']['decision'] == 'chat':
        print(f'AI : {result['messages'][-1].content}')
    else:
        print(f'AI: {result['generator'].answer}' )


