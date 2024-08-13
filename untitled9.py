

!pip install -qU langchain-openai
!pip install langchain-community langchain-core
!pip install -qU langchain-text-splitters
!pip install -qU langchain-chroma
!pip install -qU langchain-output-parsers
!pip install -qU langchain-prompts
!pip install -qU langchain-runnables

import getpass
import os
#setting api key for openai usage
os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_community.document_loaders.csv_loader import CSVLoader

file_path = "ContentAndCategories.csv"

loader = CSVLoader(file_path=file_path)
data = loader.load()

from langchain_text_splitters import CharacterTextSplitter


text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300, #length of longest category
    chunk_overlap=0, # no overlap as complete separation, structured data
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(data)
print(len(texts))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

topic = input()
level = input()
retrieved_docs = retriever.invoke("What are" + topic + "for" + level)

print(retrieved_docs)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Use the following pieces of context to answer the question at the end.
Use as many of the provided words as possible to make a sentence. Make sure the sentences are child-safe and appropriate.
Don't say anything that isn't a direct part of your answer. Write so many sentences that each word in the list given appears three times.
Do not write any more sentences than that
{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("A2, Calendar")
