from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA 
from redundant_filter_retriever import RedundantFilterRetriever

import langchain

langchain.debug =True

load_dotenv()

chat = ChatOpenAI();

# Create the embedding function
embeddings = OpenAIEmbeddings()

# have db that do not calculate embeddings each time, but used stored embs. 
# Without adding content to db. 
db = Chroma(
        persist_directory="emb",
        embedding_function=embeddings
        )

# func from chroma/langchain that has a retriever func, 
# that takes a string and return documents. e.g. get_relevant_documents(string)
# Actually just calls back to similarity_search(), so "clue code". 
# retriever = db.as_retriever()

## Use custom retriever instead 
retriever = RedundantFilterRetriever(
        embeddings=embeddings,
        chroma=db
        )


chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        # taking / injecting /"stuff"ing them into the prompt  
        # Go find some text context from vector store and "stuff" it into the 
        # prompt
        chain_type="stuff"
        )


result = chain.run("What is an interesting fact about the english language?")


print(result)
