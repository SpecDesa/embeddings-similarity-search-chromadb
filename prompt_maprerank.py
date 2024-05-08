from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA 
# import langchain
# Debugs all, can be hard to read.
# langchain.debug =True
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
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        # Will select highest rank of k returned docs from chroma
        # Can make up a fact when reranking "facts" and give it 100, but 
        # can also do it accurate or give it 0 if it rewrite the fact. 
        ## If llm makes up high ranked answer, it can outrank the other true
        # facts
        chain_type="map_rerank"
        )


result = chain.run("What is an interesting fact about the english language?")


print(result)
