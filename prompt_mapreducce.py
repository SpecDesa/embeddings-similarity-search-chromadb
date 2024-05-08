from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA 

# import langchain 
# Debugs all, can be hard to read.
# langchain.debug = True

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



# Map reduce takes longer
# Calls llm multiple times (five in total) 
# Chroma returns 4 (or k) relevant docs. 
## takes each of these and take them to individually system-human prompt and 
## feed to llm. there will be some standard "Use the following.... long text." 
### llm will return a single fact in this case for each k docs.  
### (PROBLEM) llm/chatgpt can make up fact if not in k doc. 
### All assembled in the end, so the k most relevant is put together to one
### doc, and send to llm. 
chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        # taking / injecting /"stuff"ing them into the prompt  
        # Go find some text context from vector store and "stuff" it into the 
        # prompt
        chain_type="map_reduce"
        )


result = chain.run("What is an interesting fact about the english language?")


print(result)
