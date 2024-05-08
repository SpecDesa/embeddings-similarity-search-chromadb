from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA 

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
        # Find relevant k docs, 
        # takes first most relevant doc and feeds to llm 
        ## Then takes result and feeds that to next step llm/chain, and context 
        ## of next relevant k doc. 
        ### and so on... until last k doc has been visited and chain completed. 
        # last step can really mess up chains as it can return "nothing found"
        # from given context, instead of good step from earlier step.
        chain_type="refine"
        )


result = chain.run("What is an interesting fact about the english language?")


print(result)
