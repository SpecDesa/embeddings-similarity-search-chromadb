from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi there")

# print(embed)


text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0
        )

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
        text_splitter=text_splitter
        )

# Watch out, it reach out to openai as it instantly reach out and calculate embeddings using 
# openapi key. 
# Will RUN EVERYTIME you run the program
db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="emb"
        ) 


results = db.similarity_search_with_score("What is an interesting fact about the english language?",
                                          k=3)


for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)
