from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    '''
    Write custom retriever that removes duplicate documents from db,
    where similarity score for embeddings are close to equal
    '''
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # calculate embeddings for query string,
        emb = self.embeddings.embed_query(query)
        # Take embeddings and feed into max_marginal_relevance_search_by_vector

        return self.chroma.max_marginal_relevance_search_by_vector(
                                                                   embedding=emb,
                                                                   lambda_mult=0.8)
    def _get_relevant_documents(self, query: str):
        return []

    # Not used for this example.
    def aget_relevant_documents(self, query):
        return []
