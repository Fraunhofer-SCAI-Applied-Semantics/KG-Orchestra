from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import torch
# from kg_orchestra.modules.agents import Vectorizer
from kg_orchestra.modules.biomedical_models import BiomedicalParagraph
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SparseEncoder
from qdrant_client.http.models import (
    models,
)

class BaseFetcher(QdrantClient):
    
    def __init__(self, host:str, port:int, timeout:float, collection_name:str, dense_vector_name:str, sparse_vector_name:str, vectorizer):
        super().__init__(host=host, port=port, timeout=timeout)
        self._vectorizer = vectorizer
        self._collection_name:str = collection_name
        self._dense_vector_name:str = dense_vector_name
        self._sparse_vector_name:str = sparse_vector_name

    def fetch(self, query:str, top_k=10) -> list[dict]:

        """Fetch relevant chunks from Vector Database """

        # Encode Query
        dense_emb, sparse_emb = self._vectorizer.encode([query], encoder_type='both')

        # Handle sparse tensor correctly
        sparse_tensor = sparse_emb[0].coalesce()
        idx = sparse_tensor.indices()
        if sparse_tensor.dim() == 1 or idx.size(0) == 1:
            indices = idx[0].cpu().tolist()
        else:
            indices = idx[1].cpu().tolist()
        values = sparse_tensor.values().cpu().tolist()

        # ===================  Top 100 DENSE + Top 100 SPARSE -> DBSF -> Top 1 Concept  =======================
        search_res = self.query_points(
            collection_name=self._collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_emb[0].tolist(),
                    using=self._dense_vector_name,
                    limit=100,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=self._sparse_vector_name,
                    limit=100,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            with_payload=True,
            with_vectors=False,
            limit=top_k,
        ).points

        return [point.payload for point in search_res]

class EntityHarmonizer(BaseFetcher):
    pass
    
class PubmedFetcher(BaseFetcher):
    pass

class ParentParagraphFetcher():
    def __init__(self, db_path):
        self._db_path = db_path
        self._engine = create_engine(f"sqlite:///{db_path}")
        self._session = sessionmaker(bind=self._engine)()

    def get_parent_paragraphs(self, retrieved_docs: list[dict]) -> dict[int,dict[str,str]]:
        """Retrieve parent paragraphs from SQL Database.
        Args:
            retrieved_docs (list[dict]): List of chunks.
            session: SQLAlchemy session to query the Paragraph table.
        Returns:
            dict: A mapping of paragraph IDs to their corresponding paragraph text and pmcid_or_doi.
        """
        paragraph_map = {}
        for id, doc_dict in enumerate(retrieved_docs, start=1):
            pid = doc_dict.get("parent_paragraph_id")
            paragraph = self._session.query(BiomedicalParagraph).filter_by(paragraph_id=pid).first()
            
            if len(paragraph.paragraph_text.split()) > 6:
                paragraph_map[id] = {
                    "paragraph_text": paragraph.paragraph_text,
                    "pmcid_or_doi": paragraph.pmcid
                }
        return paragraph_map 
