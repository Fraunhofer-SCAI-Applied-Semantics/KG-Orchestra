"""
KG-Orchestra __main__ entry point.

This script orchestrates:

- CLI argument parsing
- (Re)starting the local Ollama server

- Initializing vectorizers and clients (UMLS harmonizer, PubMed fetcher, SQLite parent paragraph fetcher)
- Building the PubMed/PMC web pipeline

- Loading the seed KG from Neo4j or CSV
- Running the KG enrichment loop

Notes:

- Do not modify the logic or flow; only comments, docstrings, and logging have been added to replace print statements.

"""

import os
import subprocess
import argparse
import time
import logging
from sentence_transformers import SentenceTransformer
from modules.agents import *
from modules.clients import EntityHarmonizer, PubmedFetcher, ParentParagraphFetcher
from modules.neo4j_io import SeedKG, KGEnricher
from modules.pubmed import build_pipeline

# Configure logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # ===================================================================== Terminal Commands =====================================================================

    # Define CLI and arguments (keep as-is for compatibility with other modules)

    parser = argparse.ArgumentParser()

    # Neo4J

    parser.add_argument("--neo4j_url", type=str, default="bolt://localhost:7687", help="Neo4J url (eg. bolt://localhost:7687)")
    parser.add_argument("--neo4j_username", type=str, default="neo4j", help="Neo4J username (eg. neo4j)")
    parser.add_argument("--neo4j_password", type=str, default="55555555", help="Neo4J password")
    parser.add_argument("--neo4j_database", type=str, default="seed-kg", help="Neo4J Database Name")
    parser.add_argument("--start", type=int, default=0, help="Start from Triplet ...")
    parser.add_argument("--end", type=int, default=-1, help="End at Triplet ...")

    # Ollama

    parser.add_argument("--ollama_port", type=int, default=11434, help="Ollama Port")
    parser.add_argument("--llm", type=str, default="qwen3:32b", help="Ollama LLM ID")
    parser.add_argument("--llm_temperature", type=float, default=0.0, help="Ollama Model Temperature (default = 0.0)")
    parser.add_argument("--ollama_module", type=str, default="ollama/0.11.10-GCCcore-14.3.0-CUDA-12.9.1", help="Ollama Module on Cluster")

    # Qdrant UMLS

    parser.add_argument("--umls_qdrant_p", type=int, default=11333, help="UMLS Qdrant Vector Database Port")
    parser.add_argument("--umls_collection_name", type=str, default="all-MiniLM-L12-v2-splade-v3-umls-synonyms-with-types", help="UMLS Qdrant Vector Database - Collection Name")
    parser.add_argument("--umls_dv_name", type=str, default="all-MiniLM-L12-v2", help="UMLS Qdrant Vector Database - dense vector name")
    parser.add_argument("--umls_sv_name", type=str, default="splade-v3", help="UMLS Qdrant Vector Database - sparse vector name")
    parser.add_argument("--umls_dense_m", type=str, default='sentence-transformers/all-MiniLM-L12-v2', help="UMLS Dense Embedding Model HuggingFace ID")
    parser.add_argument("--umls_sparse_m", type=str, default="naver/splade-v3", help="UMLS Sparse Embedding Model HuggingFace ID")

    # Qdrant Biomedical Corpora

    parser.add_argument("--bio_qdrant_p", type=int, default=10333, help="Biomedical Corpora Qdrant Vector Database Port")
    parser.add_argument("--bio_collection_name", type=str, default="neurodegenerative_diseases_papers", help="Biomedical Corpora Qdrant Vector Database - Collection Name")
    parser.add_argument("--bio_dv_name", type=str, default="nomic-embed-text-v2-moe", help="Biomedical Corpora Qdrant Vector Database - dense vector name")
    parser.add_argument("--bio_sv_name", type=str, default="splade-v3", help="Biomedical Corpora Qdrant Vector Database - sparse vector name")
    parser.add_argument("--bio_dense_m", type=str, default='nomic-ai/nomic-embed-text-v2-moe', help="Biomedical Corpora Dense Embedding Model HuggingFace ID")
    parser.add_argument("--bio_sparse_m", type=str, default="naver/splade-v3", help="Biomedical Corpora Sparse Embedding Model HuggingFace ID")

    # Meta Data

    parser.add_argument("--trial", type=str, default="kg_orchestra_trial", help="Trial Name")
    parser.add_argument("--user_email", type=str, default="ahmed.hossameldin.hussein.mohamed@scai.fraunhofer.de", help="User Email for PubMed")
    parser.add_argument("--paragraph_db_path", type=str, default="/home/amohamed/workspace/databases/ndd_para_paragraphs.db", help="Parent Paragraphs SQL Database Path")
    parser.add_argument("--seed_csv_path", type=str, default=None, help="Path to Seed Triplets CSV File")
    parser.add_argument("--enriched_csv_path", type=str, default=None, help="Path to Enriched Triplets CSV File")

    # Retrieval Configuration

    parser.add_argument("--top_k", type=int, default=10, help="Top K for Retieval Pipeline")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout for Models")
    args = parser.parse_args()

    # ===================================================================== Ollama Server (re)start =====================================================================

    # Keep the exact commands; replace prints with logging only.

    logger.info("starting Ollama Server ...")
    subprocess.run(f"module load {args.ollama_module}", shell=True)
    subprocess.run("killall -9 ollama", shell=True)
    time.sleep(10)
    subprocess.run("ollama serve &", shell=True)

    # ===================================================================== Initiate Clients and Agents =====================================================================

    # Initialize vectorizers (Dense + Sparse)

    bio_vectorizer = Vectorizer(dense_m=args.bio_dense_m, sparse_m=args.bio_sparse_m)
    umls_vectorizer = Vectorizer(dense_m=args.umls_dense_m, sparse_m=args.umls_sparse_m)

    # Initialize UMLS Entity Harmonizer (Qdrant)

    umls_harmonizer = EntityHarmonizer(
        host="localhost",
        port=args.umls_qdrant_p,
        timeout=args.timeout,
        collection_name=args.umls_collection_name,
        dense_vector_name=args.umls_dv_name,
        sparse_vector_name=args.umls_sv_name,
        vectorizer=umls_vectorizer
    )
    logger.info("UMLS FETCHER IS READY")

    # Initialize PubMed Fetcher (Qdrant)

    ad_pubmed_fetcher = PubmedFetcher(
        host="localhost",
        port=args.bio_qdrant_p,
        timeout=args.timeout,
        collection_name=args.bio_collection_name,
        dense_vector_name=args.bio_dv_name,
        sparse_vector_name=args.bio_sv_name,
        vectorizer=bio_vectorizer
    )
    logger.info("PUBMED FETCHER IS READY")

    # Initialize SQLite Parent Paragraph Fetcher

    parent_paragraph_fetcher = ParentParagraphFetcher(db_path=args.paragraph_db_path)

    # Initialize PubMed/PMC Web Pipeline (NCBI)

    pubmed_web_fetcher = build_pipeline(email=args.user_email, dense_ranker=bio_vectorizer)

    # ===================================================================== Load Seed KG =====================================================================

    # Supports both CSV and Neo4j modes; behavior controlled by provided CLI flags.

    seed_kg = SeedKG(
        seed_csv_path=args.seed_csv_path,
        enriched_csv_path=args.enriched_csv_path,
        url=args.neo4j_url,
        auth=(args.neo4j_username, args.neo4j_password),
        entity_harmonizer=umls_harmonizer,
        llm_model=args.llm,
        database=args.neo4j_database
    )

    seed_kg.load_seed_kg(start_from=args.start, end_at=args.end)
    logger.info(f"Databaset {args.neo4j_database} is Loaded.")

    # ===================================================================== Define Output Folder =====================================================================

    # Output folder: ./.output/{trial_title}/{llm-id}

    logger.info("+" * 150)
    logger.info(f"Model {args.llm} is running...")
    trial_title = f"{args.trial}_{args.neo4j_database}_{args.start}_{args.end}"
    output_folder = f"./.output/{trial_title}/{args.llm.replace(':','-')}"

    # ===================================================================== Start KG Enricher Loop =====================================================================

    # Orchestrates: validation, retrieval, pathway construction, alignment, matching, validation/repair, and KG updates.

    kg_loop_enricher = KGEnricher(
        llm_name=args.llm,
        temperature=args.llm_temperature,
        seed_kg=seed_kg,
        pubmed_fetcher=ad_pubmed_fetcher,
        parent_paragraph_fetcher=parent_paragraph_fetcher,
        pubmed_web_fetcher=pubmed_web_fetcher,
        export_folder=output_folder,
        top_k=args.top_k,
        timeout=args.timeout,
        ollama_port=args.ollama_port
    )

    kg_loop_enricher.enrich_triplets()
    logger.info(f"{trial_title.replace('_', '/')} >> Trial is Done [LLM used: {args.llm}!")

if __name__ == "__main__":
    main()