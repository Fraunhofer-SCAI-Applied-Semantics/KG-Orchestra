import os
import uuid
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import psutil
from sentence_transformers import SentenceTransformer
from kg_orchestra.modules.db_models import Session, Paragraph, create_tables
from lxml import etree
import xml.etree.ElementTree as ET
import re
import sys
import traceback
import numpy as np
import torch
import gc
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
    HnswConfigDiff,
    models
)
from sentence_transformers import SparseEncoder
from beir.datasets.data_loader import GenericDataLoader

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def extract_paragraphs_with_titles(xml_file_path):
    """
    Extract plain text paragraphs from PMC XML.
    For body sections, prepend section/subsection titles as first sentence.
    Skip tables, figures, captions.
    """
    # Load XML
    with open(xml_file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()

    root = etree.fromstring(xml_content.encode('utf-8'))

    # Find relevant nodes
    body_nodes = root.xpath('.//body')

    paragraphs = []

    def clean_text(xml_node):
        soup = BeautifulSoup(etree.tostring(xml_node, encoding='unicode'), 'lxml')
        for tag in soup(['xref', 'table', 'table-wrap', 'fig', 'ext-link', 'inline-formula']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)

    # 2) Extract from Body — handle <sec> and <title>
    def process_section(section):
        # Get this section's title
        title = section.find('title')
        title_text = clean_text(title) if title is not None else ''

        # For each <p> directly under this section (not in nested sec)
        for elem in section:
            if elem.tag == 'title':
                continue  # already got it
            elif elem.tag == 'p':
                p_text = clean_text(elem)
                if p_text:
                    paragraphs.append(p_text)
            elif elem.tag in ['sec', 'subsec']:
                # Recursively process subsections
                process_section(elem)

    for body in body_nodes:
        for section in body.xpath('./sec'):
            process_section(section)

    return [para for para in paragraphs if len(para.split()) > 6]  # Filter out very short paragraphs

def extract_chunks_from_search_text(xml_file_path, chunk_words=150, overlap_chars=200):
    """
    Extracts text from <meta-name>search-text</meta-name> and chunks it into ~150-word segments
    with 200-character overlaps. Assumes unstructured flat content (no <body>).
    """
    with open(xml_file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()

    root = ET.fromstring(xml_content)

    # Find search-text meta-value
    search_text = None
    for meta in root.findall(".//custom-meta"):
        name = meta.find("meta-name")
        if name is not None and name.text == "search-text":
            value = meta.find("meta-value")
            if value is not None and value.text:
                search_text = value.text.strip()
                break

    if not search_text:
        return []

    # Normalize whitespace
    search_text = re.sub(r'\s+', ' ', search_text)

    # Split into words
    words = search_text.split()
    chunks = []
    start_idx = 0

    while start_idx < len(words):
        end_idx = start_idx + chunk_words
        word_chunk = words[start_idx:end_idx]
        chunk_text = " ".join(word_chunk)
        chunks.append(chunk_text)

        # Find next start by backtracking ~200 chars
        if end_idx >= len(words):
            break  # last chunk

        # Backtrack start position by ~200 characters
        # Step size in words will depend on avg word length (~6 chars)
        backtrack_words = overlap_chars // 6
        start_idx = max(end_idx - backtrack_words, 0)

    return chunks

def chunk_paragraphs(original_p, tokenizer, max_length=512):
    stack = [original_p]   # LIFO to preserve left-to-right order (push right, then left)
    chunks = []

    while stack:
        text = stack.pop()
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            continue

        n_tokens = len(tokenizer.encode(text, add_special_tokens=True))

        if n_tokens <= max_length:
            chunks.append(text)
            continue
        
        # Prefer word-based splitting
        try:
            words = word_tokenize(text)
        except Exception:
            words = text.split()

        left = right = None

        if len(words) >= 6:
            L = len(words)
            q1 = max(1, L // 4)            # 25%
            q3 = min(L - 1, L - q1)        # 75%, guaranteed < L
            left = " ".join(words[:q3]).strip()
            right = " ".join(words[q1:]).strip()
        else:
            # Ignore short leftouts
            continue

        # Push right then left so left is processed first (preserves original order)
        if right:
            stack.append(right)
        if left:
            stack.append(left)

    return chunks

def process_article(xml_files, pmc_folder):
    pmcid_paragraph_map = {}
    for xml_file in xml_files:
        pmcid = os.path.splitext(xml_file)[0]
        full_path = os.path.join(pmc_folder, xml_file)
        paragraphs = extract_paragraphs_with_titles(full_path)

        if len(paragraphs) < 5:
            paragraphs = extract_chunks_from_search_text(full_path)

        pmcid_paragraph_map[pmcid] = paragraphs

    session = Session()

    pmcids = []
    paragraph_ids = []
    batch_sentences = []
    for pmcid, paragraphs in pmcid_paragraph_map.items():
        for paragraph_text in paragraphs:
            paragraph_id = str(uuid.uuid4())

            # Save paragraph to SQLite via SQLAlchemy
            paragraph = Paragraph(paragraph_id=paragraph_id, pmcid=pmcid, paragraph_text=paragraph_text)
            session.add(paragraph)

            # Process paragraph into subparagraphs if len > 512
            sub_paragraphs = chunk_paragraphs(paragraph_text, tokenizer=dense_tokenizer)

            for sub_p in sub_paragraphs:
                pmcids.append(pmcid)
                paragraph_ids.append(paragraph_id)
                batch_sentences.append(sub_p)
        
    with torch.no_grad():
        encoded_input = dense_tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        encoded_input = {k: v.cuda(non_blocking=True) for k, v in encoded_input.items()}

        with torch.amp.autocast(device_type=device):
            model_output = dense_model(**encoded_input)
            batch_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        
        dense_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1).cpu().numpy()

        # Sparse embeddings from SPLADE (PyTorch sparse tensors)
        sparse_embeddings = sparse_model.encode(batch_sentences)  # list/torch sparse

    batch_points = []
    
    lengths = {len(pmcids), len(paragraph_ids), len(batch_sentences), len(dense_embeddings), len(sparse_embeddings)}

    if len(lengths) > 1:
        print(f"Error: Mismatched lengths detected -> "
            f"pmcids={len(pmcids)}, "
            f"paragraph_ids={len(paragraph_ids)}, "
            f"batch_sentences={len(batch_sentences)}, "
            f"dense_embeddings={len(dense_embeddings)}, "
            f"sparse_embeddings={len(sparse_embeddings)}")
        sys.exit("Stopping script due to inconsistent lengths.")

    for sentence_id, (pmcid, paragraph_id, sentence, dense_vec, sparse_vec) in enumerate(zip(pmcids, paragraph_ids, batch_sentences, dense_embeddings, sparse_embeddings)):
        sentence_id = str(uuid.uuid4())
        payload = {
            "id_sentence": sentence_id,
            "sub_paragraph_text": sentence,
            "parent_paragraph_id": paragraph_id,
            "pmcid": pmcid
        }

        dense_vec = dense_vec.tolist()

        # Handle sparse tensor correctly
        sparse_tensor = sparse_vec.coalesce()
        idx = sparse_tensor.indices()
        if sparse_tensor.dim() == 1 or idx.size(0) == 1:
            indices = idx[0].cpu().tolist()
        else:
            indices = idx[1].cpu().tolist()
        values = sparse_tensor.values().cpu().tolist()

        point = PointStruct(
            id=sentence_id,
            vector={
                    "nomic-embed-text-v2-moe": dense_vec,
                    "splade-v3": {"indices": indices, "values": values},
            },
            payload=payload)
        batch_points.append(point)

    for start_id in range(0, len(batch_points), 500):
        batch = batch_points[start_id:start_id+500]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    # Free CUDA
    del encoded_input, model_output, batch_embeddings
    torch.cuda.empty_cache()
    gc.collect()

    session.commit()
    session.close()

def process_all(pmc_folder):
    create_tables()

    success_csv_path = 'ndd_para_report.csv'
    failed_csv_path = 'failed_ndd_para_report.csv'

    # Collect all XML files
    articles = set([f for f in os.listdir(pmc_folder) if f.endswith(".xml")])
    print(f"{len(articles)} Articles to be processed.")

    # If CSV exists, filter out already processed articles
    if os.path.exists(success_csv_path):
        df = pd.read_csv(success_csv_path)
        existing_articles = set(df['article'].dropna().tolist())
        print(f"{len(existing_articles)} Articles has already been processed.")
        # articles = [a for a in articles if a not in existing_articles]
        articles = articles - existing_articles
        print(f"{len(articles)} Remaining Articles to be processed.")

    articles_unique = list(articles)
    
    for start_id in tqdm(range(0,len(articles_unique),BATCH_SIZE), desc="Processing articles", unit=f'{BATCH_SIZE} Articles'):
        try:
            files = articles_unique[start_id:start_id+BATCH_SIZE]

            process_article(files, pmc_folder)
            status = 'success'
            # Create a DataFrame for the current report
            new_rows = pd.DataFrame([{'article': file, 'status': status} for file in files])

            # Append to CSV incrementally
            if os.path.exists(success_csv_path):
                new_rows.to_csv(success_csv_path, mode='a', header=False, index=False)
            else:
                new_rows.to_csv(success_csv_path, index=False)
        except Exception as e:
            status = str(e)
            # Create a DataFrame for the current report
            new_rows = pd.DataFrame([{'article': file, 'status': status} for file in files])

            # Append to CSV incrementally
            if os.path.exists(failed_csv_path):
                new_rows.to_csv(failed_csv_path, mode='a', header=False, index=False)
            else:
                new_rows.to_csv(failed_csv_path, index=False)

            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

            if 'disconnect' in status.lower() or 'timeout' in status.lower():
                print(f"Files {files} have Error: {status}")
                break

if __name__ == "__main__":

    # Models ID from HuggingFace (DENSE + SPARSE)
    DENSE_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
    SPARSE_MODEL = "naver/splade-v3"

    # Load DENSE model
    dense_tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, trust_remote_code=True, use_fast=True)
    dense_model = AutoModel.from_pretrained(DENSE_MODEL, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = max(torch.cuda.device_count(), 1)
    if num_gpus > 1:
        print(f"Number of GPUs = {num_gpus}")
        dense_model = torch.nn.DataParallel(dense_model)
    if device == "cuda":
        dense_model.cuda()
    dense_model.eval()

    sparse_model = SparseEncoder(SPARSE_MODEL, device='cuda', trust_remote_code=True)

    BATCH_SIZE = 5 * num_gpus

    print("Model loaded successfully!")

    # Connect to local Qdrant (adjust URL and timeout as needed)
    client = QdrantClient(url="http://localhost:10333", timeout=600.0)

    # Name of the collection
    COLLECTION_NAME = 'neurodegenerative_diseases_papers'

    print("qdrant connected successfully!")

    # Define collection schema (Qdrant uses vector size, distance metric, and optional payload schema)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "nomic-embed-text-v2-moe": models.VectorParams(
                size=768,   # Important: must match model output size.
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "splade-v3": models.SparseVectorParams(modifier=models.Modifier.IDF
            )
        },
        hnsw_config=HnswConfigDiff(
            m=64,
            ef_construct=512
        )
    )
    print("Schema defined successfully!")
    print("Starting article processing...")
    
    # Define PMC folder path (Papers to be processed)
    pmc_folder = "/home/bio/groupshare/amohamed/workspace/alzminer/data/pubmed_ad/neurodegenerative_diseases_pmc"

    # Process all articles in the folder
    process_all(pmc_folder)

    print("✅ All articles processed successfully!")

    # Ensure the SQLite session is properly closed
    Session.close_all()
    print("SQLite session closed.")
