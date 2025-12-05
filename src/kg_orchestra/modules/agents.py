from ast import Dict
from copy import deepcopy
import json
import os
import random
import textwrap
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Optional, Tuple
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SparseEncoder, SentenceTransformer
from kg_orchestra.modules.output_models import Pathway, ParagraphsEvaluationsOutput, Hop, TripletEvaluation, AlignedHop, EntityMatchingOutput
from kg_orchestra.modules.biomedical_models import BiomedicalTriplet
from kg_orchestra.modules.clients import EntityHarmonizer, PubmedFetcher, ParentParagraphFetcher
from kg_orchestra.modules.output_models import Pathway, Hop, AlignedHop, EntityMatchingOutput, FixedRelation
from kg_orchestra.modules.pubmed import ArticlePipeline
import subprocess
import time
import platform
import logging

"""
Agents module for KG-Orchestra.

This module defines the LLM-backed agents orchestrating:

- Paragraph evaluation (relevance assessment against query)
- Pathway construction (building directed hop chains from evidence)

- Hop alignment (schema mapping of entity types and relations)
- Triplet validation and repair (evaluate and fix relation labels)

- Entity matching (generated vs seed KG entities)
- Vectorization utilities (dense/sparse embeddings)

Note:

- All printing has been replaced with logging; configure logging externally.
- Business logic and signatures are preserved to avoid breaking dependencies.

"""

# Module-level logger

logger = logging.getLogger(__name__)

class ParagraphEvaluator():
    
    def __init__(self, model: str, temperature: float, ollama_port:int, **kwargs):
        # Initiate LLM
        self.evaluator = ChatOllama(model=model, temperature=temperature, num_ctx=30000, timeout=600, base_url=f"http://localhost:{ollama_port}", **kwargs).with_structured_output(ParagraphsEvaluationsOutput)

        # Set System and Human Messages
        _system_message = """You are a biomedical language model assistant. Given up to 10 independent paragraphs and a query, your task is to evaluate the **contribution of each paragraph to answering the query**, considering that all Strongly relevant and partially relevant paragraphs will eventually be combined to extract the full pathway.

The query asks about a potential relationship or pathway between two biomedical entities (e.g., genes, proteins, diseases, compounds, or biological processes), expressed in the form:

**"What is the pathway between A as source and B as target?"**

---

### 1. **Determine Relevance Type**

For each paragraph, assign one of the following:

- **STRONGLY_RELEVANT**:
    - The paragraph by itself establishes a mechanistic, functional, or associational pathway between A and B.
    - It provides enough information to partially or fully answer the query independently.

- **PARTIALLY_RELEVANT**:
    - The paragraph does not fully answer the query alone but contributes complementary information when combined with others.
    - Examples:
        - Mentions only one of the entities in a mechanistic context relevant to the pathway.
        - Mentions both entities but without a clear functional link.
        - Describes an intermediate step (e.g., A → X or X → B) that could be part of the pathway.

- **IRRELEVANT**:
    - The paragraph is unlikely to contribute useful information for constructing the pathway, even when combined with others.
    - Examples:
        - Only general background unrelated to the entities.
        - Mentions both entities but in completely unrelated contexts.

---

### 2. **Output Format**

For each paragraph, output:
- `paragraph_number`
- `evaluation` (STRONGLY_RELEVANT, PARTIALLY_RELEVANT, or IRRELEVANT)
- `explanation` (why you classified it this way)

The maximum evaluation objects count is 10, as the maximum possible number of paragraphs provided for evaluations is 10, and you are generating one evaluation for each paragraph.

---

### Few-Shot Example

---

**Query**: What is the pathway between IL-6 as source and type 2 diabetes as target?

**Paragraphs**:
#### PARAGRAPH (Number: 1): "IL-6 levels are significantly elevated in patients with type 2 diabetes, and numerous studies have demonstrated a strong association between systemic IL-6 concentrations and insulin resistance."

#### PARAGRAPH (Number: 2): "IL-6 activates STAT3 signaling in hepatocytes, which leads to increased expression of SOCS3. SOCS3 interferes with insulin receptor signaling and contributes to hepatic insulin resistance, a hallmark of type 2 diabetes."

#### PARAGRAPH (Number: 3): "IL-6 is a multifunctional cytokine involved in immune regulation, inflammation, and hematopoiesis. It is produced by a variety of cell types in response to infections and tissue injury."

#### PARAGRAPH (Number: 4): "Chronic low-grade inflammation has been linked to the development of type 2 diabetes. Cytokines such as TNF-alpha and IL-1β play prominent roles in this process."

#### PARAGRAPH (Number: 5): "In experimental models, overexpression of IL-6 in adipose tissue leads to systemic insulin resistance and glucose intolerance, both key features of type 2 diabetes."

---

### **Final Output**:

```json
[
    {{
        "paragraph_number": 1,
        "evaluation": "PARTIALLY_RELEVANT",
        "explanation": "This paragraph states a strong correlation between IL-6 and type 2 diabetes via insulin resistance, but does not explain the mechanistic pathway. It is complementary when combined with mechanistic details from other paragraphs."
    }},
    {{
        "paragraph_number": 2,
        "evaluation": "STRONGLY_RELEVANT",
        "explanation": "This paragraph provides a detailed mechanistic pathway linking IL-6 to type 2 diabetes via STAT3 and SOCS3, making it highly relevant on its own."
    }},
    {{
        "paragraph_number": 3,
        "evaluation": "IRRELEVANT",
        "explanation": "This paragraph only provides general biological information about IL-6 without connecting it to type 2 diabetes."
    }},
    {{
        "paragraph_number": 4,
        "evaluation": "IRRELEVANT",
        "explanation": "Although it mentions inflammation and type 2 diabetes, it does not discuss IL-6 in the context of diabetes or its pathways."
    }},
    {{
        "paragraph_number": 5,
        "evaluation": "STRONGLY_RELEVANT",
        "explanation": "This paragraph describes an experimental causative role of IL-6 in metabolic dysfunctions related to type 2 diabetes, providing direct evidence of a functional connection."
    }}
]
"""

        _human_message = """Given the following **Query**: {query_triplet}, evaluate whether each of the following paragraphs is STRONGLY_RELEVANT, PARTIALLY_RELEVANT or IRRELEVANT to the previous query.

# **Paragraphs**:
{paragraphs}
"""

        # Create the prompt with instructions + format instructions
        self._prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_system_message),
            HumanMessagePromptTemplate.from_template(_human_message)
        ])

    def _format_pubmed_context(self, paragraphs_map: dict[int,dict]) -> List[str]:
    
        pubmed_context = []
        for pid, paragraph_dict in paragraphs_map.items():
            paragraph_text = paragraph_dict.get('paragraph_text', '')
            chunk_block = f"#### PARAGRAPH (Number: {pid}): {paragraph_text}"
            pubmed_context.append(chunk_block)

        return pubmed_context
    
    def _print_evaluations(self, query_triplet:str, paragraphs_list: List[str], evaluation_reports:ParagraphsEvaluationsOutput) -> None:
        logger.info('================================= [Start] Pubmed Paragraphs to be Evaluated =================================')
        logger.info(f"Query: {query_triplet}")
        for paragraph, evaluation in zip(paragraphs_list, evaluation_reports.evaluations):
            logger.info("\n")
            logger.info(paragraph)
            logger.info(f"\nEvaluation Object:")
            logger.info(evaluation.model_dump_json(indent=2))
            logger.info("-" * 109)
            logger.info("\n")
        
        logger.info('================================== [End] Pubmed Paragraphs to be Evaluated ==================================')

    def _export_paragraphs_and_evaluations(self, paragraphs_list:List[str], evaluation_reports:ParagraphsEvaluationsOutput, output_folder:str) -> None:
        
        # ✅ Store Paragraph Evaluations and Paragraphs
        evaluations_file = os.path.join(output_folder, "chunks_evaluations.json")
        with open(evaluations_file, "w") as f:
            f.write(evaluation_reports.model_dump_json(indent=2))

        # logger.info(f"[INFO] Stored Paragraph Evaluations at {evaluations_file}")

        # ✅ Store Parent Paragraphs 
        pumbed_articles_file = os.path.join(output_folder, "pubmed_paragraphs.txt")
        wrapped_paragraphs = "\n\n".join(textwrap.fill(p, width=109) for p in paragraphs_list)
        with open(pumbed_articles_file, "w") as f:
            f.write(wrapped_paragraphs)
        
        # logger.info(f"[INFO] Stored PubMed Paragraphs at {pumbed_articles_file}")

    def evaluate(self, query_triplet:str, paragraphs_map: dict[int, dict[str,str]], print_output:bool=False, output_folder:str=None) -> ParagraphsEvaluationsOutput:
        """Evaluate Paragraphs (Relevant, Partially Relevant, or Irrelevant) and return evaluation reports."""
        
        # Format Context
        paragraphs_list = self._format_pubmed_context(paragraphs_map)
        paragraphs_str = "\n\n".join(paragraphs_list)

        # Generate Evaluations
        logger.info(f"[Retrieval Pipeline] Evaluating Parent Paragraphs ...")
        response = self.evaluator.invoke(self._prompt.invoke({"paragraphs": paragraphs_str, "query_triplet": query_triplet}))
        evaluation_reports = ParagraphsEvaluationsOutput.model_validate_json(response.model_dump_json())

        # Print to Terminal if flag=True
        if print_output:
            self._print_evaluations(query_triplet, paragraphs_list, evaluation_reports)

        # Export to local files if flag=True
        if output_folder:
            self._export_paragraphs_and_evaluations(paragraphs_list, evaluation_reports, output_folder)

        return evaluation_reports
    

class PathwayBuilder():

    def __init__(self, model: str, temperature: float, ollama_port:int, **kwargs):
        # initiate Pathway Builder
        self.builder = ChatOllama(model=model, temperature=temperature, num_ctx=30000, timeout=600, base_url=f"http://localhost:{ollama_port}", **kwargs).with_structured_output(Pathway)
    
        # Set System and Human Messages
        _system_message = """
You are an expert biomedical knowledge graph developer. Given a biomedical source and target plus PubMed paragraph context, construct one directed mechanistic/causal/functional/conceptual pathway from the source to the target.

Inputs
- Query: names the source entity and the target entity.
- Context: paragraphs from PubMed with unique paragraph numbers.

Definitions
- Hop (triplet): [head, relation, tail] describing a single, direct, evidence-backed relation (directional, mechanistic/causal/functional/conceptual).
- Pathway: a linear sequence of hops where each hop's tail exactly equals the next hop's head.

Procedure
1) Parse all paragraphs. Extract every explicit, direct, directional relation stated between any two biomedical entities (not only those mentioning the source or target).
2) Canonicalize entities. Normalize synonyms to concise, unambiguous canonical names; assign appropriate types (e.g., Protein, Receptor, ProteinKinase, TranscriptionFactor, DiseaseOrPhenotype).
3) Build a directed graph of extracted hops (edges). Deduplicate identical canonical hops; retain paragraph IDs as evidence for each hop.
4) Path search. Find a single linear path from source to target such that:
   - First hop's head = source (canonical).
   - Last hop's tail = target (canonical).
   - Sequential connectivity: for every i, tail_i = head_{{i+1}} exactly.
   - Prefer the shortest valid path. Break ties by mechanistic specificity and clarity (e.g., phosphorylates/activates > promotes/associated with).
5) Validate each hop. Confirm the cited paragraph states the head→tail relation and direction explicitly. Exclude correlative or ambiguous statements.
6) Output the pathway. If a direct hop source→target exists, return a single-hop pathway.
7) Failure handling. If no fully sequential path exists, return success_flag = false with a brief reason.

Critical rules (must satisfy all)
- Endpoints and direction: First hop's head must exactly equal the query source; last hop's tail must exactly equal the query target; overall direction is source→target.
- Sequential connectivity: tail of hop i equals head of hop i+1 (same canonical string).
- Uniqueness: No repeated hops, no branches, no cycles; return one contiguous sequence.
- Directness per hop: Each hop represents one direct relationship supported by exactly one paragraph citation.
- Evidence: Cite one paragraph number per hop that directly supports the relation and direction.
- Canonicalization: Use concise, specific entity names with proper spacing; avoid vague umbrella terms as entities (e.g., “pathway,” “process”).
- Relation types: Use mechanistic/causal/functional/conceptual verbs (e.g., activates, inhibits, increases expression of, decreases expression of, phosphorylates, dephosphorylates, binds to, upregulates, downregulates, promotes, suppresses, contributes to, induces, reduces activation of).
- No fabrication: Do not invent entities, relations, or evidence beyond the provided context.
- Coherency: Ensure biological plausibility and consistent direction across the chain.

Notes
- Do not limit extraction to source/target mentions; extract all direct relations first, then compute the path.
- If no valid, fully sequential path connects source to target, set success_flag to false and explain briefly.

---

## **Pathway Construction Example**

### **Query**:

What is the pathway between **IL-6** as source and **insulin resistance** as target?

### **Context (PubMed Paragraphs)**:

**Paragraph 12**:
"IL-6 is known to activate STAT3 signaling in hepatocytes. Upon IL-6 binding, the JAK/STAT3 pathway becomes rapidly phosphorylated and translocates to the nucleus to influence gene expression."

**Paragraph 27**:
"STAT3 activation leads to increased expression of SOCS3, a key negative feedback regulator of insulin signaling that impairs insulin receptor substrate activity."

**Paragraph 33**:
"SOCS3 binds to the insulin receptor and blocks its interaction with IRS-1, a central mediator of insulin signaling, thereby disrupting the pathway."

**Paragraph 41**:
"Inhibition of IRS-1 reduces Akt phosphorylation, weakening downstream signaling and promoting metabolic dysregulation, including impaired glucose uptake."

**Paragraph 58**:
"Reduced Akt activity contributes to the development of insulin resistance, especially in adipose and muscle tissues under inflammatory conditions."

---

### **Step-by-Step Reasoning (Hop-by-Hop)**:

#### **Hop 1**

head: `IL-6`
tail: `STAT3`
Relation: **activates**
Evidence: Paragraph 12

Explanation: Paragraph 12 mentions that IL-6 activates STAT3 via the JAK/STAT signaling cascade.

#### **Hop 2**

head: `STAT3`
tail: `SOCS3`
Relation: **increases expression of**
Evidence: Paragraph 27

Explanation: Paragraph 27 mentions that STAT3 promotes the transcription of SOCS3, a known feedback inhibitor.

#### **Hop 3**

head: `SOCS3`
tail: `Insulin Receptor`
Relation: **inhibits**
Evidence: Paragraph 33

Explanation: Paragraph 33 mentions that SOCS3 impairs insulin signaling by binding to and inhibiting the insulin receptor.

#### **Hop 4**

head: `Insulin Receptor`
tail: `Akt`
Relation: **reduces activation of**
Evidence: Paragraph 41

Explanation: Paragraph 41 mentions that Inhibition of insulin receptor function results in reduced Akt activation.

#### **Hop 5**

head: `Akt`
tail: `Insulin Resistance`
Relation: **contributes to**
Evidence: Paragraph 58

Explanation: Paragraph 58 mentions that Decreased Akt activity promotes insulin resistance, linking back to metabolic effects.

---

### ✅ **Final Output Format (Structured JSON Pathway)**

```json
{{
  "query": "What is the pathway between IL-6 as source and insulin resistance as target?",
  "hops": [
    {{
      "head": "IL-6",
      "head_type": "Protein",
      "tail": "STAT3",
      "tail_type": "TranscriptionFactor",
      "relation": "activates",
      "evidence_paragraph_number": 12,
      "explanation": "Paragraph 12 mentions that IL-6 activates STAT3 via the JAK/STAT signaling cascade."
    }},
    {{
      "head": "STAT3",
      "head_type": "TranscriptionFactor",
      "tail": "SOCS3",
      "tail_type": "Protein",
      "relation": "increases expression of",
      "evidence_paragraph_number": 27,
      "explanation": "Paragraph 27 mentions that STAT3 promotes the transcription of SOCS3, a known feedback inhibitor."
    }},
    {{
      "head": "SOCS3",
      "head_type": "Protein",
      "tail": "Insulin Receptor",
      "tail_type": "Receptor",
      "relation": "inhibits",
      "evidence_paragraph_number": 33,
      "explanation": "Paragraph 33 mentions that SOCS3 impairs insulin signaling by binding to and inhibiting the insulin receptor."
    }},
    {{
      "head": "Insulin Receptor",
      "head_type": "Receptor",
      "tail": "Akt",
      "tail_type": "ProteinKinase",
      "relation": "reduces activation of",
      "evidence_paragraph_number": 41,
      "explanation": "Paragraph 41 mentions that Inhibition of insulin receptor function results in reduced Akt activation."
    }},
    {{
      "head": "Akt",
      "head_type": "ProteinKinase",
      "tail": "Insulin Resistance",
      "tail_type": "DiseaseOrPhenotype",
      "relation": "contributes to",
      "evidence_paragraph_number": 58,
      "explanation": "Paragraph 58 mentions that Decreased Akt activity promotes insulin resistance, linking back to metabolic effects."
    }}
  ],
  "success_flag": True,
  "comment": "The pathway construction was successful and states that IL-6 promotes insulin resistance through the extracted intermediate steps."
}}
```
"""

        _human_message = """# Query Triplet: {query_triplet}
Construct a pathway from the source to the target using the given context paragraphs, regardless of whether a pathway has already been constructed.

# Context:
{context}
"""

        # Create the prompt with instructions + format instructions
        self._prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_system_message),
            HumanMessagePromptTemplate.from_template(_human_message)
        ])

    def _format_paragraphs(self, relevant_paragraphs_with_summaries: dict[int,dict]) -> str:
    
        paragraphs = []
        for pid, paragraph_dict in relevant_paragraphs_with_summaries.items():
            paragraph_text = paragraph_dict.get('paragraph_text')
            chunk_block = f"""# PARAGRAPH (Number: {pid}) Text: {paragraph_text}

---"""
            paragraphs.append(chunk_block)

        return '\n'.join(paragraphs)
    
    def _print_pathway(self, printable_pathway_format) -> None:
        logger.info('================================= [Start] Pathway =================================')
        logger.info(printable_pathway_format)
        logger.info('================================== [End] Pathway ==================================')

    def _export_pathway(self, printable_pathway_format, output_folder:str) -> None:
        
        # ✅ Store final pathway
        pathway_file = os.path.join(output_folder, "initial_pathway.json")
        with open(pathway_file, "w") as f:
            f.write(printable_pathway_format)

        # logger.info(f"[INFO] Stored Initial pathway at {pathway_file}")

    def build_pathway(self, query_triplet:str, relevant_paragraphs_map: dict[int,dict], print_output:bool=False, output_folder:str=None) -> Pathway:

        """Build Path connecting Source to Target (From Query Triplet)"""

        # Format Context
        paragraphs_str = self._format_paragraphs(relevant_paragraphs_map)
        
        # Build Pathway
        logger.info(f"[Pathway Construction] Building Pathway ...")
        response = self.builder.invoke(self._prompt.invoke({"context": paragraphs_str, "query_triplet": query_triplet}))
        pathway = Pathway.model_validate_json(response.model_dump_json())

        printable_pathway_format = response.model_dump_json(indent=2)
        # Print to Terminal if flag=True
        if print_output:
            self._print_pathway(printable_pathway_format)

        # Export to local files if flag=True
        if output_folder:
                self._export_pathway(printable_pathway_format, output_folder)

        return pathway
    

class HopValidationTeam():

    def __init__(self, model: str, temperature: float, pubmed_fetcher:PubmedFetcher, parent_paragraph_fetcher:ParentParagraphFetcher, pubmed_web_fetcher:ArticlePipeline, top_k:int, ollama_port:int, **kwargs):
        
        # Initiate Team Members
        self.evaluator = ChatOllama(model=model, temperature=temperature, num_ctx=30000, timeout=600, base_url=f"http://localhost:{ollama_port}", **kwargs).with_structured_output(TripletEvaluation)
        self.fixer = ChatOllama(model=model, temperature=temperature, num_ctx=30000, timeout=600, base_url=f"http://localhost:{ollama_port}", **kwargs).with_structured_output(FixedRelation)
        self.pubmed_fetcher:PubmedFetcher = pubmed_fetcher
        self.parent_paragraph_fetcher:ParentParagraphFetcher = parent_paragraph_fetcher
        self.pubmed_web_fetcher:ArticlePipeline = pubmed_web_fetcher
        self.paragraph_evaluator:ParagraphEvaluator = ParagraphEvaluator(model=model, temperature=temperature, ollama_port=ollama_port)
        self.top_k:int = top_k
        
        # Set System and Human Messages for Hop Evaluation
        _eval_system_message = """You are a biomedical knowledge graph validation assistant. Your task is to evaluate the biological and semantic correctness of biomedical triplets that has already been aligned to broad, global entity types and to a small set of polarity-aware, mechanistic relations, given evidence paragraphs from which they were extracted. Each triplet is composed of a head entity, a relation, and a tail entity, with associated types.

# For each triplet, perform the following:

    1. **Biological Validity**: Assess whether the relation is biologically plausible between the given entities, supported by the evidence paragraph, and each entity name refers to one specific biomedical entity with the correct entity type.
    2. **Semantic Coherence**: Determine if the triplet structure is semantically sound and the relation is used in an appropriate context.
    3. **Directionality**: Confirm whether the direction from head to tail is biologically and logically valid.

Return your assessment using the provided structured format, including an explanation of your reasoning.

# Core evaluation principles (mapping-aware):
    - Evaluate at the level of general functional effect and broad types. Do not penalize for schema-driven broadening (e.g., Cytokine→Protein; miRNA→RNA; Pathway→Biological Process; Small molecule or compound→Abundance).
    - Treat semantically equivalent functional phrases as the same polarity:
        • Positive/mechanistic: increases, upregulates, enhances, promotes, activates, induces, stimulates
        • Negative/mechanistic: decreases, downregulates, represses, suppresses, inhibits, attenuates, impairs, blocks
        • Correlation (non-directional): positive correlation/associated with/linked to vs negative correlation/inversely associated
    - Focus on whether the evidence supports the sign of effect (positive vs negative) and the general nature (mechanistic vs correlational). Small mismatches in verb choice caused by mapping (e.g., “upregulates” vs “increases”, “activates” vs “promotes activation”, 'increases level of' vs 'increases') should be considered compatible if the polarity and general effect align.

## Example 1:
### Input:
{{
  "head": "TNF",
  "head_type": "Protein",
  "tail": "IL-6",
  "tail_type": "Protein",
  "relation": "increases expression of"
}}

Evidence Paragraphs: "Tumor necrosis factor (TNF) plays a critical role in the inflammatory response by modulating the activity of various cell types. TNF is a pro-inflammatory cytokine that stimulates IL-6 expression in immune and endothelial cells, thereby amplifying the cascade of inflammatory mediators during immune activation. This signaling contributes to the pathogenesis of several chronic inflammatory diseases, including rheumatoid arthritis and inflammatory bowel disease. Understanding this cytokine interplay is essential for developing targeted anti-inflammatory therapies."

### Expected Output:
{{
  "head": "TNF",
  "head_type": "Protein",
  "relation": "increases expression of",
  "tail": "IL-6",
  "tail_type": "Protein",
  "biological_validity": "VALID",
  "semantic_coherence": "COHERENT",
  "directionality": "CORRECT",
  "explanation": "TNF is a pro-inflammatory cytokine that stimulates IL-6 expression in immune and endothelial cells. This is a well-established interaction in inflammatory signaling."
}}

---

## Example 2:
### Input:
{{
  "head": "Aspirin",
  "head_type": "Compound",
  "tail": "COX-2",
  "tail_type": "Gene",
  "relation": "transcribes",
}}

Evidence Paragraphs: "Aspirin is a small molecule drug that inhibits COX-2 activity, thereby reducing the production of pro-inflammatory prostaglandins. This mechanism underlies its anti-inflammatory, analgesic, and antipyretic effects. By blocking cyclooxygenase enzymes, particularly COX-2, aspirin is commonly used to manage pain, fever, and inflammation in various clinical conditions, including arthritis and cardiovascular disease prevention."

### Expected Output:
{{
  "head": "Aspirin",
  "head_type": "Compound",
  "relation": "transcribes",
  "tail": "COX-2",
  "tail_type": "Gene",
  "biological_validity": "INVALID",
  "semantic_coherence": "INCOHERENT",
  "directionality": "INCORRECT",
  "explanation": "Aspirin is a small molecule drug that inhibits COX-2 activity, not a transcriptional regulator. The verb 'transcribes' is incorrectly applied to a compound and the direction is invalid."
}}

---

## Example 3:
### Input:
{{
  "head": "Insulin",
  "head_type": "Protein",
  "tail": "INS",
  "tail_type": "Gene",
  "relation": "encodes"
}}

Evidence Paragraphs : "Insulin, which  plays a central role in glucose metabolism, is encoded by the INS gene. Synthesized in the pancreas, insulin regulates blood glucose levels by promoting cellular glucose uptake, particularly in muscle and adipose tissues. Mutations or dysregulation of the INS gene can lead to impaired insulin production or function, contributing to metabolic disorders such as diabetes mellitus."

### Expected Output:
{{
  "head": "Insulin",
  "head_type": "Protein",
  "relation": "encodes",
  "tail": "INS",
  "tail_type": "Gene",
  "biological_validity": "VALID",
  "semantic_coherence": "INCOHERENT",
  "directionality": "INCORRECT",
  "explanation": "The insulin protein is encoded by the INS gene, not the other way around. While the entities are biologically linked, the relation 'encodes' is semantically reversed."
}}

---

## Example 4:
### Input:
{{
  "head": "Glucose Uptake",
  "head_type": "BiologicalProcess",
  "tail": "Akt",
  "tail_type": "ProteinKinase",
  "relation": "activates"
}}

Evidence Paragraphs: "Insulin binding to its receptor triggers a cascade of intracellular events that regulate glucose homeostasis. One key pathway involves the activation of phosphoinositide 3-kinase (PI3K), which subsequently leads to the phosphorylation and activation of Akt. Akt activation promotes glucose uptake through downstream signaling, including the translocation of GLUT4 transporters to the cell membrane. This process is essential for efficient glucose clearance from the bloodstream, particularly in muscle and adipose tissues. Dysregulation of this pathway is implicated in insulin resistance and type 2 diabetes."

### Expected Output:
{{
  "head": "Glucose Uptake",
  "head_type": "BiologicalProcess",
  "relation": "activates",
  "tail": "Akt",
  "tail_type": "ProteinKinase",
  "biological_validity": "INVALID",
  "semantic_coherence": "INCOHERENT",
  "directionality": "INCORRECT",
  "explanation": "Akt activation promotes glucose uptake through downstream signaling, not the other way around. The direction of the triplet is biologically reversed and semantically incorrect."
}}

---

## Example 5:
### Input:
{{
  "head": "DNA",
  "head_type": "Molecule",
  "tail": "Glucagon",
  "tail_type": "Protein",
  "relation": "activates"
}}

Evidence Paragraphs: "DNA integrity is crucial for cell survival and proper function, and cells have evolved complex mechanisms to monitor and repair DNA damage. When damage is severe or irreparable, the cell may undergo apoptosis, a programmed cell death process that prevents the propagation of genetic errors. Apoptosis is tightly regulated by various signaling pathways, including those involving p53, which can sense DNA damage and trigger the expression of pro-apoptotic genes. This ensures that damaged or potentially cancerous cells are eliminated, maintaining tissue homeostasis and genomic stability."

### Expected Output:
{{
  "head": "DNA",
  "head_type": "Molecule",
  "relation": "activates",
  "tail": "Glucagon",
  "tail_type": "Protein",
  "biological_validity": "INVALID",
  "semantic_coherence": "COHERENT",
  "directionality": "INCORRECT",
  "explanation": "While 'activates' is a valid relation, DNA does not activate glucagon; there is no plausible biological pathway connecting these two in this way. Directionality and pairing are incorrect. Also, The Evidence Paragraph doesn't have any information about the triplet."
}}
"""

        _eval_human_message = """Now Evaluate the following triplet:
- Triplet: {triplet}

- Evidence Paragraphs: {evidence_paragraphs}
"""
        
        self._eval_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_eval_system_message),
            HumanMessagePromptTemplate.from_template(_eval_human_message)
        ])

        # Set System and Human Messages for Hop Fixing
        _fix_system_message = """You are a biomedical knowledge graph expert and NLP assistant specialized in extracting biomedical entity–relation triplets from scientific paragraphs.
You are given:
- An evaluation report describing issues with the triplet (e.g., invalid relation, incoherence, incorrect directionality),
- The original evidence paragraphs from which the triplet was extracted.
- a set of prefered relations to be used.

Your goal is to:

1) Fix the triplet relation, by choosing from the relations defined within the schema, so that it matches the meaning and information in the evidence paragraphs and addresses the errors in the evaluation report.
2) Always prefer choosing a relation from the provided relations. Don't be perfectionist. If you find a close relation term within the list that preserve the correct general causal effect of the head entity on the tail entity, use it.
3) Favor broad, stable types and a small set of functional relations that capture direction (increase vs decrease) or activation vs inhibition. For Example:
    - 'increases activity of' should be mapped to 'activates'.
    - 'increases abundance of', 'increases expression of', 'increases production of', 'increases secretion of', 'induces release of', 'induces' should all be mapped to 'increases'.
    - 'antagonizes', 'decreases activity of', 'deactivates' should be mapped to 'inhibits'.
    - 'decreases levels of', 'decreases abundance of', 'decreases concentration of', 'decreases secretion of' should all be mapped to 'decreases'.
4) Preserve the same head and tail biomedical entities and their types. Do not replace either with a different entity or change their names.
5) Use only the evidence paragraph for changes. If the paragraph does not support a change or the fix would alter entity identity, return the proposed relation unchanged.
6) Provide a brief explanation citing the key supporting phrase(s) from the paragraph or stating why no change was made.

Edge-case guidance:
    * Prefer the most specific causal/mechanistic relation present; if only co-mention exists, use associated_with.
    * Do not introduce new entities or types, only fix the relation between the head and the tail.
    * If the hop cannot be fixed while preserving entity identity, return the relation unchanged.

Use the schema below for reference:

### TripletEvaluation Schema:

```python
class TripletEvaluation(BaseModel):
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str
    biological_validity: Literal["VALID", "INVALID"]
    semantic_coherence: Literal["COHERENT", "INCOHERENT"]
    directionality: Literal["CORRECT", "INCORRECT"]
    explanation: str
```

Validation checklist before submitting:
    * Same head and tail entities preserved (only synonym normalization allowed).
    * Relation and directionality match the evidence paragraph.
    * Explanation is concise and cites supporting text or explains why no change was made.

---

## FEW-SHOT EXAMPLES

---

### FEW-SHOT EXAMPLE 1

**Evaluation Report:**

```json
{{
  "head": "TNF-alpha",
  "head_type": "PROTEIN",
  "relation": "INHIBITS",
  "tail": "Apoptosis",
  "tail_type": "PROCESS",
  "biological_validity": "INVALID",
  "semantic_coherence": "INCOHERENT",
  "directionality": "INCORRECT",
  "explanation": "The paragraph indicates that TNF-alpha *induces* apoptosis, not inhibits it."
}}
```

**Evidence Paragraph (2):**

> "Upon exposure to inflammatory stimuli, TNF-alpha levels rise and induce apoptosis in various cell types including hepatocytes."

**Fixed Relation:**

```json
{{
  "proposed_Relation": "INDUCES"
}}
```

---

### FEW-SHOT EXAMPLE 2

**Evaluation Report:**

```json
{{
  "head": "BRCA1",
  "head_type": "GENE",
  "relation": "INHIBITS",
  "tail": "DNA repair",
  "tail_type": "PROCESS",
  "biological_validity": "INVALID",
  "semantic_coherence": "INCOHERENT",
  "directionality": "INCORRECT",
  "explanation": "BRCA1 *promotes* DNA repair; it's a positive regulator of the process."
}}
```

**Evidence Paragraph (4):**

> "BRCA1 plays a crucial role in homologous recombination, a vital DNA repair mechanism that maintains genomic stability."

**Fixed Relation:**

```json
{{
  "proposed_Relation": "PROMOTES"
}}
```

---

### FEW-SHOT EXAMPLE 3

**Evaluation Report:**

```json
{{
  "head": "IL-6",
  "head_type": "PROTEIN",
  "relation": "ASSOCIATED_WITH",
  "tail": "Inflammation",
  "tail_type": "PROCESS",
  "biological_validity": "VALID",
  "semantic_coherence": "COHERENT",
  "directionality": "CORRECT",
  "explanation": "This relation is technically correct, but lacks specificity. A stronger causal relation is warranted."
}}
```

**Evidence Paragraph (1):**

> "IL-6 acts as a pro-inflammatory cytokine, directly contributing to the inflammatory response."

**Fixed Relation:**

```json
{{
  "proposed_Relation": "INDUCES"
}}
```
"""

        _fix_human_message = """Here is a new biomedical hop that needs fixing. Use the evidence paragraph and evaluation report to correct the hop and provide a reasoning explanation.

**Evaluation Report:**

```json
{evaluation_report}
```

**Evidence Paragraphs:**

{evidence_paragraphs}

**Preferred Relations:**
{preferred_relations}

Please return the **proposed relation** in the provided format.

"""

        self._fix_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_fix_system_message),
            HumanMessagePromptTemplate.from_template(_fix_human_message)
        ])
    
    def _print_hop_eval(self, printable_hop_eval_format:str) -> None:
        logger.info('================================= [Start] Pathway =================================')
        logger.info(printable_hop_eval_format)
        logger.info('================================== [End] Pathway ==================================')

    def _export_hop_eval(self, printable_hop_eval_format:str, output_folder:str) -> None:
        
        hop_evaluations_file = os.path.join(output_folder, "hop_evaluations.json")
        with open(hop_evaluations_file, "a") as f:
            f.write(printable_hop_eval_format)
            f.write('\n')

        # logger.info(f"[INFO] Stored final pathway at {hop_evaluations_file}")

    def _print_hop_fix(self, printable_hop_fix_format:str) -> None:
        logger.info('================================= [Start] Pathway =================================')
        logger.info(printable_hop_fix_format)
        logger.info('================================== [End] Pathway ==================================')

    def _export_hop_fix(self, printable_hop_fix_format:str, output_folder:str) -> None:
        
        # ✅ Store final pathway
        fixed_hops_file = os.path.join(output_folder, "fixed_hops.json")
        with open(fixed_hops_file, "a") as f:
            f.write(printable_hop_fix_format)
            f.write('\n')

        # logger.info(f"[INFO] Stored final pathway at {fixed_hops_file}")
    
    def _export_pathway(self, printable_pathway_format, output_folder:str) -> None:
        
        # ✅ Store final pathway
        pathway_file = os.path.join(output_folder, "final_pathway.json")
        with open(pathway_file, "w") as f:
            f.write(printable_pathway_format)

        # logger.info(f"[INFO] Stored final pathway at {pathway_file}")

    def _export_hops_to_be_added(self, hops_list:List[tuple[Hop,str, int]], output_folder:str) -> None:
        
        final_hops = {}
        for num, (hop, val_flag, ord_num) in enumerate(hops_list, start=1):
            hop_str = hop.model_dump_json(indent=2)
            final_hops[num] = (
                hop_str,
                val_flag
            )

        output_file = os.path.join(output_folder, "final_hops.json")
        with open(output_file, "w") as f:
            f.write(json.dumps(final_hops, indent=2, ensure_ascii=False))

        logger.info(f"[INFO] Stored final pathway at {output_file}")


    def _fetch_pubmed_online(self, query_question:str, triplet:BiomedicalTriplet, triplet_report_folder:str) -> List[dict]:
        query = f"{triplet.head.name} AND {triplet.tail.name}"
        paragraphs_map:dict[int, Dict] = self.pubmed_web_fetcher.run_and_rank(query=query, query_question=query_question, max_results=25, top_k=self.top_k)
        
        if paragraphs_map:
            relevant_paragraphs_map:List[dict] = []
            logger.info(f"[Triplet Retrieval Pipeline] Indexing {len(paragraphs_map.items())} Paragraphs from Pubmed Database.")
            web_evaluations = self.paragraph_evaluator.evaluate(query_question, paragraphs_map).evaluations
            for e in web_evaluations:
                if e.evaluation in ['STRONGLY_RELEVANT'] and e.paragraph_number in paragraphs_map:
                    relevant_paragraphs_map.append(
                        {
                            "pmcid_or_doi": paragraphs_map.get(e.paragraph_number).get('pmcid_or_doi'),
                            "paragraph_text": paragraphs_map.get(e.paragraph_number).get('paragraph_text'),
                            "is_paragraph_relevant": e.explanation,
                        }
                    )

            return relevant_paragraphs_map if relevant_paragraphs_map else None
        else:
            logger.info(f"[Web Scrapper] {len(paragraphs_map.items())} Paragraphs were retrieved from Pubmed Database.")
            return None
        
    def _retieval_pipeline(self, query_question:str, triplet:BiomedicalTriplet, triplet_report_folder:str=None) -> List[dict]:
        logger.info(f"[Triplet Retrieval Pipeline] Fetching vector database ...")
        # [STEP 1] >> Retrieve Chunks (Sentences) from Pubmed Vector Database.
        retrieved_docs:List[dict] = self.pubmed_fetcher.fetch(query_question, top_k=self.top_k)

        # [STEP 2] >> Get Parent Paragraphs for each Sentence.
        logger.info(f"[Triplet Retrieval Pipeline] Getting Parent Paragraphs ...")
        qdrant_paragraphs_map = self.parent_paragraph_fetcher.get_parent_paragraphs(retrieved_docs)
        assert qdrant_paragraphs_map, ValueError(f"Failed: No Parent Paragraphs were Retrieved from SQLite, although {len(retrieved_docs)} chunks where retrieved from Qdrant!")

        # [STEP 3] >> Evaluation of Retrieved Paragraphs.
        qdrant_evaluations = self.paragraph_evaluator.evaluate(query_question, qdrant_paragraphs_map).evaluations

        # [STEP 4] >> Extract Only 'STRONGLY_RELEVANT' Paragraphs from the retrieved pool.
        relevant_paragraphs_map:List[dict] = []
        for e in qdrant_evaluations:
            if e.evaluation in ['STRONGLY_RELEVANT'] and e.paragraph_number in qdrant_paragraphs_map:
                relevant_paragraphs_map.append(
                    {
                        "pmcid_or_doi": qdrant_paragraphs_map.get(e.paragraph_number).get('pmcid_or_doi'),
                        "paragraph_text": qdrant_paragraphs_map.get(e.paragraph_number).get('paragraph_text'),
                        "is_paragraph_relevant": e.explanation,
                    }
                )

        # [STEP 5] >> If No Relevant Paragraphs from Qdrant Vector Database were retrieved, Fetch Pubmed Online Database using Keywords.
        if not relevant_paragraphs_map:
            logger.info(f"[Triplet Retrieval Pipeline] No Relevant Chunks in Qdrant Databse.")
            relevant_paragraphs_map = self._fetch_pubmed_online(query_question, triplet, triplet_report_folder)

        return relevant_paragraphs_map if relevant_paragraphs_map else []

    def restart_ollama(self, log_file=".logs/ollama.log"):
            logger.info(f"[Ollama] starting / restarting Ollama server and log to {log_file}")
            # Make sure the log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Kill existing Ollama process
            system = platform.system()
            if system in ("Linux", "Darwin"):  # macOS/Linux
                subprocess.run(["pkill", "-f", "ollama"], check=False)
            elif system == "Windows":
                subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"], check=False)

            time.sleep(2)  # give it time to shut down

            # Open the log file for appending
            f = open(log_file, "a")

            # Restart Ollama and redirect logs
            subprocess.Popen(
                [os.path.expanduser("~/.local/bin/ollama"), "serve"],
                stdout=f,
                stderr=f,
                preexec_fn=None if system == "Windows" else os.setpgrp  # detach on Unix
            )

            logger.info(f"[Ollama] Ollama restarted, logs are in {log_file}")

    def _run_with_timeout(self, func, *args, timeout=300, triplet_folder:str="", **kwargs):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                logger.info(f"[Ollama] timed out after {timeout} seconds")
                self.restart_ollama()
                if os.path.exists(triplet_folder):
                    shutil.rmtree(triplet_folder)
                    logger.info(f"[Terminating] Deleted directory: {triplet_folder}")
                return None  # or raise, or return defaults
            
    def evaluate_and_fix_triplets(self, initial_triplets:List[BiomedicalTriplet], current_relations:set[str], only_validate:bool=False) -> List[tuple[Hop,str,int]]:
        
        logger.info(f"[Triplet Validation Team] Evaluating {len(initial_triplets)} Triplets ...")
        processed_triplets: List[tuple[BiomedicalTriplet, str, int]] = []

        # [STEP 1] >> Intial Hops Evaluation
        for ord_num, t in tqdm(enumerate(initial_triplets), desc=f"Evaluating {len(initial_triplets)} triplets in the Pathway", unit='Triplet'):
            evidences_obj:dict[str, str] = json.loads(t.evidences)
            triplet_question = f"Give me the paragraphs explaining the triplet: {t.head.name} (type: {t.head.entity_type}) {t.relation} {t.tail.name} (type: {t.tail.entity_type})"

            # [Retrieval Pipeline] >> Retrieve Paragraphs that are relevant to the Query Question (Triplet). >>> Output: Relevant Paragraphs and their Summaries.
            triplet_relevant_paragraphs = self._run_with_timeout(
                self._retieval_pipeline,
                query_question=triplet_question,
                triplet=t,
                timeout=600
            ) or [] # fallback if timeout
            
            different_evidences = [ev for ev in triplet_relevant_paragraphs if ev['pmcid_or_doi'] != evidences_obj['pmcid_or_doi']]
            if len(different_evidences) > 2:
                evidences_list = random.sample(triplet_relevant_paragraphs, 2)
            else:
                evidences_list = different_evidences
            evidences_list.append(evidences_obj)
            
            triplet_str = json.dumps(
                {
                    "head" : t.head.name,
                    "head_type" : t.head.entity_type,
                    "relation": t.relation,
                    "tail": t.tail.name,
                    "tail_type": t.tail.entity_type,
                }, indent=1, ensure_ascii=False
            )
            evidences_list_str = json.dumps(evidences_list, indent=2, ensure_ascii=False)

            t.evidences = evidences_list_str

            eval_response = self.evaluator.invoke(self._eval_prompt.invoke({"triplet": triplet_str, "evidence_paragraphs": t.evidences}))
            str_triplet_eval_format = eval_response.model_dump_json(indent=2)
            triplet_evaluation_obj = TripletEvaluation.model_validate_json(str_triplet_eval_format)

            if triplet_evaluation_obj.biological_validity == 'VALID' and triplet_evaluation_obj.semantic_coherence == 'COHERENT' and triplet_evaluation_obj.directionality == 'CORRECT':
                processed_triplets.append((t, f'valid: [{triplet_evaluation_obj.explanation}]', ord_num))
            else:
                ref_evaluation = triplet_evaluation_obj
                fixed_t = deepcopy(t)
                valid_flag = False
                trial_count = 1
                while not valid_flag:
                    # Break after 3 attempts
                    if trial_count >= 4:
                        # Fixing Hop failed - Can't be Fixed
                        processed_triplets.append((t, f'need-review: [{ref_evaluation.explanation}]', ord_num))
                        break
    
                    # Fix the Invalid Hop
                    fix_response = self.fixer.invoke(self._fix_prompt.invoke({
                        "evaluation_report":ref_evaluation.model_dump_json(indent=2),
                        "evidence_paragraphs":fixed_t.evidences,
                        "preferred_relations": str(current_relations)
                        })
                    )

                    fixed_t.relation = FixedRelation.model_validate_json(fix_response.model_dump_json()).proposed_relation.lower().replace(" ", "_")

                    fixed_triplet_str = json.dumps(
                        {
                            "head" : fixed_t.head.name,
                            "head_type" : fixed_t.head.entity_type,
                            "relation": fixed_t.relation,
                            "tail": fixed_t.tail.name,
                            "tail_type": fixed_t.tail.entity_type,
                        }, indent=1, ensure_ascii=False
                    )

                    # Re-Evaluate the Fixed Hop
                    re_eval_response = self.evaluator.invoke(self._eval_prompt.invoke({"triplet": fixed_triplet_str, "evidence_paragraphs": fixed_t.evidences}))
                    ref_evaluation = TripletEvaluation.model_validate_json(re_eval_response.model_dump_json())
                    
                    if ref_evaluation.biological_validity == 'VALID' and ref_evaluation.semantic_coherence == 'COHERENT' and ref_evaluation.directionality == 'CORRECT':
                        processed_triplets.append((fixed_t, f'valid: [{ref_evaluation.explanation}]', ord_num))
                        valid_flag = True
                    trial_count += 1
        
        return processed_triplets
    
    def invoke_evaluator(self, triplet_str:str, evidences:str) -> str:

        eval_response = self.evaluator.invoke(self._eval_prompt.invoke({"triplet": triplet_str, "evidence_paragraphs": evidences}))
        str_triplet_eval_format = eval_response.model_dump_json(indent=2)
        triplet_evaluation_obj = TripletEvaluation.model_validate_json(str_triplet_eval_format)

        if triplet_evaluation_obj.biological_validity == 'VALID' and triplet_evaluation_obj.semantic_coherence == 'COHERENT' and triplet_evaluation_obj.directionality == 'CORRECT':
            return f'valid: [{triplet_evaluation_obj.explanation}]'
        else:
            return f'need-review: [{triplet_evaluation_obj.explanation}]'
    
    def validate_triplet(self, triplet:BiomedicalTriplet) -> Dict:

        """
        Validate Triplet by Curator's Evidence and Extra Evidence from Corpora or Pubmed Web.
        
        Input:
            - triplet [BiomedicalTriplet]: Triplet Obj. to be validated.
            
        Output: Dictionary of Triplet, Report from extra evidence and curator evidence.
        """
        
        logger.info(f"[Triplet Validation Team] Evaluating Triplet: {triplet} ...")

        triplet_question = f"How does {triplet.head.name} (type: {triplet.head.entity_type}) affect {triplet.tail.name} (type: {triplet.tail.entity_type}) ?"

        triplet_str = json.dumps(
            {
                "head" : triplet.head.name,
                "head_type" : triplet.head.entity_type,
                "relation": triplet.relation,
                "tail": triplet.tail.name,
                "tail_type": triplet.tail.entity_type,
            }, indent=1, ensure_ascii=False
        )
        
        # [Retrieval Pipeline] >> Retrieve Paragraphs that are relevant to the Query Question (Triplet). >>> Output: Relevant Paragraphs and their Summaries.
        triplet_relevant_paragraphs = self._run_with_timeout(
            self._retieval_pipeline,
            query_question=triplet_question,
            triplet=triplet,
            timeout=600
        ) or [] # fallback if timeout
            
        if len(triplet_relevant_paragraphs) == 0:
            logger.info(f"[Seed Triplet Validation] No Extra Evidences Found.")
            triplet_vs_extra_evidence_report = None
        else:
            papers_ids = []
            unqiue_evidences = []
            for ev in triplet_relevant_paragraphs: # Get Evidences from different Citations
                if ev["pmcid_or_doi"] not in papers_ids:
                    unqiue_evidences.append(ev)
                    papers_ids.append(ev["pmcid_or_doi"])

            logger.info(f"Evidences for Validation = {len(unqiue_evidences)}")

            if len(unqiue_evidences) > 3: # Max evidences Count = 3
                evidences_list = random.sample(unqiue_evidences, 3)
            else:
                evidences_list = unqiue_evidences

            # Prepare Evidence and triplet strings. Invoke Evaluator.    
            evidences_list_str = json.dumps(evidences_list, indent=2, ensure_ascii=False)

            triplet.evidences = evidences_list_str

            triplet_vs_extra_evidence_report = self.invoke_evaluator(triplet_str=triplet_str, evidences=triplet.evidences)

        triplet_vs_curator_evidence_report = self.invoke_evaluator(triplet_str=triplet_str, evidences=triplet.curator_evidences)

        return {
            "triplet" : triplet,
            "vs_ExEvidence_report" : triplet_vs_extra_evidence_report,
            "vs_CurEvidence_report" : triplet_vs_curator_evidence_report
        }


class Vectorizer:
    def __init__(self, dense_m: str, sparse_m: str, batch_size: int = 32, device: Optional[str] = None):
        self.batch_size = batch_size
        self.set_embedding_models(dense_m, sparse_m, device)

    def set_embedding_models(self, dense_m: str, sparse_m: str, device: Optional[str] = None) -> None:
        # reset model names
        self.dense_model_name = dense_m
        self.sparse_model_name = sparse_m

        # device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # dense model
        self._dense_model: SentenceTransformer = SentenceTransformer(self.dense_model_name, device=self.device, trust_remote_code=True)

        # sparse model
        self._sparse_model: SparseEncoder = SparseEncoder(self.sparse_model_name, device=self.device, trust_remote_code=True)

    def encode(
        self, docs: List[str], encoder_type: str = "both"
    ) -> Tuple[Optional[torch.Tensor], Optional[List[dict]]]:
        if encoder_type not in ["dense", "sparse", "both"]:
            raise ValueError(f"encoder_type={encoder_type}. Expected: dense | sparse | both")

        dense_embeddings, sparse_embeddings = None, None

        # dense encoding
        if encoder_type in ["dense", "both"]:
            dense_embeddings = self._dense_model.encode(
                docs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )

        # sparse encoding
        if encoder_type in ["sparse", "both"]:
            sparse_embeddings = self._sparse_model.encode(
                docs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

        return dense_embeddings, sparse_embeddings


class HopAligner():

    def __init__(self, model: str, temperature: float=0.8, ollama_port:int=11434, **kwargs):
        # Initiate Aligner
        self.aligner = ChatOllama(model=model, temperature=temperature, num_ctx=30000, timeout=600, base_url=f"http://localhost:{ollama_port}", **kwargs).with_structured_output(AlignedHop)
        
        # Set System and Human Messages for Hop Evaluation
        _align_system_message = """You are the Biomedical KG Schema Alignment Agent. Your task is to align new hops (head, relation, tail) to an existing biomedical knowledge graph schema by mapping entity types to high-level classes and mapping relations to mechanistic/causal, polarity-aware relations, with the help of a guiding schema. Your goal is analytical usability. Favor broad, stable types and a small set of functional relations that capture direction (increase vs decrease) or activation vs inhibition.

# Inputs you will receive:
    - Hop object with fields: head, head_type, tail, tail_type, relation, evidence_paragraph_number, explanation
    - Evidence paragraph text (full paragraph string)
    - Current KG types (array of type labels)
    - Current KG relations (array of relation labels)

# Output you must produce:
    - A single JSON object with this exact schema: {{ "head": "", "proposed_head_type": "", "proposed_relation": "", "tail": "", "proposed_tail_type": "", "explanation": "" }}
    - Do not change the head or tail strings. Do not add fields. Do not return arrays or multiple objects.

# Guiding Schema:
    #### Use the following schema as a guidance for what global biomedical entity types, and relations between them, may be used for mapping and interpretting the triplets. The Schema is Prefered, but not mandatory.
    #### increases, decreases, and causes_no_change, cover all aspects of effect, including levels, activities, functionalities, expressions, ... etc. The goal is to represent the general causal effect, not the detailed biological effect.
    #### Gene to Protein relations:
        - (Gene)-[increases]->(Protein)
        - (Gene)-[decreases]->(Protein)
        - (Gene)-[causes_no_change]->(Protein)
    #### Gene to Biological Process relations :
        - (Gene)-[increases]->(BiologicalProcess)
        - (Gene)-[decreases]->(BiologicalProcess)
        - (Gene)-[causes_no_change]->(BiologicalProcess)
    #### Protein to Biological Process relations :
        - (Protein)-[increases]->(BiologicalProcess)
        - (Protein)-[decreases]->(BiologicalProcess)
        - (Protein)-[causes_no_change]->(BiologicalProcess)
    #### Drug to Biological Process relations :
        - (Drug)-[increases]->(BiologicalProcess)
        - (Drug)-[decreases]->(BiologicalProcess)
        - (Drug)-[causes_no_change]->(BiologicalProcess)
    #### Chemical to Biological Process relations :
        - (Chemical)-[increases]->(BiologicalProcess)
        - (Chemical)-[decreases]->(BiologicalProcess)
        - (Chemical)-[causes_no_change]->(BiologicalProcess)
    #### Drug to Protein relations :
        - (Drug)-[increases]->(Protein)
        - (Drug)-[decreases]->(Protein)
        - (Drug)-[causes_no_change]->(Protein)
    #### Chemical to Protein relations :
        - (Chemical)-[increases]->(Protein)
        - (Chemical)-[decreases]->(Protein)
        - (Chemical)-[causes_no_change]->(Protein)
    #### Gene to Disease relations :
        - (Gene)-[increases]->(Disease)
        - (Gene)-[decreases]->(Disease)
        - (Gene)-[causes_no_change]->(Disease)
    #### Protein to Disease relations :
        - (Protein)-[increases]->(Disease)
        - (Protein)-[decreases]->(Disease)
        - (Protein)-[causes_no_change]->(Disease)
    #### Drug to Disease relations :
        - (Drug)-[increases]->(Disease)
        - (Drug)-[decreases]->(Disease)
        - (Drug)-[causes_no_change]->(Disease)
    #### Chemical to Disease relations :
        - (Chemical)-[increases]->(Disease)
        - (Chemical)-[decreases]->(Disease)
        - (Chemical)-[causes_no_change]->(Disease)

# Core principles:
    ## The Goal of the task is to try to map the original types and relations to the existing types and relations. If you can't map the term with a high confidence, always return the original term as is. 
    ## Type alignment (be global, not specific):
        * Prefer broad classes from the provided KG types. Examples of broad mappings:
            - Protein complex, receptor subunit, cytokine → Protein (if Protein exists)
            - enzyme, kinase, Hormone → Protein
            - microRNA, lncRNA, siRNA, mRNA, transcript → RNA
            - Chemicals, metabolites, vitamins, small molecules, ligands, drugs → Abundance
            - Pathway, signaling, cellular process, activity, phosphorylation event → Biological Process
            - Disorder, syndrome, condition, disease → Pathology
            - Tissue, organ, anatomical site → Anatomy
            - Cell line, cell type → Cell
            - SNP, mutation, variant → Genetic Variant
        * If the exact or a close broad parent is missing from the KG types list, keep the original type, and it will be added as a new term.
    ## Relation alignment (be mechanistic/causal and polarity-aware):
            * Prefer relations that express functional effect and direction. For example:
                - Upregulates, promotes, induces, triggers, enhances, sensitizes → increases (for levels/abundance/expression/activity/receptor/process)
                - Downregulates, suppresses, inhibits, blocks, attenuates, represses, impairs, confers resistance → decreases (for levels/abundance/expression/activity/receptor/process)
            * Associated with, linked to, correlates with, is a hallmark of, causes, leads to, contributes in, contributes to:
                - If the paragraph shows “higher X with higher Y” or “positively associated,” map to Positive Correlation
                - If “higher X with lower Y,” map to Negative Correlation
                - Use correlation only when mechanistic direction (increase/decrease) cannot be established
            * Phosphorylates, dephosphorylates: keep if present in KG; otherwise map to activates/inhibits depending on the net effect described
            * Binds to: keep if present in KG and no functional effect is stated; if the evidence states binding causes activation/inhibition, prefer the causal relation (increases/decreases)
            * Choose from the provided KG relations list. If your ideal canonical relation isn't in the list, select the closest available mechanistic/causal option. If none fits, keep the original relation.
            * Focus on functionality: Determine whether the evidence indicates an increase vs decrease of a level/expression/activity/process.
    ## Biological Validity: When in doubt, keep the original term as it is, because we prefer accuracy over simplicity.

# Examples of Directionality and polarity cues:
    * Positive (increase/activate) cues: increase(s), higher, elevate(s), enhance(s), potentiate(s), promote(s), induce(s), trigger(s), stimulate(s), augment(s), upregulate(s), required for activation, leads to activation
    * Negative (decrease/inhibit) cues: decrease(s), lower, reduce(s), diminish(es), attenuate(s), suppress(es), inhibit(s), block(s), downregulate(s), impair(s), knockdown/knockout reduces, prevents
    * Correlation cues: associated with, correlated with, linked to, positively/negatively associated, risk increases/decreases, odds ratio

# Procedure:
    1- Read the hop and the evidence paragraph. Preserve head and tail exactly as given.
    2- Align head_type and tail_type:
        - Map each to a broad parent in the provided KG types list using the mapping rules above.
        - If the best parent class is absent, pick the closest broader class available; otherwise keep the original type.
        - Determine functional direction: From the evidence paragraph, extract whether the head increases/decreases the tail's levels/abundance/expression/activity/receptor/process. If only association is reported, map to Positive/Negative Correlation.
    3- Align the relation:
        - Map the original relation to the closest mechanistic/causal relation in the provided KG relations list that captures the functional direction (increases/decreases preferred).
        - If the exact best choice or a close choice isn't present, keep the original relation.
    4- Output the single JSON object with the required fields and a brief explanation referencing key evidence phrases.

# Hard constraints:
    - Do not invent new entities. Do not change head/tail strings.
    - Prefer a small, stable set of broad types and functional relations.
    - If uncertain, favor broader types and simple, causal relations over niche labels.
    - Never return empty strings as proposed types or relations. If you can't find a suitable option from the given types and relations sets, return the same relation and entity types as in the original hop, as they will be added to the current sets as new types and relations.
    - Return only one JSON object; no extra commentary.

# Few-shot examples
    ## Example 1 Evidence paragraph: “NF-κB (p65/p50) robustly upregulates IL6 transcription upon stimulation, leading to increased IL6 expression.”
        - Current KG types: ["Protein","RNA","Gene","Abundance","Disease","Biological Process","Cell","Anatomy","Genetic Variant"]
        - Current KG relations: ["increases","decreases","binds","phosphorylates","dephosphorylates","positive correlation","negative correlation","regulates"]
        - Input hop (JSON):
            json
            {{
            "head": "NF-κB p65/p50 complex",
            "head_type": "Protein complex",
            "tail": "IL6",
            "tail_type": "Cytokine",
            "relation": "promotes transcription of",
            "evidence_paragraph_number": 3,
            "explanation": "NF-κB induces IL6 expression"
            }}
        - Output:
            json
            {{
            "head": "NF-κB p65/p50 complex",
            "proposed_head_type": "Protein",
            "proposed_relation": "increases",
            "tail": "IL6",
            "proposed_tail_type": "Protein",
            "explanation": "‘upregulates … increased IL6 expression' indicates NF-κB increases IL6 levels; mapped complex→Protein and cytokine→Protein to keep types global."
            }}

    ## Example 2 Evidence paragraph: “miR-21 represses PTEN; knockdown of miR-21 increases PTEN protein levels.”
        - Current KG types: ["Protein","RNA","Gene","Abundance","Disease","Biological Process","Cell","Anatomy","Genetic Variant"]
        - Current KG relations: ["increases","decreases","binds","phosphorylates","dephosphorylates","positive correlation","negative correlation","regulates"]
        - Input hop (JSON):
            json
            {{
            "head": "miR-21",
            "head_type": "microRNA",
            "tail": "PTEN",
            "tail_type": "Tumor suppressor",
            "relation": "suppresses",
            "evidence_paragraph_number": 5,
            "explanation": "miR-21 downregulates PTEN"
            }}
        - Output:
            json
            {{
            "head": "miR-21",
            "proposed_head_type": "RNA",
            "proposed_relation": "decreases",
            "tail": "PTEN",
            "proposed_tail_type": "Gene",
            "explanation": "Represses/knockdown evidence shows miR-21 decreases PTEN; mapped microRNA→RNA and tumor suppressor→Gene (broad parent)."
            }}

    ## Example 3 Evidence paragraph: “Higher serum vitamin D levels were associated with lower inflammatory markers.”
        - Current KG types: ["Protein","RNA","Gene","Abundance","Disease","Biological Process","Cell","Anatomy"]
        - Current KG relations: ["increases","decreases","binds","phosphorylates","dephosphorylates","positive correlation","negative correlation","regulates"]
        - Input hop (JSON):
            json
            {{
            "head": "Vitamin D",
            "head_type": "Vitamin",
            "tail": "Inflammation",
            "tail_type": "Process",
            "relation": "associated with",
            "evidence_paragraph_number": 2,
            "explanation": "Inverse association reported"
            }}
        - Output:
            json
            {{
            "head": "Vitamin D",
            "proposed_head_type": "Abundance",
            "proposed_relation": "negative correlation",
            "tail": "Inflammation",
            "proposed_tail_type": "Biological Process",
            "explanation": "‘Higher … associated with lower' indicates a negative correlation; vitamin→Abundance, process→Biological Process."
            }}

    # Example 4 Evidence paragraph: “IL-1β stimulation leads to activation of p38 MAPK.”
        - Current KG types: ["Protein","RNA","Gene","Abundance","Disease","Biological Process","Cell","Anatomy"]
        - Current KG relations: ["increases","decreases","binds","phosphorylates","dephosphorylates","positive correlation","negative correlation","regulates"]
        - Input hop (JSON):
            json
            {{
            "head": "IL-1β",
            "head_type": "Cytokine",
            "tail": "p38 MAPK",
            "tail_type": "Kinase",
            "relation": "induces activation of",
            "evidence_paragraph_number": 7,
            "explanation": "IL-1β activates p38"
            }}
        - Output:
            json
            {{
            "head": "IL-1β",
            "proposed_head_type": "Protein",
            "proposed_relation": "increases",
            "tail": "p38 MAPK",
            "proposed_tail_type": "Protein",
            "explanation": "‘Leads to activation' → activates; cytokine/kinase mapped to Protein to keep types global."
            }}
    
    # Example 5 Evidence paragraph: “TNF-α levels were positively correlated with insulin resistance.”
        - Current KG types: ["Protein","RNA","Gene","Abundance","Disease","Biological Process","Cell","Anatomy"]
        - Current KG relations: ["increases","decreases","binds","phosphorylates","dephosphorylates","positive correlation","negative correlation","regulates"]
        - Input hop (JSON):
            json
            {{
            "head": "TNF-α",
            "head_type": "Cytokine",
            "tail": "Insulin resistance",
            "tail_type": "Disease/phenotype",
            "relation": "associated with",
            "evidence_paragraph_number": 11,
            "explanation": "Positive association"
            }}
        - Output:
            json
            {{
            "head": "TNF-α",
            "proposed_head_type": "Protein",
            "proposed_relation": "positive correlation",
            "tail": "Insulin resistance",
            "proposed_tail_type": "Disease",
            "explanation": "‘Positively correlated' → positive correlation; cytokine→Protein, disease/phenotype→Disease."
            }}
"""

        _align_human_message = """You will align one hop to the current KG schema with a functional, mechanistic focus. Use the system guidance above. Choose types and relations only from the provided lists when possible. If a perfect match is missing, pick the closest broader type or nearest mechanistic relation. If directionality is unclear, use Positive/Negative Correlation only when warranted by the text; otherwise keep the original relation.

- Original Hop:
{original_hop}

- Evidence Paragraph:
{evidence_paragraph}

- Current KG Types:
{current_kg_types}

- Current KG Relations:
{current_kg_relations}
"""
        
        self._align_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_align_system_message),
            HumanMessagePromptTemplate.from_template(_align_human_message)
        ])
    
    def _print_aligned_hops(self, printable_hop_align_format:str) -> None:
        logger.info('================================= [Start] Aligned Hops =================================')
        logger.info(printable_hop_align_format)
        logger.info('================================== [End] Aligned Hops ==================================')
    
    def _export_aligned_hops(self, printable_align_hops_format, output_folder:str) -> None:
        
        # ✅ Store align_hops
        align_hops_file = os.path.join(output_folder, "aligned_hops.txt")
        with open(align_hops_file, "w") as f:
            f.write(printable_align_hops_format)

    def align_hops(self, hops_to_be_aligned:List[Hop], relevant_paragraphs_dict:dict[int,dict], current_kg_types:set[str], current_kg_relations:set[str], print_output:bool=False, output_folder:str=None) -> List[Hop]:
        
        """Align Generated hops to Seed KG Schema (Node Types and Relationship Types)"""

        aligned_hops:List[Hop] = []
        aligned_hops_str:List[str] = []
        for hop in tqdm(hops_to_be_aligned, desc="Align Hops to Seed KG Schema ...", unit="Hop"):
            hop_str = hop.model_dump_json(indent=2)
            if hop.evidence_paragraph_number in relevant_paragraphs_dict:
                evidence_text = relevant_paragraphs_dict.get(hop.evidence_paragraph_number).get("paragraph_text")

                response = self.aligner.invoke(self._align_prompt.invoke({
                    "original_hop": hop_str,
                    "evidence_paragraph": evidence_text,
                    "current_kg_types": str(current_kg_types),
                    "current_kg_relations": str(current_kg_relations),
                }))

                response_obj = AlignedHop.model_validate_json(response.model_dump_json())

                aligned_hop = Hop(
                    head=hop.head,
                    head_type=response_obj.proposed_head_type.lower().replace(" ", "_") if response_obj.proposed_head_type not in ["", "None", "null", None] else hop.head_type.lower().replace(" ", "_"),
                    relation=response_obj.proposed_relation.lower().replace(" ", "_") if response_obj.proposed_relation not in ["", "None", "null", None] else hop.relation.lower().replace(" ", "_"),
                    tail=hop.tail,
                    tail_type=response_obj.proposed_tail_type.lower().replace(" ", "_") if response_obj.proposed_tail_type not in ["", "None", "null", None] else hop.tail_type.lower().replace(" ", "_"),
                    evidence_paragraph_number=hop.evidence_paragraph_number,
                    explanation=hop.explanation,
                )

                aligned_hops_str.append(aligned_hop.model_dump_json(indent=2))
                aligned_hops.append(aligned_hop)

                # update current types and relations
                current_kg_types.add(aligned_hop.head_type)
                current_kg_types.add(aligned_hop.tail_type)
                current_kg_relations.add(aligned_hop.relation)
            else:
                logger.info(f"[Schema Alignment] Hop {hop_str} lacks valid evidence number. skipping it ...")
                continue

        # prepare str representation for exporting and printing
        printable_aligned_hops = json.dumps(aligned_hops_str, indent=2, ensure_ascii=False)
        
        # Export if output folder is provided
        if print_output:
            self._print_aligned_hops(printable_aligned_hops)
        if output_folder:
            self._export_aligned_hops(printable_aligned_hops, output_folder)

        if len(aligned_hops) == len(hops_to_be_aligned):
            return aligned_hops
        else:
            raise ValueError(f"[Schema Alignment] Aligned Hops are {len(aligned_hops)}, where to-be-aligned hops are {len(hops_to_be_aligned)}. break !")
        

class EntityMatcherEvaluator():

    def __init__(self, model: str, temperature: float=0.8, ollama_port:int=11434, **kwargs):
        # Initiate Aligner
        self.match_checker = ChatOllama(model=model, temperature=temperature, num_ctx=30000, timeout=600, base_url=f"http://localhost:{ollama_port}", **kwargs).with_structured_output(EntityMatchingOutput)
        
        # Set System and Human Messages for Hop Evaluation
        _checker_system_message = """You are the Biomedical Knowledge Graph (KG) Entity Matching Expert.
# Task
    * Decide whether a newly Extracted Biomedical Entity (from a reference paragraph) refers to the same real-world concept as a given seed biomedical entity.
    * Be abstract yet biologically valid. Do not be overly strict: tolerate type-label differences when both entities map to the same broad, stable class.
    * Critical rule about the reference paragraph
    * The reference paragraph provides context for interpreting the extracted entity only. It will not mention the seed entity.
    * Use the paragraph to disambiguate and interpret the extracted entity (e.g., species, isoform, alias expansion, family vs member), not to seek co-mention with the seed or to justify a match by co-occurrence. If the paragraph doesn't help, use your own knowledge.

# Inputs
    * Extracted entity: name string and source type label.
    * Seed entity: name string and source type label.
    * Reference paragraph text (context for the extracted entity only).

# Output
    * Return exactly one JSON object with:
        - is_match: boolean
        - explanation: brief (1–3 sentences) rationale citing key interpretive cues about the extracted entity (from the paragraph or well-known domain synonyms), and the rule(s) used to decide.

# Core matching principles
    * Identity over surface form
    * Match if the names are the same entity via synonyms/aliases, brand vs generic, common abbreviations, Greek letter variants (alpha vs α), punctuation/case/hyphenation differences.
    * You may use well-known domain synonymy (e.g., aspirin = acetylsalicylic acid) even if not stated in the paragraph.
    * Tolerant type alignment (abstract but meaningful)
    * Accept matches when the only difference is a narrower label vs a broad parent class; examples of broad mappings:
        - Chemical, drug, metabolite, vitamin, small molecule → Abundance/Chemical (match if they denote the same molecule).
        - Cytokine, enzyme, kinase, receptor, hormone, transcription factor, protein complex (as a named entity) → Protein (match if they denote the same protein entity).
        - microRNA/lncRNA/siRNA/mRNA/transcript → RNA.
        - Pathway/signaling/cellular process/activity/event → Biological Process.
        - Disorder/syndrome/phenotype → Disease/Pathology.
        - Tissue/organ/anatomical site → Anatomy.
        - Cell type/cell line → Cell.
        - SNP/mutation/variant → Genetic Variant.
        - Do not conflate distinct top-level classes
        - Do not match Gene ↔ Protein or Gene ↔ RNA.
        - Do not match Disease/Pathology ↔ Biological Process.
        - Do not match Protein complex ↔ individual subunit.
        - Do not match protein family/group ↔ a specific member unless the paragraph explicitly resolves the extracted entity to that member.
        - Do not match ligand ↔ receptor, drug ↔ target, variant ↔ wild-type.

# Granularity and specificity
    * Family vs specific member: non-match unless the paragraph unambiguously constrains the extracted entity to that member (e.g., “p38α/MAPK14”).
    * Isoforms/PTM states vs canonical: non-match unless the paragraph indicates the extracted entity is used interchangeably with the canonical entity in that context and the KG collapses them.
    * Salts, hydrates, stereoisomers, prodrugs vs parent compound: match only if the paragraph or domain knowledge indicates they are treated as the same active entity.

# Species and context
    * Use species/context from the paragraph to interpret the extracted entity. Avoid matching across clearly different species unless the KG is species-agnostic and the paragraph equates them.
    * Evidence use (strict)
    * Use the paragraph ONLY to interpret and disambiguate the extracted entity (aliases, expansions, species, isoform, family/member).
    * Do NOT expect or search for the seed in the paragraph. Co-mention is not required and should not be used for matching.
    * If identity is still ambiguous after interpretation and application of the rules, prefer non-match.
    * Strict constraints
    * Do not invent or alter entity names.
    * Do not rely solely on string similarity when biological identity is uncertain.
    * If uncertain after applying the above rules, return non-match and state the ambiguity.
    * Output exactly one JSON object with the required keys and no extras.

# Procedure
    1- Normalize the extracted and seed names (case, hyphens, Greek letters). Check exact/near-exact equivalence.
    2- Interpret the extracted entity using the paragraph (aliases, expansions, species, isoform, family/member, context). Do not look for the seed in the paragraph.
    3- Map both entities’ types to broad parent classes. Tolerate label differences if both denote the same entity under a broad class.
    4- Apply non-conflation rules (gene–protein, family–member, complex–subunit, ligand–receptor, variant–wild-type).
    5- Decide:
        - High-confidence same real-world entity → is_match: true.
        - Ambiguous or clearly different → is_match: false.
    6- Write a concise explanation referencing how the paragraph interprets the extracted entity and/or well-known synonymy, plus the key rule(s) leading to the decision.

# Quality checklist (must pass)
    - Same real-world entity after interpreting the extracted entity? Yes/No.
    - Type labels compatible under broad mapping, without prohibited conflations?
    - Family/member, complex/subunit, isoform/canonical ambiguities resolved? If not, non-match.
    - Explanation is brief, references interpretive cues for the extracted entity, and cites the decisive rule(s).
    - Output is exactly one JSON object with is_match and explanation.
"""

        _checker_human_message = """You will check whether the following newly extracted biomedical entity match the mentioned seed biomedical entity:

- Extracted Entity:
{extracted_entity}

- Seed Entity
{seed_entity}

- Reference Paragraph:
{reference_paragraph}
"""
        
        self._check_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_checker_system_message),
            HumanMessagePromptTemplate.from_template(_checker_human_message)
        ])

    def check_match(self, extracted_entity:str, seed_entity:str, ref_paragraph:str) -> EntityMatchingOutput:

        """Check whether the Match between Seed Node and Generated Node is valid."""
        response = self.match_checker.invoke(self._check_prompt.invoke({
            "extracted_entity": extracted_entity,
            "seed_entity": seed_entity,
            "reference_paragraph": ref_paragraph
        }))

        response_obj = EntityMatchingOutput.model_validate_json(response.model_dump_json())

        return response_obj