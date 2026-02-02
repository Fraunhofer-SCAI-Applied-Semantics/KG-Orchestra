import ast
from datetime import datetime, time
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import signal
import traceback
import numpy as np
random.seed(42)  # pick any integer
import re
from time import perf_counter
from typing import Dict, List
import pandas as pd
import torch
from tqdm import tqdm
from kg_orchestra.modules.biomedical_models import BiomedicalEntity, BiomedicalTriplet
from kg_orchestra.modules.clients import EntityHarmonizer
from neo4j import GraphDatabase
from kg_orchestra.modules.agents import ParagraphEvaluator, HopValidationTeam, PathwayBuilder, HopAligner, EntityMatcherEvaluator,restart_ollama
from kg_orchestra.modules.clients import EntityHarmonizer, PubmedFetcher, ParentParagraphFetcher
from kg_orchestra.modules.output_models import Pathway, Hop, AlignedHop, EntityMatchingOutput
from kg_orchestra.modules.pubmed import ArticlePipeline
import subprocess
import time
import platform
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

        
class SeedKG(GraphDatabase):
    def __init__(self, seed_csv_path:str=None, enriched_csv_path:str=None, url:str=None, auth:tuple[str, str]=None, entity_harmonizer:EntityHarmonizer=None, llm_model:str=None, database:str=None, allowed_relations:list[str]=None, allowed_types:list[str]=None):
        
        if seed_csv_path != None:
            logger.info(f"[Seed KG] Seed KG will be loaded from CSV file.")
            self._driver = None
            self._session = None
        elif url != None and auth != None and database != None:
            logger.info(f"[Seed KG] Seed KG will be loaded from Neo4j Port.")
            self._driver = self.driver(uri=url, auth=auth)
            self._session = self._driver.session(database=database)
        else:
            raise ValueError(f"[Seed KG] No Seed KG file/port deteted.")

        self._allowed_relations = allowed_relations
        self._allowed_types = allowed_types
        self._entity_harmonizer: EntityHarmonizer = entity_harmonizer
        self._llm_model:str = llm_model
        self._seed_csv_file_path = seed_csv_path
        self._enriched_csv_file_path = enriched_csv_path
        self._current_triplets_df = None

        # Queries
        self._seed_kg_init_query:str = """
MATCH (n)-[r]->(m)
WHERE 
  r.evidence IS NOT NULL AND
  r.llm_validated is NULL
RETURN 
  n.name AS head, 
  labels(n) AS head_type,
  n.mapped_umls_cui AS head_cui,
  n.mapped_umls_synonym AS head_syn,
  type(r) AS relation, 
  m.name AS tail, 
  labels(m) AS tail_type,
  m.mapped_umls_cui AS tail_cui,
  m.mapped_umls_synonym AS tail_syn,
  properties(r) AS r_properties, 
  elementId(n) AS head_id, 
  elementId(m) AS tail_id
ORDER BY head ASC 
"""
        
        self._get_seed_nodes_with_cui_query:str = """MATCH (n)
        WHERE n.mapped_umls_cui is not null
        RETURN n.name as node_name, labels(n) as node_type, elementId(n) as node_id, n.mapped_umls_cui as node_cui, n.mapped_umls_synonym as node_syn"""

        self._get_seed_nodes_with_no_cui_query:str = """MATCH (n)
        WHERE n.mapped_umls_cui is null
        RETURN n.name as node_name, labels(n) as node_type, elementId(n) as node_id"""

        # Internal Variables
        self.seed_triplets:list[BiomedicalTriplet] = []
        self.triplets:list[BiomedicalTriplet] = []
        self.existing_nodes:List[tuple[str,BiomedicalEntity]] = []
        self._current_entity_types:set = {}
        self._current_relation_types:set = {}
        self.kg_name = database.replace("0","_")

    def set_seed_kg_query(self, query:str):
        self._seed_kg_init_query = query
        return
    
    def current_entity_types(self) -> set[str]:
        return self._current_entity_types
    
    def current_relation_types(self) -> set[str]:
        return self._current_relation_types
    
    def _fetch_nodes_and_triplets_from_seed_csv(self):

        if not os.path.isfile(self._seed_csv_file_path.replace(".csv", "_enriched.csv")):
            logger.info(f"Prepare Enriched CSV File ...")
            seed_df = pd.read_csv(self._seed_csv_file_path)
            seed_df["head_node_mapped_umls_cui"] = "no_value"
            seed_df["head_node_mapped_umls_synonyms"] = "no_value"
            seed_df["tail_node_mapped_umls_cui"] = "no_value"
            seed_df["tail_node_mapped_umls_synonyms"] = "no_value"
            seed_df["extra_evidence"] = "no_value"
            seed_df["llm_generated_rel"] = False
            seed_df["llm_generated_head"] = False
            seed_df["llm_generated_tail"] = False
            seed_df["llm_model"] = self._llm_model
            seed_df["validation_by_extra_evidence"] = "no_value"
            seed_df["llm_validated"] = "no_value"
            seed_df["validation_by_curator_evidence"] = "no_value"

            self._enriched_csv_file_path = self._seed_csv_file_path.replace(".csv", "_enriched.csv")
            seed_df.to_csv(self._enriched_csv_file_path, index=False)
        else:
            self._enriched_csv_file_path = self._seed_csv_file_path.replace(".csv", "_enriched.csv")
            logger.info(f"Read Enriched CSV file ...")

        current_existing_nodes: List[tuple[str, BiomedicalEntity]] = []
        current_triplets: List[BiomedicalTriplet] = []
    
        self._current_triplets_df = pd.read_csv(self._enriched_csv_file_path)
        self._seed_triplets_df = self._current_triplets_df[self._current_triplets_df["llm_generated_rel"] == False]
        self._current_triplets_df.fillna("no_value", inplace=True)


        for idx, triplet in tqdm(self._current_triplets_df.iterrows(), desc="Loading Current Triplets from CSV", unit="Triplet"):
            # --- HEAD NODE ---
            if not triplet['head_node_mapped_umls_cui'] or triplet['head_node_mapped_umls_cui'] == "no_value" or pd.isna(triplet['head_node_mapped_umls_cui']):
                # logger.info(f"Map Head Entity to UMLS Concept.")
                head_entity_inferred = self._entity_harmonizer.fetch(
                    f"{triplet['head_node_name']} {triplet['head_node_type']}", 1
                )[0]
                self._current_triplets_df.at[idx, 'head_node_mapped_umls_cui'] = head_entity_inferred.get('cui')
                self._current_triplets_df.at[idx, 'head_node_mapped_umls_synonyms'] = head_entity_inferred.get('synonyms')

            head_cui = self._current_triplets_df.at[idx, 'head_node_mapped_umls_cui']
            head_syn = self._current_triplets_df.at[idx, 'head_node_mapped_umls_synonyms']

            # logger.info(f"{triplet['head_node_name']} {triplet['head_node_type']} >>> {head_syn} of CUI: {head_cui}")
            head_entity = BiomedicalEntity(
                    name=triplet["head_node_name"],
                    entity_id=triplet["head_node_id"],
                    entity_type=triplet['head_node_type'],
                    mapped_umls_cui=head_cui,
                    mapped_synonym=head_syn,
                )

            current_existing_nodes.append(
                (head_cui, head_entity)
            )

            # --- TAIL NODE ---
            if not triplet['tail_node_mapped_umls_cui'] or triplet['tail_node_mapped_umls_cui'] == "no_value" or pd.isna(triplet['tail_node_mapped_umls_cui']):
                # logger.info(f"Map Tail Entity to UMLS Concept.")
                tail_entity_inferred = self._entity_harmonizer.fetch(
                    f"{triplet['tail_node_name']} {triplet['tail_node_type']}", 1
                )[0]
                self._current_triplets_df.at[idx, 'tail_node_mapped_umls_cui'] = tail_entity_inferred.get('cui')
                self._current_triplets_df.at[idx, 'tail_node_mapped_umls_synonyms'] = tail_entity_inferred.get('synonyms')

            tail_cui = self._current_triplets_df.at[idx, 'tail_node_mapped_umls_cui']
            tail_syn = self._current_triplets_df.at[idx, 'tail_node_mapped_umls_synonyms']
            # logger.info(f"{triplet['tail_node_name']} {triplet['tail_node_type']} >>> {tail_syn} of CUI: {tail_cui}")
            
            tail_entity = BiomedicalEntity(
                    name=triplet["tail_node_name"],
                    entity_id=triplet["tail_node_id"],
                    entity_type=triplet['tail_node_type'],
                    mapped_umls_cui=tail_cui,
                    mapped_synonym=tail_syn,
                )

            current_existing_nodes.append(
                (tail_cui, tail_entity)
            )

            # Create evidences dictionary
            curator_evidence = {
                "pmid": triplet["curator_evidence_pmid"],
                "evidence": triplet["curator_evidence_text"]
            }

            # Create a BiomedicalTriplet instance
            triplet_obj = BiomedicalTriplet(
                head=head_entity,
                tail=tail_entity,
                relation=triplet["relation_type"],
                curator_evidences=json.dumps(curator_evidence, indent=2, ensure_ascii=False),
                evidences=None if pd.isna(triplet["extra_evidence"]) else triplet["extra_evidence"],
                llm_generated=triplet["llm_generated_rel"]
            )

            current_triplets.append(triplet_obj)

        self.existing_nodes = current_existing_nodes

        # --- Collect unique relation and entity types ---
        self._current_relation_types = set(self._current_triplets_df['relation_type'].dropna().unique())
        self._current_entity_types = set(
            self._current_triplets_df['head_node_type'].dropna().unique()
        ).union(
            set(self._current_triplets_df['tail_node_type'].dropna().unique())
        )

        if len(self.seed_triplets) == 0:
            self.seed_triplets = deepcopy([triplet for triplet in current_triplets if triplet.llm_generated==False])
        self.triplets = deepcopy(current_triplets)

        logger.info(f"Current Triplets: {len(self.triplets)}")
        logger.info(f"Seed Triplets: {len(self.seed_triplets)}")
        return

    def _load_seed_nodes_from_neo4j(self):
        # Return Nodes with CUI
        nodes_with_cui = self._session.run(
            self._get_seed_nodes_with_cui_query
        )
        current_existing_nodes: List[tuple[str,BiomedicalEntity]] = [
            (
                record['node_cui'],
                BiomedicalEntity(
                    name=str(record["node_name"]),
                    entity_id=str(record["node_id"]),
                    entity_type=str([label for label in record['node_type']][0]),
                    mapped_umls_cui = str(record['node_cui']),
                    mapped_synonym = str(record['node_syn']),
                )
            ) for record in list(nodes_with_cui)]
        
        logger.info(f"Nodes with CUI = {len(current_existing_nodes)}")
        
        # Harmonize Nodes with no CUI
        nodes_with_no_cui = self._session.run(
            self._get_seed_nodes_with_no_cui_query
        )
        records = list(nodes_with_no_cui)
        for record in tqdm(records, desc=f"Harmonizing Seed Nodes . . .", unit="Node"):
            # Create BiomedicalEntity instances for head and tail entities and add them to existing nodes
            entity = BiomedicalEntity(
                name=str(record["node_name"]),
                entity_id=str(record["node_id"]),
                entity_type=str([label for label in record['node_type']][0])
            )

            entity_inferred = self._entity_harmonizer.fetch(f"{entity.name} {entity.entity_type}", 1)[0]
            entity.mapped_umls_cui = entity_inferred.get('cui')
            entity.mapped_synonym = entity_inferred.get('synonyms')
            current_existing_nodes.append((entity.mapped_umls_cui, entity))
            self._session.run(
                """
                MATCH (n)
                WHERE elementId(n) = $entity_id
                SET n.mapped_umls_cui = $cui
                SET n.mapped_umls_synonym = $synonym
                """,
                entity_id=entity.entity_id,
                cui=entity.mapped_umls_cui,
                synonym=entity.mapped_synonym
            )
        self.existing_nodes = current_existing_nodes
        return

    def _load_seed_kg_from_neo4j(self, sample_k:int=None, start_from:int=None, end_at:int=None):
        
        """Load current KG by loading seed triplets, current triplets (including seed and generated triplets), current relations and nodes types.
        
        # Inputs:
            - sample_k [int]: Load a random sample of triplets, not full KG.
            - start_from [int]: start from triplet index n, given sorted triplets by head names ASC.
            - end_at [int]: Stop at triplet index m, given sorted triplets by head names ASC.
        # Output:
          - None"""
        
        if self._session == None:
            raise ValueError("No Neo4j Session defined. Terminating ...")
          
        logger.info("Loading Seed KG Started.")

        # Get Entity Types and Relation Types
        self._current_entity_types, self._current_relation_types = self._get_entitypes_and_reltypes_from_neo4j()

        logger.info(f"[Loading Seed KG] {len(self._current_entity_types)} Entity Types and {len(self._current_relation_types)} Relation Types detected in Seed KG.")
        
        self._reload_seed_kg_from_neo4j()

        logger.info(f"Total Seed Nodes = {len(self.existing_nodes)}")
        # set current variables
        current_triplets: List[BiomedicalTriplet] = []

        result = self._session.run(
            self._seed_kg_init_query
        )

        counter: int = 0
        records = list(result)
        if sample_k:
            logger.info(f"[Seed KG] Load {sample_k} random triplets.")
            records_to_load = random.sample(records, k=sample_k)
        elif start_from != None and end_at != None:
            records_to_load = records[start_from:end_at]
        else:
            records_to_load = records
            logger.info(f"[Seed KG] Load Full KG ({len(records_to_load)} Triplets).")

        for record in tqdm(records_to_load, desc=f"Loading Triplets and harmonizing their nodes . . .", unit="Triplet"):
            counter += 1 # Count the number of records processed
        
            r_properties = ast.literal_eval(record["r_properties"]) if isinstance(record["r_properties"], str) else record["r_properties"] or {} # Handle properties safely
        
            # Create BiomedicalEntity instances for head and tail entities and add them to existing nodes
            head_entity = BiomedicalEntity(
                name=str(record["head"]),
                entity_id=str(record["head_id"]),
                entity_type=str([label for label in record['head_type']][0]),
                mapped_umls_cui = str(record['head_cui']),
                mapped_synonym = str(record['head_syn']),
            )

        
            tail_entity = BiomedicalEntity(
                name=str(record["tail"]),
                entity_id=str(record["tail_id"]),
                entity_type=str([label for label in record['tail_type']][0]),
                mapped_umls_cui = str(record['tail_cui']),
                mapped_synonym = str(record['tail_syn']),
            )
        
            # Create evidences dictionary
            evidences = {
                "pmid": r_properties.get("pmid", None),
                "evidence": r_properties.get("evidence", None)
            }
                
            # Create a BiomedicalTriplet instance
            triplet = BiomedicalTriplet(
                head=head_entity,
                tail=tail_entity,
                relation=record["relation"],
                curator_evidences=json.dumps(evidences, indent=2, ensure_ascii=False),
            )

            current_triplets.append(triplet)
        
        # initiate or reset class variables
        self.seed_triplets = current_triplets

        logger.info(f"Seed triplets to be enriched = {len(self.seed_triplets)}")
        logger.info(f"Total triplets = {len(self.triplets)}")
        return

    def _reload_seed_kg_from_neo4j(self):

        """Reload/Load All Triplets (Seed + LLM Generated) and store them as current triplets after each iteration"""
        
        if self._session == None:
            raise ValueError("No Neo4j Session defined. Terminating ...")
        
        # Get Entity Types and Relation Types
        self._current_entity_types, self._current_relation_types = self._get_entitypes_and_reltypes_from_neo4j()

        logger.info(f"[ReLoading Seed KG] {len(self._current_entity_types)} Entity Types and {len(self._current_relation_types)} Relation Types detected in Seed KG.")
        
        # set current variables
        current_triplets: List[BiomedicalTriplet] = []
        self._load_seed_nodes_from_neo4j()

        result = self._session.run(
            """
MATCH (n)-[r]->(m)
WHERE 
  n.name IS NOT NULL AND m.name IS NOT NULL AND
  (r.evidence IS NOT NULL OR r.llm_generated = True)
RETURN 
  n.name AS head, 
  labels(n) AS head_type,
  n.mapped_umls_cui as head_mapped_umls_cui,
  n.mapped_umls_synonym as head_mapped_umls_synonym,
  type(r) AS relation, 
  m.name AS tail, 
  labels(m) AS tail_type,
  m.mapped_umls_cui as tail_mapped_umls_cui,
  m.mapped_umls_synonym as tail_mapped_umls_synonym,
  elementId(n) AS head_id,
  elementId(m) AS tail_id,
  r.extra_evidence AS extra_evidence
ORDER BY head ASC 
"""
        )

        counter: int = 0
        records_to_load = list(result)
        logger.info(f"[Reloading Seed KG] Detected ({len(records_to_load)} Triplets).")

        for record in records_to_load:
            counter += 1 # Count the number of records processed
        
            # Create BiomedicalEntity instances for head and tail entities and add them to existing nodes
            head_entity = BiomedicalEntity(
                name=str(record["head"]),
                entity_id=str(record["head_id"]),
                entity_type=str([label for label in record['head_type']][0]),
                mapped_umls_cui=str(record["head_mapped_umls_cui"]),
                mapped_synonym=str(record["head_mapped_umls_synonym"]),
            )
        
            tail_entity = BiomedicalEntity(
                name=str(record["tail"]),
                entity_id=str(record["tail_id"]),
                entity_type=str([label for label in record['tail_type']][0]),
                mapped_umls_cui=str(record["tail_mapped_umls_cui"]),
                mapped_synonym=str(record["tail_mapped_umls_synonym"]),
            )
                
            # Create a BiomedicalTriplet instance
            triplet = BiomedicalTriplet(
                head=head_entity,
                tail=tail_entity,
                relation=record["relation"],
                evidences=json.dumps(record["extra_evidence"], ensure_ascii=False, indent=2)
            )

            current_triplets.append(triplet)

        self.triplets = current_triplets

    def _get_entitypes_and_reltypes_from_neo4j(self) -> tuple[set[str], set[str]]:
        """
        Connects to Neo4j and returns (node_labels, relationship_types) as sets of strings.
        Tries system procedures first; falls back to MATCH-based queries if procedures are unavailable.
        """
        # Labels
        try:
            entity_types = self._session.run("CALL db.labels() YIELD label RETURN label").value()
        except Exception:
            entity_types = self._session.run("""
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN DISTINCT label
            """).value()

        # Relationship types
        try:
            rel_types = self._session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").value()
        except Exception:
            rel_types = self._session.run("""
                MATCH ()-[r]->()
                RETURN DISTINCT type(r) AS relationshipType
            """).value()

        return set(entity_types), set(rel_types)

    def add_new_triplet_to_kg(self, new_triplet:BiomedicalTriplet, llm_validation_flag:str):
        """
        Add new triplets representing a missing hop to the knowledge graph.
        Args:
            new_triplets (dict[str,BiomedicalTriplet]): dict of new triplets to be added to the knowledge graph.
            session: Neo4j session object.
        Returns:
            None
        """

        if self._session:
            q = f"""
        MERGE (head:{new_triplet.head.entity_type} {{ name: $head_name}})
        SET head.mapped_umls_cui = $h_mapped_umls_cui
        SET head.mapped_umls_synonym = $h_mapped_umls_synonym
        SET head.llm_generated = $h_llm_generated 

        MERGE (tail:{new_triplet.tail.entity_type} {{ name: $tail_name }})
        SET tail.mapped_umls_cui = $t_mapped_umls_cui
        SET tail.mapped_umls_synonym = $t_mapped_umls_synonym 
        SET tail.llm_generated = $t_llm_generated  

        MERGE (head)-[r:{new_triplet.relation}]->(tail)
        SET r.extra_evidence = $evidences,
            r.llm_generated = true,
            r.llm_model = $llm_model,
            r.validation_by_extra_evidence = $llm_validation_flag
        """

            # Run the query to add the new triplets to the knowledge graph
            self._session.run(q, {
                "head_name": new_triplet.head.name,
                "h_mapped_umls_cui": new_triplet.head.mapped_umls_cui,
                "h_mapped_umls_synonym": new_triplet.head.mapped_synonym,
                "h_llm_generated": new_triplet.head.llm_generated,

                "tail_name": new_triplet.tail.name,
                "t_mapped_umls_cui": new_triplet.tail.mapped_umls_cui,
                "t_mapped_umls_synonym": new_triplet.tail.mapped_synonym,
                "t_llm_generated": new_triplet.tail.llm_generated,

                "evidences": [new_triplet.evidences],
                "llm_model": self._llm_model,
                "llm_validation_flag": llm_validation_flag
            })
        elif self._enriched_csv_file_path:
            hops_list = [
                {
                    "head_node_id": new_triplet.head.entity_id,
                    "head_node_name": new_triplet.head.name,
                    "head_node_type": new_triplet.head.entity_type,
                    "relation_type": new_triplet.relation,
                    "tail_node_id": new_triplet.tail.entity_id,
                    "tail_node_name": new_triplet.tail.name,
                    "tail_node_type": new_triplet.tail.entity_type,
                    
                    "curator_evidence_pmid": None,
                    "curator_evidence_text": None,

                    "head_node_mapped_umls_cui": new_triplet.head.mapped_umls_cui,
                    "head_node_mapped_umls_synonyms": new_triplet.head.mapped_synonym,
                    "tail_node_mapped_umls_cui": new_triplet.tail.mapped_umls_cui,
                    "tail_node_mapped_umls_synonyms": new_triplet.tail.mapped_synonym,

                    "extra_evidence": new_triplet.evidences,
                    "llm_generated_rel": True,
                    "llm_generated_head": new_triplet.head.llm_generated,
                    "llm_generated_tail": new_triplet.tail.llm_generated,
                    "llm_model": self._llm_model,

                    "validation_by_extra_evidence": llm_validation_flag,
                    "llm_validated": None,
                    "validation_by_curator_evidence": None,
                }
            ]
            hops_df = pd.DataFrame(hops_list)

            # Read header only (for column alignment)
            existing_df = pd.read_csv(self._enriched_csv_file_path, nrows=0)
            expected_cols = list(existing_df.columns)

            # Reindex hops_df to include all columns, filling missing with None
            hops_df = hops_df.reindex(columns=expected_cols)

            # Append aligned row(s)
            hops_df.to_csv(self._enriched_csv_file_path, mode="a", header=False, index=False)
        else:
            raise ValueError(f"[Seed KG Updater] No Seed KG defined.")

    def add_new_evidence_to_kg(self, existing_triplet:BiomedicalTriplet, new_evidence):
        # Prepare the Cypher query to add the new evidence

        q = f"""MATCH (seed_head)-[r]-(seed_tail)
    WHERE seed_head.name = $head_name AND labels(seed_head) = $head_type AND seed_tail.name = $tail_name AND labels(seed_tail) = $tail_type AND type(r) = $relation
    SET r.extra_evidence = coalesce(r.extra_evidence, []) + $new_evidence"""

        # Run the query to add the new triplets to the knowledge graph
        self._session.run(q, {
            "head_name": existing_triplet.head.name,
            "head_type": existing_triplet.head.entity_type,
            "tail_name": existing_triplet.tail.name,
            "tail_type": existing_triplet.tail.entity_type,
            "relation": existing_triplet.relation,
            "new_evidence": new_evidence,
            'llm_model': self._llm_model
        })

    def add_triplet_validation(self, evaluated_triplet:Dict):

        """Add validation reports to seed triplet."""
        # Prepare the Cypher query to add the new evidence

        if self._session:

            q = f"""MATCH (seed_head)-[r]->(seed_tail)
        WHERE elementId(seed_head) = $headId AND elementId(seed_tail) = $tailId AND type(r) = $relation
        SET r.extra_evidence = coalesce(r.extra_evidence, []) + $new_evidence
        SET r.validation_by_extra_evidence = $validation_by_extra_evidence
        SET r.validation_by_curator_evidence = $validation_by_curator_evidence
        SET r.llm_validated = true"""

            # Run the query to add the new triplets to the knowledge graph
            self._session.run(q, {
                "headId": evaluated_triplet['triplet'].head.entity_id,
                "tailId": evaluated_triplet['triplet'].tail.entity_id,
                "relation": evaluated_triplet['triplet'].relation,
                "new_evidence": evaluated_triplet['triplet'].evidences,
                'validation_by_extra_evidence': evaluated_triplet['vs_ExEvidence_report'],
                'validation_by_curator_evidence': evaluated_triplet['vs_CurEvidence_report'],
            })
        
        elif self._enriched_csv_file_path:

            head_id = evaluated_triplet['triplet'].head.entity_id
            tail_id = evaluated_triplet['triplet'].tail.entity_id
            relation = evaluated_triplet['triplet'].relation

            # Locate the specific record
            mask = (
                (self._current_triplets_df["head_node_id"] == head_id) &
                (self._current_triplets_df["tail_node_id"] == tail_id) &
                (self._current_triplets_df["relation_type"] == relation)
            )

            if mask.any():
                idx = self._current_triplets_df.index[mask][0]

                updated_evidence = evaluated_triplet['triplet'].evidences

                # Update fields
                self._current_triplets_df.at[idx, "extra_evidence"] = updated_evidence
                self._current_triplets_df.at[idx, "validation_by_extra_evidence"] = evaluated_triplet['vs_ExEvidence_report']
                self._current_triplets_df.at[idx, "validation_by_curator_evidence"] = evaluated_triplet['vs_CurEvidence_report']
                self._current_triplets_df.at[idx, "llm_validated"] = True

                # Save back to CSV
                self._current_triplets_df.to_csv(self._enriched_csv_file_path, index=False)
            else:
                logger.info(f"⚠️ Triplet not found in CSV: ({head_id}, {relation}, {tail_id})")

    def update_kg_with_triplets(self, triplets_to_be_added:List[tuple[BiomedicalTriplet,str, int]]):
        """
        Align Schemas and Update the knowledge graph with new triplets extracted from a missing hop.
        Args:
            missing_hop_object (MissingHopOutput): The object containing the missing hop information.
            seed_triplet (BiomedicalTriplet): The original triplet that is being extended with the missing hop.
            existing_nodes (dict): A dictionary of existing nodes in the knowledge graph.
            paragraphs_map (dict): A mapping of paragraph numbers to their corresponding PMCIDs and texts.
            session: Neo4j session object.
            llm_model (str): The name of the LLM model used for extraction.
        Returns:
            None
        """

        n_new_triplets = 0
        n_new_evidences = 0
        n_ignored_hops = 0
        try:
            for new_triplet, llm_evaluation_flag, num in tqdm(triplets_to_be_added, desc='Adding Triplets/Evidences', unit='Triplet'):
                # Ignore Hop if Head == Tail
                if new_triplet.head == new_triplet.tail:
                    n_ignored_hops += 1
                    continue

                # ================== Adding New Triplet ==================
                neutral_relations = ["positive_correlation", "negative_correlation", 'regulates', "associated_with", "interacts_with"]
                new_flag, existing_triplet = self.is_new_triplet(new_triplet, neutral_relations)
                if new_flag:
                    n_new_triplets += 1
                    self.add_new_triplet_to_kg(new_triplet, llm_evaluation_flag)
                    self.triplets.append(new_triplet)
                else:
                    if self._session:
                        n_new_evidences += 1
                        new_evidence = new_triplet.evidences
                        self.add_new_evidence_to_kg(existing_triplet, new_evidence)
                    else:
                        logger.info(f"No Neo4j detected. Adding Extra Evidences to Already Existing Triplets is not possible.")

            return n_new_evidences, n_new_triplets, n_ignored_hops
        except Exception as e:
            logger.info(f"[KG Updater] Stopped due to: {e}")
            traceback.print_exc()
            return n_new_evidences, n_new_triplets, n_ignored_hops

    def is_new_triplet(self, new_triplet:BiomedicalTriplet, neutral_relations:List[str]):
        for existing_triplet in self.triplets:
            if existing_triplet.relation == new_triplet.relation:
                if existing_triplet.head == new_triplet.head and existing_triplet.tail == new_triplet.tail:
                    return False, existing_triplet # Already exists
                elif existing_triplet.tail == new_triplet.head and existing_triplet.head == new_triplet.tail:
                    return False, existing_triplet if existing_triplet.relation in neutral_relations else True, None
        return True, new_triplet
    
    def load_seed_kg(self, start_from=None, end_at=None, sample_k=None):
        if self._seed_csv_file_path:
            self._fetch_nodes_and_triplets_from_seed_csv()
        elif self._session:
            self._load_seed_kg_from_neo4j(sample_k=sample_k, start_from=start_from, end_at=end_at)
        else:
            raise ValueError("No Seed KG files/ports detected.")
        
    def reload_seed_kg(self):
        if self._seed_csv_file_path:
            self._fetch_nodes_and_triplets_from_seed_csv()
        elif self._session:
            self._reload_seed_kg_from_neo4j()
        else:
            raise ValueError("No Seed KG files/ports detected.")
        
class KGEnricher():
    def __init__(self,
        llm_name:str,
        temperature:float,
        seed_kg:SeedKG,
        pubmed_fetcher:PubmedFetcher,
        parent_paragraph_fetcher:ParentParagraphFetcher,
        pubmed_web_fetcher:ArticlePipeline,
        n_query:int=10,
        print_process:bool=False,
        export_folder:str=None,
        top_k:int=5,
        timeout:int=300,
        ollama_port:int=11434):

        self.n_query:int = n_query
        self.print_process:bool = print_process
        self.top_k:int = top_k
        self.export_folder:str = export_folder
        self.llm_name = llm_name
        self._timeout = timeout
        self.ollama_port=ollama_port
        
        # Clients
        self.seed_kg:SeedKG = seed_kg
        self._new_entity:BiomedicalEntity = BiomedicalEntity(name="None",entity_type="None")
        self.pubmed_fetcher:PubmedFetcher = pubmed_fetcher
        self.parent_paragraph_fetcher:ParentParagraphFetcher = parent_paragraph_fetcher
        self.pubmed_web_fetcher:ArticlePipeline = pubmed_web_fetcher

        # Agents
        self.paragraph_evaluator:ParagraphEvaluator = ParagraphEvaluator(model=llm_name, temperature=temperature, ollama_port=ollama_port)
        self.pathway_builder:PathwayBuilder = PathwayBuilder(model=llm_name, temperature=temperature, ollama_port=ollama_port)
        self.schema_aligner:HopAligner = HopAligner(model=llm_name, ollama_port=ollama_port)
        self._entity_matcher:EntityMatcherEvaluator = EntityMatcherEvaluator(model=llm_name, ollama_port=ollama_port)
        self.hop_evaluator:HopValidationTeam = HopValidationTeam(
            model=llm_name,
            temperature=temperature,
            pubmed_fetcher=self.pubmed_fetcher,
            parent_paragraph_fetcher=self.parent_paragraph_fetcher,
            pubmed_web_fetcher=self.pubmed_web_fetcher,
            top_k=self.top_k
            , ollama_port=ollama_port)

    def _evaluate_paragraphs(self, query_question:str, paragraphs_map:dict[int,dict[str,str]], triplet_report_folder:str=None):
        return self.paragraph_evaluator.evaluate(query_question, paragraphs_map, print_output=True, export_output=True, output_folder=triplet_report_folder).evaluations
    
    def _run_with_timeout(self, func, *args, timeout=300, triplet_folder:str="", **kwargs):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                logger.info(f"[Ollama] timed out after {timeout} seconds")
                restart_ollama(port=self.ollama_port)
                if os.path.exists(triplet_folder):
                    shutil.rmtree(triplet_folder)
                    logger.info(f"[Terminating] Deleted directory: {triplet_folder}")
                return None  # or raise, or return defaults
            

    def _fetch_pubmed_online(self, query_question:str, triplet:BiomedicalTriplet, triplet_report_folder:str) -> dict[int,dict]:
        """Fetch Pubmed Web for relevant articles and paragraphs."""
        query = f"{triplet.head.name} AND {triplet.tail.name}"
        paragraphs_map:Dict[int, Dict] = self.pubmed_web_fetcher.run_and_rank(query=query, query_question=query_question, max_results=25, top_k=self.top_k)
        
        if paragraphs_map:
            relevant_paragraphs_map:dict[int,dict] = {}
            logger.info(f"[Retrieval Pipeline] Indexing {len(paragraphs_map.items())} Paragraphs from Pubmed Database.")
            web_evaluations = self.paragraph_evaluator.evaluate(query_question, paragraphs_map, print_output=self.print_process, output_folder=triplet_report_folder).evaluations
            for e in web_evaluations:
                if e.evaluation in ['STRONGLY_RELEVANT', 'PARTIALLY_RELEVANT'] and e.paragraph_number in paragraphs_map:
                    relevant_paragraphs_map[e.paragraph_number] = {
                        "pmcid_or_doi": paragraphs_map.get(e.paragraph_number).get('pmcid_or_doi'),
                        "paragraph_text": paragraphs_map.get(e.paragraph_number).get('paragraph_text'),
                        "is_paragraph_relevant": e.explanation,
                    }
        
            return relevant_paragraphs_map if relevant_paragraphs_map else None
        else:
            logger.info(f"[Web Scrapper] {len(paragraphs_map.items())} Paragraphs were retrieved from Pubmed Database.")
            return None

    def _retieval_pipeline(self, query_question:str, triplet:BiomedicalTriplet, triplet_report_folder:str):

        """Fetch relevant paragraphs and evaluate them.
        Inputs:
        - query_question : Str
        - triplet : BiomedicalTriplet object
        - triplet_report_folder: Triplet reporting folder path
        
        Output:
        Dictionary: Relevant Paragraphs and IDs."""

        logger.info(f"[Retrieval Pipeline] Fetching vector database ...")

        # [STEP 1] >> Retrieve Chunks (Sentences) from Pubmed Vector Database.
        retrieved_docs:list[dict] = self.pubmed_fetcher.fetch(query_question, top_k=self.top_k)

        # [STEP 2] >> Get Parent Paragraphs for each Chunk.
        logger.info(f"[Retrieval Pipeline] Getting Parent Paragraphs ...")
        qdrant_paragraphs_map = self.parent_paragraph_fetcher.get_parent_paragraphs(retrieved_docs)
        assert qdrant_paragraphs_map, ValueError(f"Failed: No Parent Paragraphs were Retrieved from SQLite, although {len(retrieved_docs)} chunks where retrieved from Qdrant!")

        # [STEP 3] >> Evaluation of Retrieved Paragraphs.
        qdrant_evaluations = self.paragraph_evaluator.evaluate(query_question, qdrant_paragraphs_map, print_output=self.print_process, output_folder=triplet_report_folder).evaluations

        # [STEP 4] >> Extract Only 'STRONGLY_RELEVANT'and 'PARTIALLY_RELEVANT' Paragraphs from the retrieved pool.
        relevant_paragraphs_map:dict[int,dict] = {}
        for e in qdrant_evaluations:
            if e.evaluation in ['STRONGLY_RELEVANT', 'PARTIALLY_RELEVANT'] and e.paragraph_number in qdrant_paragraphs_map:
                relevant_paragraphs_map[e.paragraph_number] = {
                    "pmcid_or_doi": qdrant_paragraphs_map.get(e.paragraph_number).get('pmcid_or_doi'),
                    "paragraph_text": qdrant_paragraphs_map.get(e.paragraph_number).get('paragraph_text'),
                    "is_paragraph_relevant": e.explanation,
                }

        # [STEP 5] >> If No Relevant Paragraphs from Qdrant Vector Database were retrieved, Fetch Pubmed Online Database using Keywords.
        if not relevant_paragraphs_map:
            logger.info(f"[Retrieval Pipeline] No Relevant Chunks in Qdrant Databse.")
            relevant_paragraphs_map = self._fetch_pubmed_online(query_question, triplet, triplet_report_folder)

        return relevant_paragraphs_map if relevant_paragraphs_map else None
    
    def _match_entities(self, hops_to_be_matched:List[Hop], relevant_paragraphs_dict:dict[int,dict]) -> List[BiomedicalTriplet]:

        matched_triplets: List[BiomedicalTriplet] = []
        for hop in tqdm(hops_to_be_matched, desc='Matching Hops Entities ...', unit='Hop'):
            evidence_text = relevant_paragraphs_dict.get(hop.evidence_paragraph_number).get('paragraph_text')
            pmcid_or_doi = relevant_paragraphs_dict.get(hop.evidence_paragraph_number).get('pmcid_or_doi')
            explanation = hop.explanation
            
            # ===================================================================== solve head entity name ===================================================================== 
            final_head_entity:BiomedicalEntity = None
            if hop.head.lower() == self._new_entity.name.lower():
                final_head_entity = self._new_entity

            if not final_head_entity:
                head_umls_match = self.seed_kg._entity_harmonizer.fetch(f"{hop.head} {hop.head_type}", 1)[0]
                head_umls_synonym, head_umls_cui = head_umls_match.get('synonyms'), head_umls_match.get('cui')

                
                for cui, node in self.seed_kg.existing_nodes:
                    # Try to Match using Name and Type OR CUI
                    if (node.name.lower().strip() == hop.head.lower().strip() and node.entity_type.lower().replace(" ", "_").replace("[","").replace("]","").strip() == hop.head_type.lower().replace(" ", "_").replace("[","").replace("]","").strip()) or (node.llm_generated == True and node.name.lower().strip() == hop.head.lower().strip()):
                        final_head_entity = node
                        break
                    else:
                        # Match with CUI
                        if head_umls_cui == cui:
                            final_head_entity = node
                            break
            
            # If final_head_entity is Matched to Existing Entity
            if final_head_entity:
                # Evaluate the Matched Entity using an EntityMatcherAgent
                original_head_entity = json.dumps({"entity_name" : hop.head, "entity_type" : hop.head_type}, ensure_ascii=False)
                matched_head_entity_str = json.dumps({"entity_name" : final_head_entity.name, "entity_type" : final_head_entity.entity_type}, ensure_ascii=False)

                head_matching_evaluating_obj = self._entity_matcher.check_match(extracted_entity=original_head_entity, seed_entity=matched_head_entity_str, ref_paragraph=evidence_text)

                if not head_matching_evaluating_obj.is_match:
                    logger.info(f"[Entity Matcher] [invalid] {original_head_entity} >>> {matched_head_entity_str}")
                    final_head_entity = BiomedicalEntity(
                        name=hop.head,
                        entity_type=hop.head_type.lower().replace(" ", "_"),
                        entity_id=None,  # New entity, so no ID yet
                        mapped_umls_cui=head_umls_cui,
                        mapped_synonym=head_umls_synonym,
                        llm_generated=True
                    )
                    self.seed_kg.existing_nodes.append((head_umls_cui, final_head_entity))
                    logger.info(f"[Entity Matcher] New Entity to be added: {final_head_entity.name} (Type: {final_head_entity.entity_type})")
                else:
                    logger.info(f"[Entity Matcher] [Success] {original_head_entity} >>> {matched_head_entity_str}")
            else:
                # No matching Nodes in Seed KG
                final_head_entity = BiomedicalEntity(
                    name=hop.head,
                    entity_type=hop.head_type.lower().replace(" ", "_"),
                    entity_id=None,  # New entity, so no ID yet
                    mapped_umls_cui=head_umls_cui,
                    mapped_synonym=head_umls_synonym,
                    llm_generated=True
                )
                self.seed_kg.existing_nodes.append((head_umls_cui, final_head_entity))
                logger.info(f"[Entity Matcher] New Entity to be added: {final_head_entity.name} (Type: {final_head_entity.entity_type})")


            # ==================================================================================================================================================================    
            # ===================================================================== solve tail entity name =====================================================================
            final_tail_entity:BiomedicalEntity = None
            if hop.tail == self._new_entity.name:
                final_tail_entity = self._new_entity

            if not final_tail_entity:
                tail_umls_match = self.seed_kg._entity_harmonizer.fetch(f"{hop.tail} {hop.tail_type}", 1)[0]
                tail_umls_synonym, tail_umls_cui = tail_umls_match.get('synonyms'), tail_umls_match.get('cui')

                # Try to Match using Name and Type
                for cui, node in self.seed_kg.existing_nodes:
                    # Try to Match using Name and Type
                    if (node.name.lower().strip() == hop.tail.lower().strip() and node.entity_type.lower().replace(" ", "_").replace("[","").replace("]","").strip() == hop.tail_type.lower().replace(" ", "_").replace("[","").replace("]","").strip()) or (node.llm_generated == True and node.name.lower().strip() == hop.tail.lower().strip()):
                        final_tail_entity = node
                        break
                    else:
                        # Match with CUI
                        if tail_umls_cui in cui:
                            final_tail_entity = node
            
            # If final_tail_entity is Matched to Existing Entity
            if final_tail_entity:
                # Evaluate the Matched Entity using an EntityMatcherAgent
                original_tail_entity = json.dumps({"entity_name" : hop.tail, "entity_type" : hop.tail_type}, ensure_ascii=False)

                matched_tail_entity_str = json.dumps({"entity_name" : final_tail_entity.name, "entity_type" : final_tail_entity.entity_type}
                                                     , ensure_ascii=False)
                tail_matching_evaluating_obj = self._entity_matcher.check_match(extracted_entity=original_tail_entity, seed_entity=matched_tail_entity_str, ref_paragraph=evidence_text)

                if not tail_matching_evaluating_obj.is_match:
                    final_tail_entity = BiomedicalEntity(
                        name=hop.tail,
                        entity_type=hop.tail_type.lower().replace(" ", "_"),
                        entity_id=None,  # New entity, so no ID yet
                        mapped_umls_cui=tail_umls_cui,
                        mapped_synonym=tail_umls_synonym,
                        llm_generated=True
                    )
                    self.seed_kg.existing_nodes.append((tail_umls_cui, final_tail_entity))
                    logger.info(f"[Entity Matcher] New Entity to be added: {final_tail_entity.name} (Type: {final_tail_entity.entity_type})")
                else:
                    logger.info(f"[Entity Matcher] {original_tail_entity} >>> {matched_tail_entity_str}")
            else:
                # No matching Nodes in Seed KG
                final_tail_entity = BiomedicalEntity(
                    name=hop.tail,
                    entity_type=hop.tail_type.lower().replace(" ", "_"),
                    entity_id=None,  # New entity, so no ID yet
                    mapped_umls_cui=tail_umls_cui,
                    mapped_synonym=tail_umls_synonym,
                    llm_generated=True
                )
                self.seed_kg.existing_nodes.append((tail_umls_cui, final_tail_entity))
                logger.info(f"[Entity Matcher] New Entity to be added: {final_tail_entity.name} (Type: {final_tail_entity.entity_type})")
            
            new_triplet = BiomedicalTriplet(
                head=final_head_entity,
                tail=final_tail_entity,
                relation=hop.relation.lower().replace(" ", "_"),
                evidences=json.dumps({
                    "pmcid_or_doi": pmcid_or_doi,
                    "paragraph_text": evidence_text,
                    "is_paragraph_relevant": explanation
                }, indent=2, ensure_ascii=False)
            )

            matched_triplets.append(new_triplet)

        return matched_triplets
        
    def _process_hops(self, question_q:str, pathway:Pathway, pathway_ref_paragraphs:dict[int,dict], triplet_report_folder:str) -> List[tuple[BiomedicalTriplet, str, int]]:
        
        # [FIRST] Original Hops
        original_hops = pathway.hops
        self._report_hops_and_pathways(
            query_question=question_q,
            hops=[(hop, 'original', num) for num, hop in enumerate(pathway.hops, start=1)],
            used_relevant_paragraph_map=pathway_ref_paragraphs,
            hops_file_name='original_hops.csv',
            pathways_file_name='original_pathways.csv')
        
        # [SECOND] Aligned Hops
        aligned_hops: List[Hop] = self.schema_aligner.align_hops(
            hops_to_be_aligned=original_hops,
            relevant_paragraphs_dict=pathway_ref_paragraphs,
            current_kg_types=self.seed_kg._current_entity_types,
            current_kg_relations=self.seed_kg._current_relation_types,
            output_folder=triplet_report_folder,
        ) or None  # fallback if timeout

        if not aligned_hops:
            raise ValueError(f"[Hops Alignment] Aligned Hops are {aligned_hops}.")
        
        self._report_hops_and_pathways(
            query_question=question_q,
            hops=[(hop, 'aligned', num) for num, hop in enumerate(aligned_hops, start=1)],
            used_relevant_paragraph_map=pathway_ref_paragraphs,
            hops_file_name='aligned_hops.csv',
            pathways_file_name='aligned_pathways.csv')


        # [THIRD] Generated Entity >> Seed Entity Matching
        matched_triplets: List[BiomedicalTriplet] = self._match_entities(
            hops_to_be_matched=aligned_hops,
            relevant_paragraphs_dict=pathway_ref_paragraphs,
        ) or None  # fallback if timeout

        if not matched_triplets:
            raise ValueError(f"[EntityMatcher] Matched Hops are {matched_triplets}.")
        
        self._report_hops_and_pathways(
            query_question=question_q,
            hops=[(triplet, 'matched-entities', num) for num, triplet in enumerate(matched_triplets, start=1)],
            used_relevant_paragraph_map=pathway_ref_paragraphs,
            hops_file_name='matched_triplets.csv',
            pathways_file_name='matched_pathways.csv')

        # [FOURTH] Evaluate and Fix Triplets
        evaluated_fixed_triplets: List[tuple[BiomedicalTriplet, str, int]] = self.hop_evaluator.evaluate_and_fix_triplets(
            initial_triplets=matched_triplets,
            current_relations=self.seed_kg._current_relation_types,
        ) or None  # fallback if timeout

        if not evaluated_fixed_triplets:
            raise ValueError(f"[Triplet Validation Team] Evaluated/Fixed Triplets are {evaluated_fixed_triplets}.")
        
        self._report_hops_and_pathways(
            query_question=question_q,
            hops=evaluated_fixed_triplets,
            used_relevant_paragraph_map=pathway_ref_paragraphs,
            hops_file_name='evaluated_fixed_triplets.csv',
            pathways_file_name='evaluated_fixed_pathways.csv')
        
        return evaluated_fixed_triplets

    def _pathway_construction_pipeline(self, triplet:BiomedicalTriplet, query_question:str, relevant_paragraphs_map:dict[int,dict], triplet_report_folder:str) -> tuple[Pathway, list[tuple[Hop,str, int]], dict[int,dict], dict]:
        
        # [STEP 1] >> From qdrant chunks, construct the pathway that connects the source to the target (based on the query question) >>> Output: Pathway Object with InValidated Hops.
        q_pathway:Pathway = self.pathway_builder.build_pathway(query_question, relevant_paragraphs_map, print_output=self.print_process, output_folder=triplet_report_folder)
        
        if q_pathway.hops and q_pathway.success_flag == True:
            # [STEP 2] >> Process Each Hop [Align -> Match -> Fix -> Final]
            q_evaluated_fixed_triplets = self._process_hops(query_question, q_pathway, relevant_paragraphs_map, triplet_report_folder)
            return q_evaluated_fixed_triplets
        
        logger.info(f"[Pathway Construction] Pathway Construction from Qdrant Failed due to: {q_pathway.comment}")
        logger.info(f"[Pathway Construction] Trying Web Scraping...")
        web_relevant_paragraphs_map = self._fetch_pubmed_online(query_question, triplet, triplet_report_folder)
        if web_relevant_paragraphs_map:
            # [STEP 1] >> From web paragraphs, construct the pathway that connects the source to the target (based on the query question) >>> Output: Pathway Object with InValidated Hops.
            w_pathway:Pathway = self.pathway_builder.build_pathway(query_question, web_relevant_paragraphs_map, print_output=self.print_process, output_folder=triplet_report_folder)
            if w_pathway.hops and w_pathway.success_flag == True:
                # [STEP 2] >> If Pathway was constructed, Validate and Fix individual hops. >>> Output: Labeled Hops ['Valid' or 'Need-Review'] to be added to the SeedKG.
                w_evaluated_fixed_triplets = self._process_hops(query_question, w_pathway, web_relevant_paragraphs_map, triplet_report_folder)
                return w_evaluated_fixed_triplets
            else:
                # Failed
                logger.info(f"[Pathway Construction] Pathway Construction from Pubmed Web Failed due to: {w_pathway.comment}")
                return None
        
        logger.info(f"[Pathway Construction] Pathway Construction from Pubmed Web Failed due to: {web_relevant_paragraphs_map} Paragraphs retrieved from Pubmed Database.")
        return None
          
    def _report_hops_and_pathways(self, query_question:str, hops:List[tuple[Hop, str, int]] | List[tuple[BiomedicalTriplet, str, int]], used_relevant_paragraph_map:dict[int, dict], hops_file_name:str, pathways_file_name:str) -> None:
        # add row to generated pathways
        if isinstance(hops[0][0], Hop):
            hops=[
                {
                    "head": hop.head,
                    "head_type": hop.head_type,
                    "tail" : hop.tail,
                    "tail_type" : hop.tail_type,
                    "relation" : hop.relation,
                    "evidence" : used_relevant_paragraph_map.get(hop.evidence_paragraph_number).get("paragraph_text"),
                    "llm_validation" : validation_flag
                }
                for hop, validation_flag, _ in hops if hop.evidence_paragraph_number in used_relevant_paragraph_map]
        elif isinstance(hops[0][0], BiomedicalTriplet):
            hops=[
                {
                    "head": hop.head.name,
                    "head_type": hop.head.entity_type,
                    "tail" : hop.tail.name,
                    "tail_type" : hop.tail.entity_type,
                    "relation" : hop.relation,
                    "evidence" : hop.evidences,
                    "llm_validation" : validation_flag
                }
                for hop, validation_flag, _ in hops]
        
        pathway_generated = {
            "query_question": query_question,
            "Hops" : json.dumps(hops, indent=2, ensure_ascii=False),
        }

        pathway_df = pd.DataFrame([pathway_generated])
        hops_df = pd.DataFrame(hops)

        # Check if pathway_df file exists
        if not os.path.isfile(f"{self.export_folder}/{pathways_file_name}"):
            # Create new CSV
            pathway_df.to_csv(f"{self.export_folder}/{pathways_file_name}", index=False)
            logger.info(f"[File Created] {f'{self.export_folder}/{pathways_file_name}'}")
        else:
            # Append without writing the header
            pathway_df.to_csv(f"{self.export_folder}/{pathways_file_name}", mode="a", header=False, index=False)
            logger.info(f"[Pathway Added] In {f'{self.export_folder}/{pathways_file_name}'}")

        # Check if hops file exists
        if not os.path.isfile(f"{self.export_folder}/{hops_file_name}"):
            # Create new CSV
            hops_df.to_csv(f"{self.export_folder}/{hops_file_name}", index=False)
            logger.info(f"[File Created] {f'{self.export_folder}/{hops_file_name}'}")
        else:
            # Append without writing the header
            hops_df.to_csv(f"{self.export_folder}/{hops_file_name}", mode="a", header=False, index=False)
            logger.info(f"[Hops Added] In {f'{self.export_folder}/{hops_file_name}'}")

    def _new_entity_triplets_generator(self, new_entity:BiomedicalEntity, allowed_types:list=None) -> List[BiomedicalTriplet]:

        if allowed_types:
            allowed_nodes = [(cui, entity) for cui, entity in self.seed_kg.existing_nodes if any([allowed_type in entity.entity_type or allowed_type == entity.entity_type for allowed_type in allowed_types])]
        else:
            allowed_nodes = self.seed_kg.existing_nodes

        triplets_to_be_enriched = [
            BiomedicalTriplet(
                head=new_entity,
                tail=tail_entity,
                relation='',
                evidences=''
            )
            for _, tail_entity in allowed_nodes
        ]

        self._new_entity = new_entity

        logger.info(f"{len(triplets_to_be_enriched)} Queries to be Processed!")
        return triplets_to_be_enriched

    def enrich_triplets(self, only_validate:bool=False, start_from:int=0):

        """
            Enrich Seed Triplets by validating using Extra Evidence retrieved from corpora, and adding missing indirect pathways if any.

            Args:
            - only_validate [boolean]: If True, validate seed triplets, without adding missing indirect paths. If False, Validate Seed triplets and add missing indirect paths.
            - Start_from [int]: Triplet index from which enrichment would start.
        """
        triplet_index = 0
        latencies = []
        fails = []
        n_evidence_enrichments = []
        n_triplets_enrichments = []
        n_skipped_triplets = []
        n_ignored_triplets = []

        # Prepare Not-Enriched Triplets list
        seed_triplets = deepcopy(self.seed_kg.seed_triplets)
        triplets_to_be_enriched:List[BiomedicalTriplet] = []
        for triplet in seed_triplets:
            triplet_str = f'{triplet}'
            triplet_report_folder = f'{self.export_folder}/triplets/{triplet_str.replace(" ","_").lower()}'
            if os.path.exists(triplet_report_folder):
                continue
            else:
                triplets_to_be_enriched.append(triplet)

        enriched_triplets = []

        # Start Enrichment
        for triplet in tqdm(triplets_to_be_enriched, desc="Enriching Triplets", unit="Triplet"):
            
            # Print Reports
            if latencies:
                logger.info("+" * 150)
                logger.info(f"[REPORT] {np.median(latencies)} seconds per Query ...")
                logger.info(f"[REPORT] {np.sum(n_evidence_enrichments)} Evidence Enrichments ...")
                logger.info(f"[REPORT] {np.sum(n_triplets_enrichments)} Triplets Enrichments ...")
                logger.info(f"[REPORT] {np.sum(n_ignored_triplets)} Ignored hops ...")
                logger.info(f"[REPORT] {np.sum(n_skipped_triplets)} Skipped Triplets ...")
                logger.info(f"[REPORT] {np.sum(fails)} Failed Queries ...")
                logger.info("+" * 150)
            
            start_time = perf_counter() # Start Time to report latency
            triplet_index += 1 # Triplet Index

            if triplet_index < start_from: # If Triplet index is smaller than the start index, skip triplet.
                logger.info(f"Start From Query {start_from}. Skipping Query {triplet_index}")
                continue
            
            if triplet.head.name == triplet.tail.name and triplet.head.entity_type == triplet.tail.entity_type: # If the relation is from and to the same node, skip triplet.
                logger.info(f'Same Biomedical Entity Triplet Detected. Skipping Triplet: {triplet}')
                continue

            # Prepare Export folder for triplet in-process files. Skip Triplet, if a similar triplet has been processed before.
            if self.export_folder:
                # Create Triplet Folder for report files
                triplet_str = f'{triplet}'
                triplet_report_folder = f'{self.export_folder}/triplets/{triplet_str.replace(" ","_").lower()[:150]}'
                if os.path.exists(triplet_report_folder):
                    logger.info(f"Query {triplet_index}/{len(triplets_to_be_enriched)} has been processed. Skipping...")
                    continue
                elif (f"{triplet.head.name.lower()} (type: {triplet.head.entity_type.lower()})", f"{triplet.tail.name.lower()} (type: {triplet.tail.entity_type.lower()})") in enriched_triplets:
                    logger.info(f"Similar Query to {triplet_index}/{len(triplets_to_be_enriched)} has been already enriched. Skipping...")
                    continue
                else:
                    os.makedirs(triplet_report_folder, exist_ok=True)
            else:
                triplet_report_folder = None

            try:
                logger.info("\n" * 2)
                logger.info("=" * 150)
                # format question query
                logger.info(f"Query {triplet_index}/{len(triplets_to_be_enriched)}: {triplet.head.name} (type: {triplet.head.entity_type}) >>> {triplet.tail.name} (type: {triplet.tail.entity_type})")
                enriched_triplets.append((f"{triplet.head.name.lower()} (type: {triplet.head.entity_type.lower()})", f"{triplet.tail.name.lower()} (type: {triplet.tail.entity_type.lower()})"))

                # [Validate Seed Triplet] >> Validate Seed Triplet first, using Curator's and extra evidences retrieved from Corpora and/or Pubmed Web Fetcher
                evaluated_triplet: Dict = self.hop_evaluator.validate_triplet(
                    triplet=triplet
                ) or None  # fallback if timeout
                
                logger.info(f"[Seed Triplet Validation] Adding Validation to Seed Triplet ...")
                self.seed_kg.add_triplet_validation(evaluated_triplet)
                
                if only_validate: # Don't search for missing paths if only_validate = True
                    logger.info(f"Seed Triplet validation Finished. Continue to the next triplet.")
                    elapsed = perf_counter() - start_time
                    latencies.append(round(elapsed))
                    continue
                
                logger.info(f'Seed Triplet Validation Finished. Searching for missing indirect Pathway ...')
                # [Retrieval Pipeline] >> Retrieve Paragraphs that are relevant to the Query Question (Triplet). >>> Output: Relevant Paragraphs and their Summaries.
                query_question = f"What is the biomedical pathway that connects {triplet.head.name} (type: {triplet.head.entity_type}), as the source, to {triplet.tail.name} (type: {triplet.tail.entity_type}), as the target, with the direction from source to target ?"
                relevant_paragraphs_map = self._run_with_timeout(
                    self._retieval_pipeline,
                    query_question=query_question,
                    triplet=triplet,
                    triplet_report_folder=triplet_report_folder,
                    timeout=self._timeout * 2,
                    triplet_folder=triplet_report_folder
                ) or None # fallback if timeout

                if not relevant_paragraphs_map:
                    logger.info(f"[Retrieval Pipeline] No Relevant Paragraphs have been Retrieved.")
                    n_skipped_triplets.append(1)
                    continue
                else:
                    logger.info(f"[Retrieval Pipeline] {len(relevant_paragraphs_map)} Relevant Paragraphs have been Retrieved.")
                
                # [Pathway Construction Pipeline] >> Construct a path from Source to Target from the given paragraphs >>> Output: Hops to be added to Seed Knowledge Graph.
                evaluated_fixed_triplets = self._run_with_timeout(
                    self._pathway_construction_pipeline,
                    triplet=triplet,
                    query_question=query_question,
                    relevant_paragraphs_map=relevant_paragraphs_map,
                    triplet_report_folder=triplet_report_folder,
                    timeout=self._timeout,
                    triplet_folder=triplet_report_folder
                ) or None  # fallback if timeout
                
                if not evaluated_fixed_triplets:
                    logger.info(f"[Enriching Seed KG] No Triplets to be added to Seed KG.")
                    n_skipped_triplets.append(1)
                    continue
                else:
                    logger.info(f"[Enriching Seed KG] Adding Triplets ...")
                n_new_evidences, n_new_triplets, n_ignored_hops = self.seed_kg.update_kg_with_triplets(triplets_to_be_added=evaluated_fixed_triplets)

                # Report
                n_evidence_enrichments.append(n_new_evidences)
                n_triplets_enrichments.append(n_new_triplets)
                n_ignored_triplets.append(n_ignored_hops)

                self.seed_kg.reload_seed_kg()
                
                elapsed = perf_counter() - start_time
                latencies.append(round(elapsed))
            except Exception as e:
                logger.info(f"[FAIL] Triplet: {triplet} failed due to : {e}")
                traceback.print_exc()
                fails.append(1)
                continue
        
        logger.info(f"[SUCCESS] model {self.llm_name} has been used to enrich the Seed KG.")
        logger.info("+" * 150)
        logger.info(f"[REPORT] {np.median(latencies)} seconds per Query ...")
        logger.info(f"[REPORT] {np.sum(n_evidence_enrichments)} Evidence Enrichments ...")
        logger.info(f"[REPORT] {np.sum(n_triplets_enrichments)} Triplets Enrichments ...")
        logger.info(f"[REPORT] {np.sum(n_ignored_triplets)} Ignored hops ...")
        logger.info(f"[REPORT] {np.sum(fails)} Failed Queries ...")
        logger.info("+" * 150)
