from typing import List, Literal
from pydantic import BaseModel, Field, model_validator

"""
Pydantic output models used across KG-Orchestra agents.

These models define the structured inputs/outputs for:

- Pathway construction (Hop, Pathway)
- Schema alignment (AlignedHop)

- Paragraph evaluation and summarization (TextSummaryOutput, ParagraphSummaries, ParagraphEvaluation, ParagraphsEvaluationsOutput)
- Triplet validation and repair (TripletEvaluation, FixedRelation)

- Entity matching (EntityMatchingOutput)

Note:

- Field descriptions are designed to guide LLMs toward consistent, schema-aligned outputs.
- Keep the schema stable to avoid breaking downstream parsing/validation.

"""


class Hop(BaseModel):
    """Represents a hop in a pathway, which includes the head and tail entities, their types, the relation between them, and the paragraph number from which this hop was extracted."""
    # Canonical head entity name
    head: str = Field(..., description="The name of the biomedical head entity in the hop.")
    # Head entity type (aligned or proposed)
    head_type: str = Field(..., description="The Biomedical type of the biomedical head entity in the hop. For example, Protein, Drug, Biological Process, Disease, ...etc.")
    # Canonical tail entity name
    tail: str = Field(..., description="The name of the biomedical tail entity in the hop.")
    # Tail entity type (aligned or proposed)
    tail_type: str = Field(..., description="The Biomedical type of the biomedical tail entity in the hop. For example, Protein, Drug, Biological Process, Disease, ...etc.")
    # Mechanistic/causal/functional/conceptual relation verb/phrase
    relation: str = Field(..., description="The relation type between the head and tail entities in the hop. Use mechanistic/causal/functional/conceptual verbs (e.g., activates, inhibits, increases expression of, decreases expression of, phosphorylates, dephosphorylates, binds to, upregulates, downregulates, promotes, suppresses, contributes to, induces, reduces activation of).")
    # Evidence paragraph number from the provided context
    evidence_paragraph_number: int = Field(..., description="The number of the paragraph from which this hop was extracted, which is found in the pattern 'PARAGRAPH (Number: paragraph_number)'.")
    # Short justification based on the evidence paragraph
    explanation: str = Field(..., description="A short reasoning of why this hop was extracted from the evidence paragraph.")
    

class AlignedHop(BaseModel):
    """Represents a aligned hop to be added to the original KG, which includes the head and tail entities, their aligned types, the aligned relation between them, and the reasoning why types and relation were aligned in such way."""
    # Original head name (must not be changed)
    head: str = Field(..., description="The exact name of the biomedical head entity in the original hop.")
    # Proposed head type aligned to the KG schema
    proposed_head_type: str = Field(..., description="The proposed biomedical type of the head entity.")
    # Original tail name (must not be changed)
    tail: str = Field(..., description="The exact name of the biomedical tail entity in the original hop.")
    # Proposed tail type aligned to the KG schema
    proposed_tail_type: str = Field(..., description="The proposed biomedical type of the tail entity.")
    # Proposed relation aligned to the KG relation set
    proposed_relation: str = Field(..., description="The proposed relation type between the head and tail entities, which reflect mechanistic or causal effect of the head on the tail. ")
    # Brief justification for the mapping decisions
    explanation: str = Field(..., description="A short reasoning of why entity types and relation were aligned in such a way.")


class Pathway(BaseModel):
    """A single, linear sequence of hops connecting a source to a target for a given query."""
    # Original user query string
    query: str = Field(..., description="The Query given by the user.")
    # Ordered list of hops forming the path (from source to target)
    hops: List[Hop] = Field(..., description="The List of logically ordered Hops describing the pathway from the source to the target.")
    # Path construction status (True if a valid path was found)
    success_flag: Literal[True, False] = Field(..., description="Indicates if building pathway was successful (True) or not successful (False).")
    # Brief comment describing success/failure reasons
    comment: str = Field(..., description="A comment stating if the process of building the pathway was successful or not and why.")


class ParagraphEvaluation(BaseModel):
    """Represents Evaluation of a Paragraph, whether it is Relevant or Not Relevant to the query."""
    # Paragraph numeric identifier to evaluate (as in the prompt)
    paragraph_number: int = Field(..., description="The number of the paragraph, found in the pattern 'PARAGRAPH (Number: paragraph_number)', to be Evaluated by the LLM")
    # Relevance grading with clear semantics
    evaluation: Literal['STRONGLY_RELEVANT', 'PARTIALLY_RELEVANT', 'IRRELEVANT'] = Field(..., description="""For each paragraph, assign one of the following:


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
        - Mentions both entities but in completely unrelated contexts.""")

    # Short justification for the assigned relevance
    explanation: str = Field(..., description="A short explanation for your evaluation.")


class ParagraphsEvaluationsOutput(BaseModel):
    """Model for the output of paragraphs evaluations."""
    evaluations: List[ParagraphEvaluation] = Field(..., description="A list of multiple paragraphs relevancy evaluations")


class TripletEvaluation(BaseModel):
    """Evaluation result for a biomedical triplet against evidence paragraphs."""
    # Triplet components and their types
    head: str = Field(..., description="The name of the biomedical head entity in the hop.")
    head_type: str = Field(..., description="The type of the biomedical head entity in the hop.")
    relation: str = Field(..., description="The relation type between the head and tail entities in the hop.")
    tail: str = Field(..., description="The name of the biomedical tail entity in the hop.")
    tail_type: str = Field(..., description="The type of the biomedical tail entity in the hop.")

    # Validation dimensions
    biological_validity: Literal["VALID", "INVALID"] = Field(..., description='whether the relation is biologically plausible between the given entities.')
    semantic_coherence: Literal["COHERENT", "INCOHERENT"] = Field(..., description='whether the relation is biologically plausible between the given entities.')
    directionality: Literal["CORRECT", "INCORRECT"] = Field(..., description='whether the relation is biologically plausible between the given entities.')
    # Short reasoning for the evaluation
    explanation: str = Field(..., description='an explanation of LLM reasoning.')


class EntityMatchingOutput(BaseModel):
    """Decision on whether an extracted entity matches a seed entity, with a brief explanation."""
    is_match: bool = Field(..., description="Given a reference paragraph, set True if the original biomedical entity can be matched to the seed biomedical entity; False if not.")
    explanation: str = Field(..., description="Short reasoning of the judgement.")


class FixedRelation(BaseModel):
    """Proposed repaired relation label for an invalid or incoherent triplet, based on evidence."""
    proposed_relation: str = Field(..., description="The proposed relation, based on the provided evidence paragraphs, that makes the triplet valid and compatible with the evidence paragraphs.")