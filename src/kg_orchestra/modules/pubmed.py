# python
"""
Agentic pipeline (PubMed + PubMed Central) to:
- Search PubMed by query
- Prefer PMC full text (JATS XML); else try PDF; else HTML; else fall back to PubMed abstract
- Chunk strictly into complete paragraphs with no length limits
- Return { article_id: [paragraphs] }, where article_id prefers DOI, then PMCID, then PMID

Requirements:
- requests (pip install requests)
- Optional: beautifulsoup4 (HTML parsing), pdfminer.six (PDF text)
"""

from __future__ import annotations
import io
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional dependencies
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
except Exception:
    pdf_extract_text = None


# ---------------------------
# Utilities
# ---------------------------

class RateLimiter:
    """Simple thread-safe rate limiter enforcing a minimum interval between calls."""
    def __init__(self, max_per_sec: float = 3.0):
        self.min_interval = 1.0 / max_per_sec if max_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.time()


class HttpClient:
    """HTTP client with retries and default headers."""
    def __init__(self, timeout: float = 20.0, max_retries: int = 3, backoff: float = 0.7):
        self.session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._lock = threading.Lock()
        self.session.headers.update({
            "User-Agent": "AgenticScholar/1.0 (+https://example.org/bot; contact: you@example.org)"
        })

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, accept: Optional[str] = None) -> requests.Response:
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self._lock:
                    if accept:
                        self.session.headers["Accept"] = accept
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if 200 <= resp.status_code < 300:
                    return resp
                elif resp.status_code in (429, 502, 503, 504):
                    sleep_for = self.backoff * attempt
                    time.sleep(sleep_for)
                else:
                    resp.raise_for_status()
            except requests.RequestException as e:
                last_exc = e
                sleep_for = self.backoff * attempt
                time.sleep(sleep_for)
        if last_exc:
            raise last_exc
        raise RuntimeError("Unexpected HTTP retry loop fall-through.")
    
@dataclass
class RankedParagraph:
    para_idx: int
    text: str
    doi: str
    score: float


class DenseParagraphRanker:
    def __init__(
        self,
        model,
        device: Optional[str] = None,  # e.g., "cuda" or "cpu"
        add_task_prefix: bool = False,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ):
        self.model = model
        self.add_task_prefix = add_task_prefix
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        self._paragraphs: List[str] = []
        self._dois: List[str] = []
        self._doc_embs: Optional[np.ndarray] = None  # shape: [N, D]

    def _prep_query(self, query: str) -> str:
        return (self.query_prefix + query) if self.add_task_prefix else query

    def _prep_passages(self, paragraphs: List[str]) -> List[str]:
        if not self.add_task_prefix:
            return paragraphs
        return [self.passage_prefix + p for p in paragraphs]

    def index(self, doi_paras_map: Dict[str, List[str]], batch_size: int = 64) -> None:
        """
        Embed and store the whole paragraphs (no chunking).
        Call once if you'll search multiple queries against the same set.
        """
        for doi, paras in doi_paras_map.items():
            for para in paras:
                self._paragraphs.append(para)
                self._dois.append(doi)

        texts = self._prep_passages(self._paragraphs)

        print(f"[Web Scrapper] Indexing {len(texts)} Paragraphs from Pubmed Database.")
        self._doc_embs, _ = self.model.encode(
            docs=texts,
            encoder_type="dense"
        )

    def search(self, query: str, top_k: int = 10) -> List[RankedParagraph]:
        """
        Rank stored paragraphs for a query using cosine similarity.
        Requires index() to be called first.
        """
        if self._doc_embs is None or len(self._paragraphs) == 0:
            return []

        q_text = self._prep_query(query)
        q_emb, _ = self.model.encode(docs=[q_text], encoder_type="dense")

        print(f"[Web Scrapper] Ranking Paragraphs from Pubmed Database.")
        scores = self._doc_embs @ q_emb[0]  # cosine similarity (embeddings are normalized)
        order = np.argsort(-scores)[: min(top_k, len(scores))]

        return [RankedParagraph(int(i), self._paragraphs[int(i)], self._dois[int(i)], float(scores[int(i)])) for i in order]

    def rank(self, doi_paras_map: Dict[str, List[str]], query: str, top_k: int = 10, batch_size: int = 64) -> List[RankedParagraph]:
        """
        Convenience method: embeds the provided paragraphs and immediately searches.
        """
        self.index(doi_paras_map, batch_size=batch_size)
        return self.search(query, top_k=top_k)
    
# ---------------------------
# Data model
# ---------------------------

@dataclass
class ArticleRecord:
    key: str
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    source: str = "PubMed"
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# NCBI Clients (E-utilities + PMC)
# ---------------------------

class PubMedClient:
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, http: HttpClient, tool: str, email: str, rate: Optional[RateLimiter] = None):
        self.http = http
        self.tool = tool
        self.email = email
        self.rate = rate or RateLimiter(3.0)

    def _params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = {"tool": self.tool, "email": self.email}
        if extra:
            base.update(extra)
        return base

    def esearch(self, query: str, retmax: int = 25) -> List[str]:
        """Return PMIDs from PubMed for a query."""
        self.rate.wait()
        url = f"{self.EUTILS_BASE}/esearch.fcgi"
        params = self._params({
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max(1, min(retmax, 100000)),
            "sort": "relevance"
        })
        resp = self.http.get(url, params=params, accept="application/json")
        data = resp.json()
        ids = data.get("esearchresult", {}).get("idlist", []) or []
        return [str(i) for i in ids]

    def efetch_pubmed_xml(self, pmids: List[str]) -> str:
        """Fetch PubMed records (XML) for a list of PMIDs (batched)."""
        if not pmids:
            return ""
        chunks = [pmids[i:i + 200] for i in range(0, len(pmids), 200)]
        xml_parts: List[str] = []
        for chunk in chunks:
            self.rate.wait()
            url = f"{self.EUTILS_BASE}/efetch.fcgi"
            params = self._params({
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(chunk)
            })
            resp = self.http.get(url, params=params, accept="application/xml")
            xml_parts.append(resp.text)
        return "\n".join(xml_parts)

    def details(self, pmids: List[str]) -> Dict[str, ArticleRecord]:
        """Parse PubMed XML into ArticleRecord dict keyed by PMID."""
        xml_text = self.efetch_pubmed_xml(pmids)
        if not xml_text:
            return {}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            records: Dict[str, ArticleRecord] = {}
            for frag in _split_xml_documents(xml_text):
                try:
                    sub = ET.fromstring(frag)
                    records.update(self._parse_pubmed_set(sub))
                except ET.ParseError:
                    continue
            return records
        return self._parse_pubmed_set(root)

    def _parse_pubmed_set(self, root: ET.Element) -> Dict[str, ArticleRecord]:
        out: Dict[str, ArticleRecord] = {}

        for art in root.findall(".//PubmedArticle"):
            # PMID
            pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip() or None

            # Article title
            article_el = art.find(".//MedlineCitation/Article")
            title = None
            if article_el is not None:
                title_el = article_el.find("ArticleTitle")
                if title_el is not None:
                    title = _itertext_join(title_el).strip() or None

            # Abstract (joined paragraphs, keep labeled sections)
            abstract_texts: List[str] = []
            if article_el is not None:
                abstract_el = article_el.find("Abstract")
                if abstract_el is not None:
                    for at in abstract_el.findall("AbstractText"):
                        txt = _itertext_join(at).strip()
                        if txt:
                            label = at.attrib.get("Label")
                            abstract_texts.append(f"{label}: {txt}" if label else txt)
            abstract = "\n\n".join(abstract_texts) if abstract_texts else None

            # DOI — handle all common placements
            doi = None
            for el in art.findall(".//ArticleId"):
                if el.attrib.get("IdType", "").lower() == "doi":
                    doi = (el.text or "").strip()
                    break

            # PMCID — handle both <ArticleId IdType="pmc"> and <OtherID>PMCxxxxxx</OtherID>
            pmcid = None
            for el in art.findall(".//ArticleId"):
                if el.attrib.get("IdType", "").lower() == "pmc":
                    pmcid = _normalize_pmcid(el.text or "")
                    break
            if not pmcid:
                for el in art.findall(".//OtherID"):
                    txt = (el.text or "").strip()
                    if txt.upper().startswith("PMC"):
                        pmcid = _normalize_pmcid(txt)
                        break

            # Compute key (prefer DOI → PMCID → PMID)
            key = ArticleKey.compute(doi=doi, pmcid=pmcid, pmid=pmid)

            # Ensure every record is stored under a valid identifier
            record_id = pmid or pmcid or doi or key

            out[record_id] = ArticleRecord(
                key=key,
                pmid=pmid,
                pmcid=pmcid,
                doi=doi,
                title=title,
                abstract=abstract,
                source="PubMed",
                raw={}
            )

        return out



class PMCClient:
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    OA_UTILS = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    ARTICLE_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"

    def __init__(self, http: HttpClient, tool: str, email: str, rate: Optional[RateLimiter] = None):
        self.http = http
        self.tool = tool
        self.email = email
        self.rate = rate or RateLimiter(3.0)

    def _params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = {"tool": self.tool, "email": self.email}
        if extra:
            base.update(extra)
        return base

    def efetch_jats(self, pmcid: str) -> Optional[str]:
        """Fetch PMC JATS XML via E-utilities."""
        pmcid = _normalize_pmcid(pmcid)
        self.rate.wait()
        url = f"{self.EUTILS_BASE}/efetch.fcgi"
        params = self._params({"db": "pmc", "id": pmcid, "retmode": "xml"})
        try:
            resp = self.http.get(url, params=params, accept="application/xml")
            txt = resp.text.strip()
            if txt and txt.startswith("<"):
                return txt
        except requests.RequestException:
            return None
        return None

    def oa_links(self, pmcid: str) -> Dict[str, List[str]]:
        """Return available OA links (pdf, tgz, etc.) from PMC OA utils."""
        pmcid = _normalize_pmcid(pmcid)
        params = {"id": pmcid}
        try:
            resp = self.http.get(self.OA_UTILS, params=params, accept="application/xml")
        except requests.RequestException:
            return {}
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            return {}
        links: Dict[str, List[str]] = {}
        for link in root.findall(".//link"):
            fmt = link.attrib.get("format", "").lower()
            href = link.attrib.get("href", "").strip()
            if not href:
                continue
            links.setdefault(fmt, []).append(href)
        return links

    def fetch_pdf_bytes(self, url: str) -> Optional[bytes]:
        """Fetch PDF binary bytes."""
        try:
            resp = self.http.get(url, accept="application/pdf")
            if resp.status_code == 200 and resp.content:
                return resp.content
        except requests.RequestException:
            return None
        return None

    def fetch_article_html(self, pmcid: str) -> Optional[str]:
        """Fetch PMC article HTML page."""
        pmcid = _normalize_pmcid(pmcid)
        url = f"{self.ARTICLE_BASE}/{pmcid}/"
        try:
            resp = self.http.get(url, accept="text/html")
            return resp.text if resp.status_code == 200 else None
        except requests.RequestException:
            return None


# ---------------------------
# Paragraph handling
# ---------------------------

class TextUtils:
    @staticmethod
    def clean_whitespace(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def normalize_linebreaks(s: str) -> str:
        return s.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def dehyphenate(s: str) -> str:
        # Merge words split across lines with hyphenation
        return re.sub(r"(\w)-\n(\w)", r"\1\2", s)


class StrictParagraphs:
    """Split into whole paragraphs only (no length limits), unwrapping line-wrapped text."""
    BLANK_BLOCK_RE = re.compile(r"\n\s*\n+", flags=re.MULTILINE)

    @staticmethod
    def split(text: str) -> List[str]:
        if not text:
            return []
        text = TextUtils.normalize_linebreaks(text)
        blocks = [b for b in StrictParagraphs.BLANK_BLOCK_RE.split(text) if b.strip()]
        paragraphs: List[str] = []
        if blocks:
            for b in blocks:
                lines = [ln.strip() for ln in b.split("\n") if ln.strip()]
                para = TextUtils.clean_whitespace(" ".join(lines))
                if para:
                    paragraphs.append(para)
            return paragraphs
        # No blank lines: treat entire text as a single paragraph
        single = TextUtils.clean_whitespace(text)
        return [single] if single else []


class JATSParser:
    """Extract paragraphs from PMC JATS XML body (one paragraph per <p>)."""
    @staticmethod
    def extract_paragraphs(xml_text: str) -> List[str]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        paras: List[str] = []
        for body in root.findall(".//{*}body"):
            for p in body.findall(".//{*}p"):
                txt = "".join(p.itertext())
                cleaned = TextUtils.clean_whitespace(txt)
                if cleaned:
                    paras.append(cleaned)
        # Dedupe consecutive duplicates
        out: List[str] = []
        for p in paras:
            if not out or p != out[-1]:
                out.append(p)
        return out


class HTMLExtractor:
    """Extract paragraphs from PMC HTML pages (one paragraph per <p>)."""
    @staticmethod
    def extract_paragraphs(html: str) -> List[str]:
        if not html:
            return []
        if BeautifulSoup is None:
            # Fallback: naive tag stripping, then strict split
            text = re.sub(r"<[^>]+>", "\n", html)
            return StrictParagraphs.split(text)

        soup = BeautifulSoup(html, "html.parser")

        # Prefer main content container if present
        main = soup.select_one("#maincontent") or soup.find("article") or soup.find("main")
        container = main or soup.body or soup

        # Remove non-content areas
        for sel in ["nav", "aside", "footer", ".social", ".ref-list", "#reference-list", ".references", ".fig", ".table"]:
            for el in container.select(sel):
                el.decompose()

        paras: List[str] = []
        for p in container.find_all("p"):
            txt = p.get_text(separator=" ", strip=True)
            if txt:
                paras.append(TextUtils.clean_whitespace(txt))

        # Fallback: all text
        if not paras:
            text = container.get_text(separator="\n", strip=True)
            paras = StrictParagraphs.split(text)

        return paras


class PDFExtractor:
    """Extract paragraphs from PDF bytes using pdfminer.six, preserving full paragraphs."""
    @staticmethod
    def bytes_to_paragraphs(pdf_bytes: bytes) -> List[str]:
        if not pdf_bytes or pdf_extract_text is None:
            return []
        try:
            text = pdf_extract_text(io.BytesIO(pdf_bytes)) or ""
        except Exception:
            return []
        text = TextUtils.normalize_linebreaks(text)
        text = TextUtils.dehyphenate(text)
        # Strict paragraph split: unwrap line-wrapped text and split on blank blocks only
        return StrictParagraphs.split(text)


# ---------------------------
# Agents
# ---------------------------

class SearchAgent:
    """Agent to search PubMed and return PMIDs."""
    def __init__(self, pubmed: PubMedClient):
        self.pubmed = pubmed

    def search_pmids(self, query: str, max_results: int = 25) -> List[str]:
        return self.pubmed.esearch(query, retmax=max_results)


class MetadataAgent:
    """Agent to fetch PubMed metadata (title, abstract, DOI, PMCID)."""
    def __init__(self, pubmed: PubMedClient):
        self.pubmed = pubmed

    def fetch_records(self, pmids: List[str]) -> List[ArticleRecord]:
        details = self.pubmed.details(pmids)
        return [details[p] for p in pmids if p in details]


class FullTextAgent:
    """Agent to fetch full text from PMC (prefer JATS, then PDF, then HTML)."""
    def __init__(self, pmc: PMCClient):
        self.pmc = pmc

    def fetch_paragraphs(self, record: ArticleRecord) -> List[str]:
        pmcid = record.pmcid
        if not pmcid:
            return []

        # 1) JATS via efetch
        xml = self.pmc.efetch_jats(pmcid)
        if xml:
            paras = JATSParser.extract_paragraphs(xml)
            if paras:
                return paras

        # 2) Try PDF via OA utils
        links = self.pmc.oa_links(pmcid)
        pdf_urls = links.get("pdf", []) or []
        for pdf_url in pdf_urls:
            pdf_bytes = self.pmc.fetch_pdf_bytes(pdf_url)
            if not pdf_bytes:
                continue
            paras = PDFExtractor.bytes_to_paragraphs(pdf_bytes)
            if paras:
                return paras

        # 3) Fallback to HTML page parsing
        html = self.pmc.fetch_article_html(pmcid)
        if html:
            paras = HTMLExtractor.extract_paragraphs(html)
            if paras:
                return paras

        return []


class AbstractAgent:
    """Agent to split abstracts into whole paragraphs only (no length limits)."""
    def chunk(self, abstract: Optional[str]) -> List[str]:
        if not abstract:
            return []
        return StrictParagraphs.split(abstract)


# ---------------------------
# Keys and helpers
# ---------------------------

class ArticleKey:
    @staticmethod
    def compute(doi: Optional[str], pmcid: Optional[str], pmid: Optional[str]) -> str:
        if doi:
            return f"doi:{doi.lower()}"
        if pmcid:
            return f"pmcid:{_normalize_pmcid(pmcid)}"
        if pmid:
            return f"pmid:{pmid}"
        return "unknown"

def _normalize_pmcid(pmcid: str) -> str:
    pmcid = (pmcid or "").strip().upper()
    return pmcid if pmcid.startswith("PMC") else (f"PMC{pmcid}" if pmcid else pmcid)

def _itertext_join(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext())

def _split_xml_documents(xml_text: str) -> List[str]:
    """Best-effort splitter in case multiple XML docs are concatenated."""
    parts: List[str] = []
    buf: List[str] = []
    for line in xml_text.splitlines():
        if line.strip().startswith("<?xml"):
            if buf:
                parts.append("\n".join(buf))
                buf = []
        buf.append(line)
    if buf:
        parts.append("\n".join(buf))
    return parts


# ---------------------------
# Orchestrator
# ---------------------------

class ArticlePipeline:
    """Coordinates agents to produce {article_id: [paragraphs]}."""
    def __init__(
        self,
        search_agent: SearchAgent,
        metadata_agent: MetadataAgent,
        fulltext_agent: FullTextAgent,
        abstract_agent: AbstractAgent,
        dense_ranker,
        prefer_full_text: bool = True,
        max_workers: int = 6,
    ):
        self.search_agent = search_agent
        self.metadata_agent = metadata_agent
        self.fulltext_agent = fulltext_agent
        self.abstract_agent = abstract_agent
        self.prefer_full_text = prefer_full_text
        self.max_workers = max_workers
        self.dense_ranker = dense_ranker

    def run_and_rank(self, query: str, query_question:str, max_results: int = 20, top_k: int = 10) -> Dict[int, Dict]:

        """Search Pubmed for articles and rank Paragraphs, returning the topK list of paragraphs."""
        
        print(f"[Web Scrapper] Fetching Pubmed Database (Max Articles = {max_results}).")
        pmids = self.search_agent.search_pmids(query, max_results=max_results)
        if not pmids:
            return {}

        records = self.metadata_agent.fetch_records(pmids)
        results: Dict[str, List[str]] = {}

        def process(rec: ArticleRecord) -> Tuple[str, List[str]]:
            paragraphs: List[str] = []
            if self.prefer_full_text:
                paragraphs = self.fulltext_agent.fetch_paragraphs(rec)
            if not paragraphs:
                paragraphs = self.abstract_agent.chunk(rec.abstract)
            return rec.key, paragraphs

        if self.max_workers and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = {ex.submit(process, r): r for r in records}
                for fut in as_completed(futures):
                    pmid, paras = fut.result()
                    if paras:
                        results[pmid] = paras
        else:
            for r in records:
                key, paras = process(r)
                if paras:
                    results[key] = paras

        print(f"[Web Scrapper] {len(results.items())} Aricles Retrieved from Pubmed Database.")
        
        if results:
            ranker = DenseParagraphRanker(
                model=self.dense_ranker,
                device=None,
                add_task_prefix=False
            )
            top_k_paragraphs = ranker.rank(results, query_question, top_k=top_k)

            paragraphs_map:Dict[int, Dict] = {}

            for id, ranked_para in enumerate(top_k_paragraphs, start=1):
                paragraphs_map[id] = {
                    "paragraph_text": ranked_para.text,
                    "pmcid_or_doi": ranked_para.doi
                }
            
            return paragraphs_map
        else:
            return {}


# ---------------------------
# Builder
# ---------------------------

def build_pipeline(email: str, dense_ranker, http_timeout: float = 20.0) -> ArticlePipeline:
    http = HttpClient(timeout=http_timeout, max_retries=3, backoff=0.7)
    rate = RateLimiter(3.0)  # Respect NCBI usage guidelines (<= 3 requests/sec without API key)
    pubmed = PubMedClient(http=http, tool="AgenticScholar", email=email, rate=rate)
    pmc = PMCClient(http=http, tool="AgenticScholar", email=email, rate=rate)

    search = SearchAgent(pubmed)
    meta = MetadataAgent(pubmed)
    fulltext = FullTextAgent(pmc)
    abstract = AbstractAgent()

    return ArticlePipeline(
        search_agent=search,
        metadata_agent=meta,
        fulltext_agent=fulltext,
        abstract_agent=abstract,
        dense_ranker=dense_ranker,
        prefer_full_text=True,
        max_workers=1
    )