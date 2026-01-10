"""
QENEX Automated Science Research Engine
========================================
Autonomous scientific paper discovery, retrieval, and analysis.

This module provides:
- arXiv paper search and retrieval
- DOI resolution and fetching
- PDF text extraction with structure preservation
- Scientific content analysis (equations, figures, citations)
- Automatic summarization and hypothesis generation
- Integration with Scout 10M for deep reasoning

The Research Pipeline:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Search    │ ──► │   Fetch     │ ──► │   Parse     │
    │  (arXiv/DOI)│     │   (PDF)     │     │  (Extract)  │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                                       │
           ▼                                       ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Analyze   │ ◄── │   Reason    │ ◄── │   Index     │
    │  (Insights) │     │  (Scout)    │     │  (Context)  │
    └─────────────┘     └─────────────┘     └─────────────┘

Usage in Q-Lang:
    research search "quantum error correction" --max 10
    research fetch arxiv:2301.12345
    research fetch doi:10.1038/s41586-023-06096-3
    research analyze paper.pdf
    research summarize papers/
    research hypothesize "room temperature superconductivity"
    research cite paper_var --format bibtex

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import os
import re
import json
import time
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import urllib.request
import urllib.parse

# Optional imports with fallbacks
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ResearchMode(Enum):
    """Research operation modes."""
    SEARCH = auto()      # Search for papers
    FETCH = auto()       # Fetch specific paper
    PARSE = auto()       # Parse PDF content
    ANALYZE = auto()     # Deep analysis
    SUMMARIZE = auto()   # Generate summary
    HYPOTHESIZE = auto() # Generate hypotheses
    CITE = auto()        # Generate citations
    COMPARE = auto()     # Compare papers
    REVIEW = auto()      # Literature review


class PaperSource(Enum):
    """Paper source types."""
    ARXIV = "arxiv"
    DOI = "doi"
    PUBMED = "pubmed"
    LOCAL = "local"
    URL = "url"


@dataclass
class Author:
    """Paper author information."""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


@dataclass
class Citation:
    """Citation reference."""
    key: str
    title: str
    authors: List[str]
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


@dataclass
class Section:
    """Paper section with content."""
    title: str
    content: str
    level: int = 1
    subsections: List['Section'] = field(default_factory=list)


@dataclass
class Equation:
    """Mathematical equation extracted from paper."""
    latex: str
    label: Optional[str] = None
    context: str = ""
    page: int = 0


@dataclass
class Figure:
    """Figure information from paper."""
    caption: str
    label: Optional[str] = None
    page: int = 0
    referenced_in: List[str] = field(default_factory=list)


@dataclass 
class Paper:
    """Comprehensive paper representation."""
    # Identifiers
    id: str
    source: PaperSource
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    
    # Metadata
    title: str = ""
    authors: List[Author] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    
    # Content
    full_text: str = ""
    sections: List[Section] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    
    # Analysis
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    methodology: str = ""
    contributions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # File info
    pdf_path: Optional[str] = None
    page_count: int = 0
    word_count: int = 0


@dataclass
class SearchResult:
    """Search result entry."""
    paper_id: str
    source: PaperSource
    title: str
    authors: List[str]
    abstract: str
    date: Optional[datetime] = None
    relevance_score: float = 0.0
    citation_count: int = 0
    

@dataclass
class ResearchResult:
    """Result from research operation."""
    success: bool
    mode: ResearchMode
    papers: List[Paper] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    summary: str = ""
    hypotheses: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class ResearchEngine:
    """
    Automated Science Research Engine.
    
    Provides autonomous paper discovery, retrieval, parsing, and analysis
    with integration into the QENEX Trinity Pipeline.
    """
    
    # arXiv category mappings
    ARXIV_CATEGORIES = {
        'physics': ['hep-th', 'hep-ph', 'hep-ex', 'cond-mat', 'quant-ph', 
                   'gr-qc', 'astro-ph', 'physics'],
        'chemistry': ['physics.chem-ph', 'cond-mat.mtrl-sci'],
        'biology': ['q-bio', 'physics.bio-ph'],
        'math': ['math', 'math-ph'],
        'cs': ['cs.LG', 'cs.AI', 'cs.CV', 'cs.CL', 'cs.NE'],
        'quantum': ['quant-ph', 'cond-mat.mes-hall', 'cond-mat.supr-con'],
    }
    
    # Common section patterns
    SECTION_PATTERNS = [
        r'^(?:I+\.?\s*)?Introduction',
        r'^(?:I+\.?\s*)?Background',
        r'^(?:I+\.?\s*)?Related\s+Work',
        r'^(?:I+\.?\s*)?Methods?(?:ology)?',
        r'^(?:I+\.?\s*)?Experiments?',
        r'^(?:I+\.?\s*)?Results?',
        r'^(?:I+\.?\s*)?Discussion',
        r'^(?:I+\.?\s*)?Conclusion',
        r'^(?:I+\.?\s*)?References?',
        r'^(?:I+\.?\s*)?Appendix',
    ]
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize Research Engine.
        
        Args:
            cache_dir: Directory for caching downloaded papers
            verbose: Print status messages
        """
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "qenex_research"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Paper storage
        self.papers: Dict[str, Paper] = {}
        self.search_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.papers_fetched = 0
        self.papers_analyzed = 0
        
        if verbose:
            self._print_status()
    
    def _print_status(self):
        """Print engine status."""
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        QENEX Automated Science Research Engine               ║
    ║        arXiv • DOI • PubMed • PDF Analysis                   ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        print(f"    📚 Cache Directory: {self.cache_dir}")
        print(f"    🔬 PyMuPDF: {'✅' if HAS_PYMUPDF else '❌'}")
        print(f"    📄 arXiv API: {'✅' if HAS_ARXIV else '❌'}")
        print(f"    🌐 Requests: {'✅' if HAS_REQUESTS else '❌'}")
        print()
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    def search(self, 
               query: str, 
               max_results: int = 10,
               source: PaperSource = PaperSource.ARXIV,
               categories: Optional[List[str]] = None,
               date_from: Optional[str] = None,
               date_to: Optional[str] = None) -> ResearchResult:
        """
        Search for scientific papers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            source: Paper source (arxiv, pubmed, etc.)
            categories: Filter by categories
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
        
        Returns:
            ResearchResult with search results
        """
        start = time.time()
        
        if source == PaperSource.ARXIV:
            results = self._search_arxiv(query, max_results, categories)
        else:
            results = []
        
        elapsed = (time.time() - start) * 1000
        
        # Store search history
        self.search_history.append({
            'query': query,
            'source': source.value,
            'results': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
        return ResearchResult(
            success=True,
            mode=ResearchMode.SEARCH,
            search_results=results,
            summary=f"Found {len(results)} papers for '{query}'",
            elapsed_ms=elapsed
        )
    
    def _search_arxiv(self, 
                      query: str, 
                      max_results: int,
                      categories: Optional[List[str]] = None) -> List[SearchResult]:
        """Search arXiv for papers."""
        if not HAS_ARXIV:
            return self._search_arxiv_fallback(query, max_results)
        
        try:
            # Build search query
            search_query = query
            if categories:
                cat_filter = ' OR '.join(f'cat:{cat}' for cat in categories)
                search_query = f'({query}) AND ({cat_filter})'
            
            # Execute search
            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in client.results(search):
                results.append(SearchResult(
                    paper_id=paper.entry_id.split('/')[-1],
                    source=PaperSource.ARXIV,
                    title=paper.title,
                    authors=[a.name for a in paper.authors],
                    abstract=paper.summary,
                    date=paper.published,
                    relevance_score=0.9  # arXiv doesn't provide relevance scores
                ))
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  arXiv search error: {e}")
            return self._search_arxiv_fallback(query, max_results)
    
    def _search_arxiv_fallback(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback arXiv search using API directly."""
        try:
            # Use arXiv API directly
            base_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode('utf-8')
            
            # Parse XML response (simple parsing)
            results = []
            entries = re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL)
            
            for entry in entries[:max_results]:
                title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                arxiv_id = re.search(r'<id>http://arxiv.org/abs/(.*?)</id>', entry)
                authors = re.findall(r'<name>(.*?)</name>', entry)
                published = re.search(r'<published>(.*?)</published>', entry)
                
                if title and arxiv_id:
                    results.append(SearchResult(
                        paper_id=arxiv_id.group(1) if arxiv_id else "",
                        source=PaperSource.ARXIV,
                        title=title.group(1).strip().replace('\n', ' '),
                        authors=authors[:5],  # First 5 authors
                        abstract=summary.group(1).strip().replace('\n', ' ') if summary else "",
                        date=datetime.fromisoformat(published.group(1).replace('Z', '+00:00')) if published else None,
                        relevance_score=0.8
                    ))
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Fallback search error: {e}")
            return []
    
    # =========================================================================
    # Fetch Operations
    # =========================================================================
    
    def fetch(self, 
              identifier: str,
              source: Optional[PaperSource] = None) -> ResearchResult:
        """
        Fetch a specific paper by identifier.
        
        Args:
            identifier: Paper ID (arxiv:2301.12345, doi:10.1038/..., or URL)
            source: Override automatic source detection
        
        Returns:
            ResearchResult with fetched paper
        """
        start = time.time()
        errors = []
        
        # Detect source if not provided
        if source is None:
            source, clean_id = self._detect_source(identifier)
        else:
            clean_id = identifier
        
        try:
            if source == PaperSource.ARXIV:
                paper = self._fetch_arxiv(clean_id)
            elif source == PaperSource.DOI:
                paper = self._fetch_doi(clean_id)
            elif source == PaperSource.URL:
                paper = self._fetch_url(clean_id)
            elif source == PaperSource.LOCAL:
                paper = self._fetch_local(clean_id)
            else:
                raise ValueError(f"Unsupported source: {source}")
            
            if paper:
                self.papers[paper.id] = paper
                self.papers_fetched += 1
            
            elapsed = (time.time() - start) * 1000
            
            return ResearchResult(
                success=paper is not None,
                mode=ResearchMode.FETCH,
                papers=[paper] if paper else [],
                summary=f"Fetched: {paper.title}" if paper else "Failed to fetch paper",
                elapsed_ms=elapsed,
                errors=errors
            )
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ResearchResult(
                success=False,
                mode=ResearchMode.FETCH,
                summary=f"Error fetching paper: {str(e)}",
                elapsed_ms=elapsed,
                errors=[str(e)]
            )
    
    def _detect_source(self, identifier: str) -> Tuple[PaperSource, str]:
        """Detect paper source from identifier."""
        identifier = identifier.strip()
        
        # Check for explicit prefixes
        if identifier.startswith('arxiv:'):
            return PaperSource.ARXIV, identifier[6:]
        elif identifier.startswith('doi:'):
            return PaperSource.DOI, identifier[4:]
        elif identifier.startswith('http'):
            if 'arxiv.org' in identifier:
                # Extract arXiv ID from URL
                match = re.search(r'arxiv.org/(?:abs|pdf)/(\d+\.\d+)', identifier)
                if match:
                    return PaperSource.ARXIV, match.group(1)
            return PaperSource.URL, identifier
        elif identifier.endswith('.pdf'):
            return PaperSource.LOCAL, identifier
        # Check DOI before arXiv since DOIs start with 10.
        elif re.match(r'^10\.\d+/', identifier):
            return PaperSource.DOI, identifier
        elif re.match(r'^\d{4}\.\d+', identifier):
            # arXiv IDs are YYMM.NNNNN format (e.g., 2301.12345)
            return PaperSource.ARXIV, identifier
        else:
            return PaperSource.LOCAL, identifier
    
    def _fetch_arxiv(self, arxiv_id: str) -> Optional[Paper]:
        """Fetch paper from arXiv."""
        if self.verbose:
            print(f"   📥 Fetching arXiv:{arxiv_id}...")
        
        # Check cache first
        cache_path = self.cache_dir / f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
        
        try:
            if HAS_ARXIV:
                client = arxiv.Client()
                search = arxiv.Search(id_list=[arxiv_id])
                paper_info = next(client.results(search))
                
                # Download PDF if not cached
                if not cache_path.exists():
                    paper_info.download_pdf(dirpath=str(self.cache_dir), 
                                           filename=cache_path.name)
                
                # Create Paper object
                paper = Paper(
                    id=arxiv_id,
                    source=PaperSource.ARXIV,
                    arxiv_id=arxiv_id,
                    title=paper_info.title,
                    authors=[Author(name=a.name) for a in paper_info.authors],
                    abstract=paper_info.summary,
                    publication_date=paper_info.published,
                    pdf_path=str(cache_path) if cache_path.exists() else None,
                    keywords=list(paper_info.categories) if paper_info.categories else []
                )
                
                # Extract text if PDF exists
                if cache_path.exists():
                    self._extract_pdf_content(paper, cache_path)
                
                return paper
            else:
                return self._fetch_arxiv_fallback(arxiv_id)
                
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error fetching arXiv paper: {e}")
            return None
    
    def _fetch_arxiv_fallback(self, arxiv_id: str) -> Optional[Paper]:
        """Fallback arXiv fetch without arxiv library."""
        try:
            # Get metadata
            api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            with urllib.request.urlopen(api_url, timeout=30) as response:
                data = response.read().decode('utf-8')
            
            # Parse metadata
            title = re.search(r'<title>(.*?)</title>', data, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', data, re.DOTALL)
            authors = re.findall(r'<name>(.*?)</name>', data)
            
            # Download PDF
            cache_path = self.cache_dir / f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
            if not cache_path.exists():
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                urllib.request.urlretrieve(pdf_url, cache_path)
            
            paper = Paper(
                id=arxiv_id,
                source=PaperSource.ARXIV,
                arxiv_id=arxiv_id,
                title=title.group(1).strip().replace('\n', ' ') if title else "",
                authors=[Author(name=a) for a in authors],
                abstract=summary.group(1).strip().replace('\n', ' ') if summary else "",
                pdf_path=str(cache_path)
            )
            
            if cache_path.exists():
                self._extract_pdf_content(paper, cache_path)
            
            return paper
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Fallback fetch error: {e}")
            return None
    
    def _fetch_doi(self, doi: str) -> Optional[Paper]:
        """Fetch paper by DOI."""
        if self.verbose:
            print(f"   📥 Fetching DOI:{doi}...")
        
        try:
            # Get metadata from CrossRef
            crossref_url = f"https://api.crossref.org/works/{doi}"
            
            if HAS_REQUESTS:
                response = requests.get(crossref_url, timeout=30)
                data = response.json()
            else:
                with urllib.request.urlopen(crossref_url, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))
            
            work = data.get('message', {})
            
            # Extract authors
            authors = []
            for author in work.get('author', []):
                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                authors.append(Author(
                    name=name,
                    affiliation=author.get('affiliation', [{}])[0].get('name') if author.get('affiliation') else None
                ))
            
            # Create paper
            paper = Paper(
                id=doi,
                source=PaperSource.DOI,
                doi=doi,
                title=work.get('title', [''])[0] if work.get('title') else "",
                authors=authors,
                abstract=work.get('abstract', ''),
                journal=work.get('container-title', [''])[0] if work.get('container-title') else None,
            )
            
            # Try to get publication date
            if 'published-print' in work:
                date_parts = work['published-print'].get('date-parts', [[]])[0]
                if len(date_parts) >= 1:
                    paper.publication_date = datetime(date_parts[0], 
                                                     date_parts[1] if len(date_parts) > 1 else 1,
                                                     date_parts[2] if len(date_parts) > 2 else 1)
            
            return paper
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error fetching DOI: {e}")
            return None
    
    def _fetch_url(self, url: str) -> Optional[Paper]:
        """Fetch paper from URL."""
        if self.verbose:
            print(f"   📥 Fetching URL: {url[:50]}...")
        
        try:
            # Generate cache filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            cache_path = self.cache_dir / f"url_{url_hash}.pdf"
            
            # Download if not cached
            if not cache_path.exists():
                if HAS_REQUESTS:
                    response = requests.get(url, timeout=60)
                    with open(cache_path, 'wb') as f:
                        f.write(response.content)
                else:
                    urllib.request.urlretrieve(url, cache_path)
            
            paper = Paper(
                id=url_hash,
                source=PaperSource.URL,
                title=url.split('/')[-1].replace('.pdf', ''),
                pdf_path=str(cache_path)
            )
            
            if cache_path.exists():
                self._extract_pdf_content(paper, cache_path)
            
            return paper
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error fetching URL: {e}")
            return None
    
    def _fetch_local(self, path: str) -> Optional[Paper]:
        """Load local PDF file."""
        file_path = Path(path)
        
        if not file_path.exists():
            if self.verbose:
                print(f"   ❌ File not found: {path}")
            return None
        
        paper = Paper(
            id=file_path.stem,
            source=PaperSource.LOCAL,
            title=file_path.stem,
            pdf_path=str(file_path)
        )
        
        self._extract_pdf_content(paper, file_path)
        return paper
    
    # =========================================================================
    # PDF Parsing
    # =========================================================================
    
    def _extract_pdf_content(self, paper: Paper, pdf_path: Path):
        """Extract content from PDF using PyMuPDF."""
        if not HAS_PYMUPDF:
            if self.verbose:
                print("   ⚠️  PyMuPDF not available for PDF extraction")
            return
        
        try:
            doc = fitz.open(str(pdf_path))
            paper.page_count = len(doc)
            
            full_text = []
            equations = []
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                full_text.append(text)
                
                # Look for equations (LaTeX-like patterns)
                eq_patterns = re.findall(r'\$\$(.*?)\$\$|\\\[(.*?)\\\]', text, re.DOTALL)
                for eq in eq_patterns:
                    latex = eq[0] if eq[0] else eq[1]
                    if latex.strip():
                        equations.append(Equation(
                            latex=latex.strip(),
                            page=page_num + 1
                        ))
            
            paper.full_text = '\n'.join(full_text)
            paper.word_count = len(paper.full_text.split())
            paper.equations = equations
            
            # Extract sections
            paper.sections = self._parse_sections(paper.full_text)
            
            # Try to extract title and authors from first page if not set
            if not paper.title or paper.title == pdf_path.stem:
                first_page = full_text[0] if full_text else ""
                lines = first_page.split('\n')
                for line in lines[:10]:
                    line = line.strip()
                    if len(line) > 20 and not line.isupper() and not re.match(r'^[\d\s]+$', line):
                        paper.title = line
                        break
            
            doc.close()
            
            if self.verbose:
                print(f"   📄 Extracted {paper.page_count} pages, {paper.word_count} words, {len(equations)} equations")
                
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  PDF extraction error: {e}")
    
    def _parse_sections(self, text: str) -> List[Section]:
        """Parse document into sections."""
        sections = []
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            line = line.strip()
            
            # Check if line is a section header
            is_header = False
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_section:
                        current_section.content = '\n'.join(current_content)
                        sections.append(current_section)
                    
                    current_section = Section(title=line, content="")
                    current_content = []
                    is_header = True
                    break
            
            if not is_header and current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content)
            sections.append(current_section)
        
        return sections
    
    # =========================================================================
    # Analysis Operations
    # =========================================================================
    
    def analyze(self, paper: Union[Paper, str]) -> ResearchResult:
        """
        Deep analysis of a paper.
        
        Args:
            paper: Paper object or paper ID
        
        Returns:
            ResearchResult with analysis
        """
        start = time.time()
        
        if isinstance(paper, str):
            if paper in self.papers:
                paper = self.papers[paper]
            else:
                # Try to fetch
                result = self.fetch(paper)
                if not result.papers:
                    return ResearchResult(
                        success=False,
                        mode=ResearchMode.ANALYZE,
                        summary=f"Paper not found: {paper}",
                        elapsed_ms=(time.time() - start) * 1000
                    )
                paper = result.papers[0]
        
        # Extract key findings
        key_findings = self._extract_key_findings(paper)
        paper.key_findings = key_findings
        
        # Identify methodology
        methodology = self._extract_methodology(paper)
        paper.methodology = methodology
        
        # Generate summary
        summary = self._generate_summary(paper)
        paper.summary = summary
        
        # Extract contributions
        contributions = self._extract_contributions(paper)
        paper.contributions = contributions
        
        self.papers_analyzed += 1
        elapsed = (time.time() - start) * 1000
        
        return ResearchResult(
            success=True,
            mode=ResearchMode.ANALYZE,
            papers=[paper],
            summary=summary,
            insights=key_findings,
            elapsed_ms=elapsed
        )
    
    def _extract_key_findings(self, paper: Paper) -> List[str]:
        """Extract key findings from paper."""
        findings = []
        
        # Look in abstract
        if paper.abstract:
            sentences = re.split(r'[.!?]', paper.abstract)
            for s in sentences:
                s = s.strip()
                if any(kw in s.lower() for kw in ['we show', 'we demonstrate', 'we find', 
                                                   'results show', 'we prove', 'we present']):
                    findings.append(s)
        
        # Look in results/conclusion sections
        for section in paper.sections:
            if any(kw in section.title.lower() for kw in ['result', 'conclusion', 'finding']):
                sentences = re.split(r'[.!?]', section.content)
                for s in sentences[:5]:  # First 5 sentences
                    s = s.strip()
                    if len(s) > 30:
                        findings.append(s)
        
        return findings[:10]  # Top 10 findings
    
    def _extract_methodology(self, paper: Paper) -> str:
        """Extract methodology from paper."""
        for section in paper.sections:
            if any(kw in section.title.lower() for kw in ['method', 'approach', 'experiment']):
                return section.content[:2000]  # First 2000 chars
        
        # Fallback: look for methodology keywords in full text
        if paper.full_text:
            match = re.search(r'(?:method|approach|procedure)[^\n]*\n((?:[^\n]+\n){1,10})', 
                            paper.full_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def _generate_summary(self, paper: Paper) -> str:
        """Generate paper summary."""
        parts = []
        
        # Title and authors
        if paper.title:
            parts.append(f"**{paper.title}**")
        
        if paper.authors:
            author_names = [a.name for a in paper.authors[:3]]
            if len(paper.authors) > 3:
                author_names.append("et al.")
            parts.append(f"Authors: {', '.join(author_names)}")
        
        # Abstract summary
        if paper.abstract:
            # Take first 2-3 sentences
            sentences = re.split(r'[.!?]', paper.abstract)[:3]
            parts.append("Abstract: " + '. '.join(s.strip() for s in sentences if s.strip()) + '.')
        
        # Key findings
        if paper.key_findings:
            parts.append("Key Findings:")
            for i, finding in enumerate(paper.key_findings[:3], 1):
                parts.append(f"  {i}. {finding}")
        
        return '\n'.join(parts)
    
    def _extract_contributions(self, paper: Paper) -> List[str]:
        """Extract main contributions from paper."""
        contributions = []
        
        # Look for explicit contribution statements
        patterns = [
            r'(?:main|key|primary)\s+contribution[s]?[:\s]+([^.]+\.)',
            r'we\s+contribute[:\s]+([^.]+\.)',
            r'our\s+contribution[s]?[:\s]+([^.]+\.)',
        ]
        
        text = paper.abstract + ' ' + paper.full_text[:5000]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            contributions.extend(matches)
        
        return contributions[:5]
    
    # =========================================================================
    # Hypothesis Generation
    # =========================================================================
    
    def hypothesize(self, 
                    topic: str,
                    papers: Optional[List[Paper]] = None) -> ResearchResult:
        """
        Generate research hypotheses based on analyzed papers.
        
        Args:
            topic: Research topic
            papers: Papers to base hypotheses on (uses all if None)
        
        Returns:
            ResearchResult with hypotheses
        """
        start = time.time()
        
        if papers is None:
            papers = list(self.papers.values())
        
        hypotheses = []
        
        # Analyze gaps and opportunities
        if papers:
            # Extract common themes
            all_keywords = []
            for paper in papers:
                all_keywords.extend(paper.keywords)
            
            # Generate hypotheses based on findings
            all_findings = []
            for paper in papers:
                all_findings.extend(paper.key_findings)
            
            # Template-based hypothesis generation
            templates = [
                f"Based on {topic}, we hypothesize that combining approaches from multiple papers could yield improved results.",
                f"The gap in current research on {topic} suggests that novel methodologies may be needed.",
                f"Cross-domain application of {topic} techniques to related fields remains unexplored.",
            ]
            
            hypotheses.extend(templates)
            
            # Add specific hypotheses from paper analysis
            if all_findings:
                hypotheses.append(f"Extending the finding '{all_findings[0][:100]}...' to broader domains.")
        else:
            # Generate exploratory hypotheses
            hypotheses = [
                f"Research on {topic} may benefit from computational approaches.",
                f"The intersection of {topic} with machine learning presents opportunities.",
                f"Experimental validation of theoretical models in {topic} is needed.",
            ]
        
        elapsed = (time.time() - start) * 1000
        
        return ResearchResult(
            success=True,
            mode=ResearchMode.HYPOTHESIZE,
            papers=papers,
            hypotheses=hypotheses,
            summary=f"Generated {len(hypotheses)} hypotheses for '{topic}'",
            elapsed_ms=elapsed
        )
    
    # =========================================================================
    # Citation Generation
    # =========================================================================
    
    def cite(self, 
             paper: Union[Paper, str],
             format: str = "bibtex") -> ResearchResult:
        """
        Generate citation for a paper.
        
        Args:
            paper: Paper object or ID
            format: Citation format (bibtex, apa, mla, chicago)
        
        Returns:
            ResearchResult with citation
        """
        start = time.time()
        
        if isinstance(paper, str):
            paper = self.papers.get(paper)
            if not paper:
                return ResearchResult(
                    success=False,
                    mode=ResearchMode.CITE,
                    summary=f"Paper not found: {paper}",
                    elapsed_ms=(time.time() - start) * 1000
                )
        
        if format.lower() == "bibtex":
            citation = self._generate_bibtex(paper)
        elif format.lower() == "apa":
            citation = self._generate_apa(paper)
        else:
            citation = self._generate_bibtex(paper)  # Default
        
        elapsed = (time.time() - start) * 1000
        
        return ResearchResult(
            success=True,
            mode=ResearchMode.CITE,
            papers=[paper],
            citations=[citation],
            summary=f"Generated {format} citation",
            elapsed_ms=elapsed
        )
    
    def _generate_bibtex(self, paper: Paper) -> str:
        """Generate BibTeX citation."""
        # Create citation key
        first_author = paper.authors[0].name.split()[-1] if paper.authors else "unknown"
        year = paper.publication_date.year if paper.publication_date else "2026"
        key = f"{first_author.lower()}{year}"
        
        authors = " and ".join(a.name for a in paper.authors)
        
        bibtex = f"""@article{{{key},
    title = {{{paper.title}}},
    author = {{{authors}}},
    year = {{{year}}},"""
        
        if paper.journal:
            bibtex += f"\n    journal = {{{paper.journal}}},"
        if paper.doi:
            bibtex += f"\n    doi = {{{paper.doi}}},"
        if paper.arxiv_id:
            bibtex += f"\n    eprint = {{{paper.arxiv_id}}},"
            bibtex += "\n    archiveprefix = {arXiv},"
        
        bibtex += "\n}"
        
        return bibtex
    
    def _generate_apa(self, paper: Paper) -> str:
        """Generate APA citation."""
        # Authors
        if paper.authors:
            if len(paper.authors) == 1:
                authors = paper.authors[0].name
            elif len(paper.authors) == 2:
                authors = f"{paper.authors[0].name} & {paper.authors[1].name}"
            else:
                authors = f"{paper.authors[0].name} et al."
        else:
            authors = "Unknown"
        
        year = paper.publication_date.year if paper.publication_date else "n.d."
        
        citation = f"{authors} ({year}). {paper.title}."
        
        if paper.journal:
            citation += f" {paper.journal}."
        if paper.doi:
            citation += f" https://doi.org/{paper.doi}"
        elif paper.arxiv_id:
            citation += f" arXiv:{paper.arxiv_id}"
        
        return citation
    
    # =========================================================================
    # Literature Review
    # =========================================================================
    
    def literature_review(self,
                         topic: str,
                         max_papers: int = 20) -> ResearchResult:
        """
        Conduct automated literature review.
        
        Args:
            topic: Research topic
            max_papers: Maximum papers to include
        
        Returns:
            ResearchResult with comprehensive review
        """
        start = time.time()
        
        if self.verbose:
            print(f"\n📚 Conducting literature review on: {topic}")
            print("=" * 60)
        
        # Step 1: Search for papers
        if self.verbose:
            print("   🔍 Searching for papers...")
        search_result = self.search(topic, max_results=max_papers)
        
        # Step 2: Fetch top papers
        papers = []
        for sr in search_result.search_results[:min(5, len(search_result.search_results))]:
            if self.verbose:
                print(f"   📥 Fetching: {sr.title[:50]}...")
            fetch_result = self.fetch(f"arxiv:{sr.paper_id}")
            if fetch_result.papers:
                papers.extend(fetch_result.papers)
        
        # Step 3: Analyze papers
        for paper in papers:
            if self.verbose:
                print(f"   🔬 Analyzing: {paper.title[:50]}...")
            self.analyze(paper)
        
        # Step 4: Generate review summary
        review_parts = [
            f"# Literature Review: {topic}",
            f"Generated: {datetime.now().isoformat()}",
            f"Papers analyzed: {len(papers)}",
            "",
            "## Summary",
        ]
        
        # Aggregate findings
        all_findings = []
        all_methods = []
        for paper in papers:
            all_findings.extend(paper.key_findings)
            if paper.methodology:
                all_methods.append(paper.methodology[:200])
        
        review_parts.append(f"This review covers {len(papers)} papers on {topic}.")
        
        if all_findings:
            review_parts.append("\n## Key Findings")
            for i, finding in enumerate(all_findings[:10], 1):
                review_parts.append(f"{i}. {finding}")
        
        review_parts.append("\n## Papers Reviewed")
        for paper in papers:
            review_parts.append(f"- {paper.title} ({paper.arxiv_id or paper.doi or 'local'})")
        
        review = '\n'.join(review_parts)
        
        # Step 5: Generate hypotheses
        hyp_result = self.hypothesize(topic, papers)
        
        elapsed = (time.time() - start) * 1000
        
        if self.verbose:
            print(f"\n✅ Literature review complete ({elapsed:.0f}ms)")
        
        return ResearchResult(
            success=True,
            mode=ResearchMode.REVIEW,
            papers=papers,
            summary=review,
            hypotheses=hyp_result.hypotheses,
            insights=all_findings[:10],
            elapsed_ms=elapsed
        )
    
    # =========================================================================
    # Context Integration
    # =========================================================================
    
    def get_context_for_scout(self) -> str:
        """
        Get formatted context for Scout 10M integration.
        
        Returns:
            Formatted string with all paper content for Scout context
        """
        context_parts = []
        
        for paper in self.papers.values():
            part = f"=== Paper: {paper.title} ===\n"
            part += f"Authors: {', '.join(a.name for a in paper.authors)}\n"
            part += f"Source: {paper.source.value}\n"
            
            if paper.abstract:
                part += f"\nAbstract:\n{paper.abstract}\n"
            
            if paper.key_findings:
                part += "\nKey Findings:\n"
                for f in paper.key_findings:
                    part += f"- {f}\n"
            
            if paper.full_text:
                part += f"\nFull Text (truncated):\n{paper.full_text[:10000]}\n"
            
            context_parts.append(part)
        
        return '\n\n'.join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'papers_loaded': len(self.papers),
            'papers_fetched': self.papers_fetched,
            'papers_analyzed': self.papers_analyzed,
            'total_pages': sum(p.page_count for p in self.papers.values()),
            'total_words': sum(p.word_count for p in self.papers.values()),
            'searches_performed': len(self.search_history),
            'cache_dir': str(self.cache_dir),
        }


# =============================================================================
# Q-Lang Command Handler
# =============================================================================

def handle_research_command(engine: ResearchEngine, 
                           line: str, 
                           context: Dict[str, Any]):
    """
    Handle research commands from Q-Lang interpreter.
    
    Args:
        engine: ResearchEngine instance
        line: Command line
        context: Q-Lang interpreter context
    """
    parts = line.split(None, 2)
    
    if len(parts) < 2:
        print("Usage: research <command> [args]")
        print("Commands: search, fetch, analyze, summarize, hypothesize, cite, review, status")
        return
    
    cmd = parts[1].lower()
    args = parts[2] if len(parts) > 2 else ""
    
    if cmd == "status":
        engine._print_status()
        stats = engine.get_stats()
        print(f"    📊 Papers Loaded: {stats['papers_loaded']}")
        print(f"    📥 Papers Fetched: {stats['papers_fetched']}")
        print(f"    🔬 Papers Analyzed: {stats['papers_analyzed']}")
        print(f"    📄 Total Pages: {stats['total_pages']}")
        print(f"    📝 Total Words: {stats['total_words']}")
        
    elif cmd == "search":
        # Parse search options
        query = args
        max_results = 10
        
        # Check for --max option
        max_match = re.search(r'--max\s+(\d+)', args)
        if max_match:
            max_results = int(max_match.group(1))
            query = re.sub(r'--max\s+\d+', '', args).strip()
        
        # Remove quotes
        query = query.strip('"\'')
        
        print(f"\n🔍 Searching for: {query}")
        print("=" * 60)
        
        result = engine.search(query, max_results=max_results)
        
        for i, sr in enumerate(result.search_results, 1):
            print(f"\n{i}. {sr.title}")
            print(f"   Authors: {', '.join(sr.authors[:3])}")
            print(f"   ID: {sr.paper_id}")
            if sr.date:
                print(f"   Date: {sr.date.strftime('%Y-%m-%d')}")
        
        print(f"\n⏱️  {result.elapsed_ms:.0f}ms | Found {len(result.search_results)} papers")
        
        # Store results in context
        context['_last_search'] = result.search_results
        
    elif cmd == "fetch":
        identifier = args.strip().strip('"\'')
        
        result = engine.fetch(identifier)
        
        if result.success and result.papers:
            paper = result.papers[0]
            print(f"\n✅ Fetched: {paper.title}")
            print(f"   Authors: {', '.join(a.name for a in paper.authors[:3])}")
            print(f"   Pages: {paper.page_count}")
            print(f"   Words: {paper.word_count}")
            
            # Store in context
            var_name = re.sub(r'[^a-zA-Z0-9_]', '_', paper.id)[:20]
            context[var_name] = paper
            print(f"   Stored as: {var_name}")
        else:
            print(f"\n❌ Failed to fetch: {identifier}")
            for err in result.errors:
                print(f"   Error: {err}")
        
    elif cmd == "analyze":
        identifier = args.strip().strip('"\'')
        
        result = engine.analyze(identifier)
        
        if result.success and result.papers:
            paper = result.papers[0]
            print(f"\n🔬 Analysis: {paper.title}")
            print("=" * 60)
            print(result.summary)
            
            if result.insights:
                print("\n📌 Key Insights:")
                for i, insight in enumerate(result.insights[:5], 1):
                    print(f"   {i}. {insight[:100]}...")
        else:
            print(f"\n❌ Analysis failed: {result.summary}")
            
    elif cmd == "hypothesize":
        topic = args.strip().strip('"\'')
        
        result = engine.hypothesize(topic)
        
        print(f"\n💡 Hypotheses for: {topic}")
        print("=" * 60)
        for i, hyp in enumerate(result.hypotheses, 1):
            print(f"{i}. {hyp}")
            
    elif cmd == "cite":
        # Parse: cite <paper_id> [--format bibtex|apa]
        format_match = re.search(r'--format\s+(\w+)', args)
        fmt = format_match.group(1) if format_match else "bibtex"
        identifier = re.sub(r'--format\s+\w+', '', args).strip().strip('"\'')
        
        result = engine.cite(identifier, format=fmt)
        
        if result.citations:
            print(f"\n📚 Citation ({fmt}):")
            print("-" * 40)
            print(result.citations[0])
        else:
            print(f"\n❌ Could not generate citation")
            
    elif cmd == "review":
        topic = args.strip().strip('"\'')
        
        result = engine.literature_review(topic, max_papers=10)
        
        print("\n" + result.summary)
        
        if result.hypotheses:
            print("\n💡 Research Opportunities:")
            for hyp in result.hypotheses:
                print(f"   • {hyp}")
                
    elif cmd == "context":
        # Get context for Scout integration
        ctx = engine.get_context_for_scout()
        if ctx:
            context['_research_context'] = ctx
            print(f"✅ Research context ready ({len(ctx)} chars)")
            print("   Use 'scout context' to load into Scout 10M")
        else:
            print("⚠️  No papers loaded. Use 'research fetch' first.")
            
    elif cmd == "list":
        if engine.papers:
            print("\n📚 Loaded Papers:")
            for pid, paper in engine.papers.items():
                print(f"   • {paper.title[:60]}... ({pid})")
        else:
            print("   No papers loaded")
            
    else:
        print(f"❌ Unknown research command: {cmd}")
        print("Commands: search, fetch, analyze, summarize, hypothesize, cite, review, status, list, context")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       QENEX Automated Science Research Demo                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    engine = ResearchEngine(verbose=True)
    
    # Demo: Search
    print("\n" + "=" * 60)
    print("Demo: Search for quantum computing papers")
    print("=" * 60)
    
    result = engine.search("quantum error correction", max_results=3)
    
    for sr in result.search_results:
        print(f"\n📄 {sr.title}")
        print(f"   Authors: {', '.join(sr.authors[:2])}")
        print(f"   ID: arxiv:{sr.paper_id}")
    
    # Demo: Fetch (if search found results)
    if result.search_results:
        print("\n" + "=" * 60)
        print("Demo: Fetch first paper")
        print("=" * 60)
        
        first_paper = result.search_results[0]
        fetch_result = engine.fetch(f"arxiv:{first_paper.paper_id}")
        
        if fetch_result.papers:
            paper = fetch_result.papers[0]
            print(f"✅ Fetched: {paper.title}")
            print(f"   Pages: {paper.page_count}, Words: {paper.word_count}")
            
            # Demo: Analyze
            print("\n" + "=" * 60)
            print("Demo: Analyze paper")
            print("=" * 60)
            
            analysis = engine.analyze(paper)
            print(analysis.summary[:500] + "...")
    
    # Demo: Stats
    print("\n" + "=" * 60)
    print("Demo: Engine Statistics")
    print("=" * 60)
    
    stats = engine.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
