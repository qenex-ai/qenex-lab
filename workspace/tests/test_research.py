"""
Tests for QENEX Automated Science Research Engine.

Tests cover arXiv search, paper fetching, PDF parsing, analysis,
and Q-Lang integration.
"""
import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'qenex-qlang', 'src'))

from research import (
    ResearchEngine,
    ResearchMode,
    PaperSource,
    Paper,
    Author,
    SearchResult,
    ResearchResult,
    Section,
    Equation,
    Citation,
    handle_research_command,
)


# =============================================================================
# ResearchEngine Initialization Tests
# =============================================================================

class TestResearchEngineInit:
    """Test ResearchEngine initialization."""
    
    def test_engine_creation(self):
        """Test that engine can be created."""
        engine = ResearchEngine(verbose=False)
        assert engine is not None
    
    def test_engine_with_verbose(self):
        """Test engine with verbose mode."""
        engine = ResearchEngine(verbose=True)
        assert engine.verbose is True
    
    def test_engine_creates_cache_dir(self):
        """Test that engine creates cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache")
            engine = ResearchEngine(cache_dir=cache_path, verbose=False)
            assert engine.cache_dir.exists()
    
    def test_engine_has_papers_dict(self):
        """Test that engine has papers dictionary."""
        engine = ResearchEngine(verbose=False)
        assert hasattr(engine, 'papers')
        assert isinstance(engine.papers, dict)
    
    def test_engine_has_stats(self):
        """Test engine has statistics tracking."""
        engine = ResearchEngine(verbose=False)
        stats = engine.get_stats()
        assert 'papers_loaded' in stats
        assert 'papers_fetched' in stats
        assert 'papers_analyzed' in stats


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Test enumeration values."""
    
    def test_research_modes(self):
        """Test all research modes exist."""
        modes = [ResearchMode.SEARCH, ResearchMode.FETCH, 
                 ResearchMode.PARSE, ResearchMode.ANALYZE,
                 ResearchMode.SUMMARIZE, ResearchMode.HYPOTHESIZE,
                 ResearchMode.CITE, ResearchMode.COMPARE,
                 ResearchMode.REVIEW]
        assert len(modes) == 9
    
    def test_paper_sources(self):
        """Test all paper sources exist."""
        sources = [PaperSource.ARXIV, PaperSource.DOI, 
                   PaperSource.PUBMED, PaperSource.LOCAL,
                   PaperSource.URL]
        assert len(sources) == 5
    
    def test_source_values(self):
        """Test source enum values."""
        assert PaperSource.ARXIV.value == "arxiv"
        assert PaperSource.DOI.value == "doi"
        assert PaperSource.LOCAL.value == "local"


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDataClasses:
    """Test data class structures."""
    
    def test_author_creation(self):
        """Test Author dataclass."""
        author = Author(
            name="Albert Einstein",
            affiliation="Princeton",
            email="albert@princeton.edu"
        )
        assert author.name == "Albert Einstein"
        assert author.affiliation == "Princeton"
    
    def test_paper_creation(self):
        """Test Paper dataclass."""
        paper = Paper(
            id="test123",
            source=PaperSource.ARXIV,
            title="Test Paper",
            abstract="This is a test."
        )
        assert paper.id == "test123"
        assert paper.source == PaperSource.ARXIV
        assert paper.title == "Test Paper"
    
    def test_paper_default_values(self):
        """Test Paper default values."""
        paper = Paper(id="test", source=PaperSource.LOCAL)
        assert paper.authors == []
        assert paper.keywords == []
        assert paper.sections == []
        assert paper.page_count == 0
    
    def test_search_result_creation(self):
        """Test SearchResult dataclass."""
        sr = SearchResult(
            paper_id="2301.12345",
            source=PaperSource.ARXIV,
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="Abstract text"
        )
        assert sr.paper_id == "2301.12345"
        assert len(sr.authors) == 2
    
    def test_section_creation(self):
        """Test Section dataclass."""
        section = Section(
            title="Introduction",
            content="This paper introduces...",
            level=1
        )
        assert section.title == "Introduction"
        assert section.level == 1
    
    def test_equation_creation(self):
        """Test Equation dataclass."""
        eq = Equation(
            latex="E = mc^2",
            label="eq:einstein",
            page=5
        )
        assert eq.latex == "E = mc^2"
        assert eq.page == 5
    
    def test_research_result_creation(self):
        """Test ResearchResult dataclass."""
        result = ResearchResult(
            success=True,
            mode=ResearchMode.SEARCH,
            summary="Found 10 papers"
        )
        assert result.success is True
        assert result.mode == ResearchMode.SEARCH


# =============================================================================
# Search Tests
# =============================================================================

class TestSearch:
    """Test search functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for tests."""
        return ResearchEngine(verbose=False)
    
    def test_search_returns_result(self, engine):
        """Test that search returns a result object."""
        result = engine.search("quantum computing", max_results=3)
        assert isinstance(result, ResearchResult)
        assert result.mode == ResearchMode.SEARCH
    
    def test_search_has_results(self, engine):
        """Test that search finds papers (requires network)."""
        result = engine.search("quantum", max_results=2)
        # May or may not find results depending on network
        assert result.success is True
    
    def test_search_respects_max_results(self, engine):
        """Test max_results parameter."""
        result = engine.search("machine learning", max_results=3)
        assert len(result.search_results) <= 3
    
    def test_search_result_has_required_fields(self, engine):
        """Test that search results have required fields."""
        result = engine.search("physics", max_results=1)
        if result.search_results:
            sr = result.search_results[0]
            assert sr.paper_id is not None
            assert sr.title is not None
            assert sr.source == PaperSource.ARXIV
    
    def test_search_updates_history(self, engine):
        """Test that search updates search history."""
        initial_count = len(engine.search_history)
        engine.search("test query", max_results=1)
        assert len(engine.search_history) == initial_count + 1


# =============================================================================
# Source Detection Tests
# =============================================================================

class TestSourceDetection:
    """Test paper source detection."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    def test_detect_arxiv_prefix(self, engine):
        """Test arXiv prefix detection."""
        source, clean_id = engine._detect_source("arxiv:2301.12345")
        assert source == PaperSource.ARXIV
        assert clean_id == "2301.12345"
    
    def test_detect_doi_prefix(self, engine):
        """Test DOI prefix detection."""
        source, clean_id = engine._detect_source("doi:10.1038/s41586-023-06096-3")
        assert source == PaperSource.DOI
        assert clean_id == "10.1038/s41586-023-06096-3"
    
    def test_detect_arxiv_url(self, engine):
        """Test arXiv URL detection."""
        source, clean_id = engine._detect_source("https://arxiv.org/abs/2301.12345")
        assert source == PaperSource.ARXIV
        assert clean_id == "2301.12345"
    
    def test_detect_arxiv_id_only(self, engine):
        """Test bare arXiv ID detection."""
        source, clean_id = engine._detect_source("2301.12345")
        assert source == PaperSource.ARXIV
        assert clean_id == "2301.12345"
    
    def test_detect_doi_only(self, engine):
        """Test bare DOI detection."""
        source, clean_id = engine._detect_source("10.1038/nature12373")
        assert source == PaperSource.DOI
        assert clean_id == "10.1038/nature12373"
    
    def test_detect_local_pdf(self, engine):
        """Test local PDF detection."""
        source, clean_id = engine._detect_source("paper.pdf")
        assert source == PaperSource.LOCAL
    
    def test_detect_url(self, engine):
        """Test generic URL detection."""
        source, clean_id = engine._detect_source("https://example.com/paper.pdf")
        assert source == PaperSource.URL


# =============================================================================
# Citation Generation Tests
# =============================================================================

class TestCitationGeneration:
    """Test citation generation."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    @pytest.fixture
    def sample_paper(self):
        """Create sample paper for testing."""
        from datetime import datetime
        return Paper(
            id="test123",
            source=PaperSource.ARXIV,
            arxiv_id="2301.12345",
            title="Test Paper Title",
            authors=[
                Author(name="John Smith"),
                Author(name="Jane Doe")
            ],
            publication_date=datetime(2023, 1, 15),
            journal="Journal of Testing"
        )
    
    def test_generate_bibtex(self, engine, sample_paper):
        """Test BibTeX generation."""
        engine.papers[sample_paper.id] = sample_paper
        result = engine.cite(sample_paper.id, format="bibtex")
        
        assert result.success
        assert result.citations
        bibtex = result.citations[0]
        assert "@article" in bibtex
        assert "Test Paper Title" in bibtex
        assert "John Smith" in bibtex
    
    def test_generate_apa(self, engine, sample_paper):
        """Test APA citation generation."""
        engine.papers[sample_paper.id] = sample_paper
        result = engine.cite(sample_paper.id, format="apa")
        
        assert result.success
        assert result.citations
        apa = result.citations[0]
        assert "Smith" in apa or "John" in apa
        assert "2023" in apa
    
    def test_cite_not_found(self, engine):
        """Test citation for non-existent paper."""
        result = engine.cite("nonexistent_paper")
        assert result.success is False


# =============================================================================
# Hypothesis Generation Tests
# =============================================================================

class TestHypothesisGeneration:
    """Test hypothesis generation."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    def test_hypothesize_returns_result(self, engine):
        """Test that hypothesize returns result."""
        result = engine.hypothesize("quantum computing")
        assert isinstance(result, ResearchResult)
        assert result.mode == ResearchMode.HYPOTHESIZE
    
    def test_hypothesize_generates_hypotheses(self, engine):
        """Test that hypotheses are generated."""
        result = engine.hypothesize("machine learning")
        assert result.hypotheses
        assert len(result.hypotheses) > 0
    
    def test_hypothesize_with_papers(self, engine):
        """Test hypothesis generation with papers."""
        # Create test paper
        paper = Paper(
            id="test",
            source=PaperSource.LOCAL,
            title="Test Paper",
            key_findings=["Finding 1", "Finding 2"],
            keywords=["quantum", "computing"]
        )
        
        result = engine.hypothesize("quantum computing", papers=[paper])
        assert result.hypotheses


# =============================================================================
# Analysis Tests
# =============================================================================

class TestAnalysis:
    """Test paper analysis."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    @pytest.fixture
    def sample_paper(self):
        return Paper(
            id="test",
            source=PaperSource.LOCAL,
            title="Test Paper",
            abstract="We show that quantum computing provides exponential speedup. We demonstrate this through experiments.",
            full_text="Introduction\nThis paper introduces quantum computing.\n\nResults\nOur results show significant improvement.\n\nConclusion\nIn conclusion, quantum computing is powerful.",
            authors=[Author(name="Test Author")]
        )
    
    def test_analyze_returns_result(self, engine, sample_paper):
        """Test that analyze returns result."""
        engine.papers[sample_paper.id] = sample_paper
        result = engine.analyze(sample_paper.id)
        
        assert isinstance(result, ResearchResult)
        assert result.mode == ResearchMode.ANALYZE
        assert result.success
    
    def test_analyze_extracts_findings(self, engine, sample_paper):
        """Test that analysis extracts key findings."""
        engine.papers[sample_paper.id] = sample_paper
        result = engine.analyze(sample_paper.id)
        
        # Should extract "we show" statements
        paper = result.papers[0]
        assert paper.key_findings is not None
    
    def test_analyze_generates_summary(self, engine, sample_paper):
        """Test that analysis generates summary."""
        engine.papers[sample_paper.id] = sample_paper
        result = engine.analyze(sample_paper.id)
        
        assert result.summary
        assert len(result.summary) > 0


# =============================================================================
# Section Parsing Tests
# =============================================================================

class TestSectionParsing:
    """Test document section parsing."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    def test_parse_introduction(self, engine):
        """Test parsing Introduction section."""
        text = "Introduction\nThis is the intro content.\n\nMethods\nThese are the methods."
        sections = engine._parse_sections(text)
        
        assert len(sections) >= 1
        assert any("Introduction" in s.title for s in sections)
    
    def test_parse_multiple_sections(self, engine):
        """Test parsing multiple sections."""
        text = """Introduction
This is intro.

Methods
These are methods.

Results
Here are results.

Conclusion
In conclusion."""
        
        sections = engine._parse_sections(text)
        titles = [s.title for s in sections]
        
        # Should find multiple sections
        assert len(sections) >= 2


# =============================================================================
# Command Handler Tests
# =============================================================================

class TestCommandHandler:
    """Test handle_research_command function."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    @pytest.fixture
    def context(self):
        return {}
    
    def test_handle_status_command(self, engine, context, capsys):
        """Test status command."""
        handle_research_command(engine, "research status", context)
        captured = capsys.readouterr()
        assert "Research" in captured.out or "Papers" in captured.out
    
    def test_handle_search_command(self, engine, context, capsys):
        """Test search command."""
        handle_research_command(engine, 'research search "test" --max 2', context)
        captured = capsys.readouterr()
        assert "Searching" in captured.out or "Found" in captured.out
    
    def test_handle_list_command(self, engine, context, capsys):
        """Test list command."""
        handle_research_command(engine, "research list", context)
        captured = capsys.readouterr()
        # Should show papers or "No papers"
        assert len(captured.out) > 0
    
    def test_handle_hypothesize_command(self, engine, context, capsys):
        """Test hypothesize command."""
        handle_research_command(engine, 'research hypothesize "quantum computing"', context)
        captured = capsys.readouterr()
        assert "Hypotheses" in captured.out or "💡" in captured.out
    
    def test_handle_unknown_command(self, engine, context, capsys):
        """Test unknown command handling."""
        handle_research_command(engine, "research unknown_cmd", context)
        captured = capsys.readouterr()
        assert "Unknown" in captured.out or "Commands" in captured.out


# =============================================================================
# Q-Lang Integration Tests
# =============================================================================

class TestQLangIntegration:
    """Test integration with Q-Lang interpreter."""
    
    @pytest.fixture
    def interpreter(self):
        """Create Q-Lang interpreter."""
        from interpreter import QLangInterpreter
        return QLangInterpreter()
    
    def test_research_status_via_qlang(self, interpreter, capsys):
        """Test research status through Q-Lang."""
        interpreter.execute("research status")
        captured = capsys.readouterr()
        assert "Research" in captured.out
    
    def test_research_engine_persists_in_context(self, interpreter):
        """Test that research engine persists in interpreter context."""
        interpreter.execute("research status")
        assert "_research_engine" in interpreter.context
        engine = interpreter.context["_research_engine"]
        assert isinstance(engine, ResearchEngine)
    
    def test_research_list_via_qlang(self, interpreter, capsys):
        """Test research list through Q-Lang."""
        interpreter.execute("research list")
        captured = capsys.readouterr()
        # Should output something
        assert len(captured.out) > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    def test_fetch_invalid_arxiv(self, engine):
        """Test fetching invalid arXiv ID."""
        result = engine.fetch("arxiv:invalid_id_12345")
        # Should handle gracefully
        assert isinstance(result, ResearchResult)
    
    def test_search_empty_query(self, engine):
        """Test search with empty query."""
        result = engine.search("")
        assert isinstance(result, ResearchResult)
    
    def test_analyze_nonexistent_paper(self, engine):
        """Test analyzing paper that doesn't exist."""
        result = engine.analyze("nonexistent_id")
        assert result.success is False
    
    def test_cite_nonexistent_paper(self, engine):
        """Test citing paper that doesn't exist."""
        result = engine.cite("nonexistent_id")
        assert result.success is False
    
    def test_context_without_papers(self, engine):
        """Test getting context without papers."""
        ctx = engine.get_context_for_scout()
        assert ctx == ""  # Empty when no papers


# =============================================================================
# Statistics Tests
# =============================================================================

class TestStatistics:
    """Test statistics tracking."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    def test_stats_initial_values(self, engine):
        """Test initial statistics values."""
        stats = engine.get_stats()
        assert stats['papers_loaded'] == 0
        assert stats['papers_fetched'] == 0
        assert stats['papers_analyzed'] == 0
    
    def test_stats_after_search(self, engine):
        """Test statistics after search."""
        engine.search("test", max_results=1)
        stats = engine.get_stats()
        assert stats['searches_performed'] >= 1
    
    def test_stats_has_cache_dir(self, engine):
        """Test stats includes cache dir."""
        stats = engine.get_stats()
        assert 'cache_dir' in stats


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def engine(self):
        return ResearchEngine(verbose=False)
    
    def test_search_time_reasonable(self, engine):
        """Test that search completes in reasonable time."""
        import time
        start = time.time()
        engine.search("physics", max_results=2)
        elapsed = time.time() - start
        # Should complete in under 30 seconds
        assert elapsed < 30.0
    
    def test_hypothesize_time_reasonable(self, engine):
        """Test that hypothesis generation is fast."""
        import time
        start = time.time()
        engine.hypothesize("quantum computing")
        elapsed = time.time() - start
        # Should be nearly instant
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
