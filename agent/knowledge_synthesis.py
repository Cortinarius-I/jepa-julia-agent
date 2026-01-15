"""
Knowledge Synthesis for Unfamiliar Packages.

Before planning, gather context about unfamiliar modules via documentation
retrieval. This enriches the world state representation with semantic
information that may not be captured by static code analysis.

Inspired by Agent2World (2025): "A Deep Researcher agent designed to gather
background knowledge and fill in missing details that are not explicitly
provided in the world model description. By leveraging external information
sources, this agent not only compensates for potential knowledge gaps
inherent in large language models but also enhances the factual reliability
of world model descriptions."

Sources:
1. Julia docstrings (extracted from source)
2. Package documentation (from Documenter.jl output)
3. README files
4. Julia discourse / forums (optional web search)
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge Types
# ---------------------------------------------------------------------------


@dataclass
class DocString:
    """Extracted docstring for a symbol."""
    symbol: str
    module: str
    docstring: str
    signature: Optional[str] = None
    examples: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    source_file: Optional[str] = None
    source_line: Optional[int] = None


@dataclass
class PackageInfo:
    """High-level package information."""
    name: str
    version: str
    description: str
    authors: list[str]
    dependencies: list[str]
    exports: list[str]
    readme_summary: str = ""
    repository_url: str = ""


@dataclass
class TypeInfo:
    """Information about a type/struct."""
    name: str
    module: str
    supertype: Optional[str] = None
    fields: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    docstring: str = ""
    constructors: list[str] = field(default_factory=list)


@dataclass
class MethodInfo:
    """Information about a method."""
    name: str
    module: str
    signatures: list[str] = field(default_factory=list)
    docstring: str = ""
    examples: list[str] = field(default_factory=list)
    return_type: Optional[str] = None


@dataclass
class KnowledgeContext:
    """Aggregated knowledge context for planning."""
    package_info: Optional[PackageInfo] = None
    docstrings: dict[str, DocString] = field(default_factory=dict)
    types: dict[str, TypeInfo] = field(default_factory=dict)
    methods: dict[str, MethodInfo] = field(default_factory=dict)
    related_symbols: dict[str, list[str]] = field(default_factory=dict)
    
    # Embedding-ready representations
    symbol_descriptions: dict[str, str] = field(default_factory=dict)
    context_embedding: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Knowledge Extractors
# ---------------------------------------------------------------------------


class DocStringExtractor:
    """Extract docstrings from Julia source files."""
    
    # Pattern for Julia docstrings (triple-quoted strings before definitions)
    DOCSTRING_PATTERN = re.compile(
        r'"""(.*?)"""\s*\n\s*(function|struct|abstract type|const|macro)\s+(\w+)',
        re.DOTALL
    )
    
    # Pattern for single-line docstrings
    SINGLE_LINE_PATTERN = re.compile(
        r'"([^"]+)"\s*\n\s*(function|struct|abstract type|const|macro)\s+(\w+)'
    )
    
    def extract_from_source(self, source: str, module_name: str = "Main") -> list[DocString]:
        """Extract all docstrings from Julia source code."""
        docstrings = []
        
        # Multi-line docstrings
        for match in self.DOCSTRING_PATTERN.finditer(source):
            doc_text = match.group(1).strip()
            symbol_type = match.group(2)
            symbol_name = match.group(3)
            
            docstrings.append(self._parse_docstring(
                doc_text, symbol_name, module_name, symbol_type
            ))
        
        # Single-line docstrings
        for match in self.SINGLE_LINE_PATTERN.finditer(source):
            doc_text = match.group(1).strip()
            symbol_type = match.group(2)
            symbol_name = match.group(3)
            
            docstrings.append(DocString(
                symbol=symbol_name,
                module=module_name,
                docstring=doc_text,
            ))
        
        return docstrings
    
    def _parse_docstring(
        self,
        doc_text: str,
        symbol: str,
        module: str,
        symbol_type: str,
    ) -> DocString:
        """Parse a docstring to extract structured information."""
        lines = doc_text.split("\n")
        
        signature = None
        examples = []
        see_also = []
        main_doc = []
        
        in_examples = False
        in_see_also = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for section headers
            if stripped.lower().startswith("# example") or stripped.lower().startswith("## example"):
                in_examples = True
                in_see_also = False
                continue
            elif stripped.lower().startswith("# see also") or stripped.lower().startswith("see also:"):
                in_see_also = True
                in_examples = False
                continue
            elif stripped.startswith("#"):
                in_examples = False
                in_see_also = False
            
            # Parse content based on section
            if in_examples:
                if stripped.startswith("```"):
                    continue
                if stripped:
                    examples.append(stripped)
            elif in_see_also:
                # Extract symbol references
                refs = re.findall(r'`(\w+)`', stripped)
                see_also.extend(refs)
            else:
                # Check for signature (usually first line with parentheses)
                if signature is None and "(" in stripped and ")" in stripped:
                    signature = stripped
                else:
                    main_doc.append(stripped)
        
        return DocString(
            symbol=symbol,
            module=module,
            docstring="\n".join(main_doc).strip(),
            signature=signature,
            examples=examples,
            see_also=see_also,
        )
    
    def extract_from_file(self, path: Path, module_name: str = "Main") -> list[DocString]:
        """Extract docstrings from a Julia file."""
        try:
            source = path.read_text()
            return self.extract_from_source(source, module_name)
        except Exception as e:
            logger.warning(f"Failed to extract docstrings from {path}: {e}")
            return []


class PackageInfoExtractor:
    """Extract package information from Project.toml and README."""
    
    def extract(self, repo_path: Path) -> Optional[PackageInfo]:
        """Extract package info from a Julia repository."""
        project_toml = repo_path / "Project.toml"
        readme = self._find_readme(repo_path)
        
        if not project_toml.exists():
            logger.warning(f"No Project.toml found in {repo_path}")
            return None
        
        try:
            import tomli
            with open(project_toml, "rb") as f:
                data = tomli.load(f)
        except ImportError:
            # Fallback to basic parsing
            data = self._parse_toml_basic(project_toml)
        except Exception as e:
            logger.warning(f"Failed to parse Project.toml: {e}")
            return None
        
        readme_summary = ""
        if readme:
            readme_summary = self._extract_readme_summary(readme)
        
        return PackageInfo(
            name=data.get("name", "Unknown"),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            authors=data.get("authors", []),
            dependencies=list(data.get("deps", {}).keys()),
            exports=self._extract_exports(repo_path, data.get("name", "")),
            readme_summary=readme_summary,
            repository_url=data.get("repo", ""),
        )
    
    def _find_readme(self, repo_path: Path) -> Optional[Path]:
        """Find README file in repository."""
        for name in ["README.md", "README.MD", "Readme.md", "readme.md", "README"]:
            path = repo_path / name
            if path.exists():
                return path
        return None
    
    def _extract_readme_summary(self, readme_path: Path) -> str:
        """Extract summary from README (first paragraph)."""
        try:
            content = readme_path.read_text()
            
            # Skip badges and title
            lines = content.split("\n")
            summary_lines = []
            started = False
            
            for line in lines:
                stripped = line.strip()
                
                # Skip badges, titles, and empty lines at start
                if not started:
                    if stripped.startswith("#") or stripped.startswith("[!") or not stripped:
                        continue
                    started = True
                
                if started:
                    if not stripped:
                        break  # End of first paragraph
                    summary_lines.append(stripped)
            
            return " ".join(summary_lines)[:500]  # Limit length
            
        except Exception as e:
            logger.warning(f"Failed to extract README summary: {e}")
            return ""
    
    def _extract_exports(self, repo_path: Path, package_name: str) -> list[str]:
        """Extract exported symbols from the main module file."""
        exports = []
        
        # Look for src/{PackageName}.jl
        main_file = repo_path / "src" / f"{package_name}.jl"
        if not main_file.exists():
            return exports
        
        try:
            source = main_file.read_text()
            
            # Find export statements
            export_pattern = re.compile(r'export\s+([\w,\s]+)', re.MULTILINE)
            for match in export_pattern.finditer(source):
                symbols = match.group(1).split(",")
                exports.extend([s.strip() for s in symbols if s.strip()])
            
        except Exception as e:
            logger.warning(f"Failed to extract exports: {e}")
        
        return exports
    
    def _parse_toml_basic(self, path: Path) -> dict:
        """Basic TOML parsing without external dependencies."""
        data = {}
        current_section = data
        
        try:
            for line in path.read_text().split("\n"):
                line = line.strip()
                
                if not line or line.startswith("#"):
                    continue
                
                if line.startswith("[") and line.endswith("]"):
                    section_name = line[1:-1]
                    data[section_name] = {}
                    current_section = data[section_name]
                elif "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    current_section[key] = value
        except Exception:
            pass
        
        return data


class TypeExtractor:
    """Extract type/struct information from Julia source."""
    
    STRUCT_PATTERN = re.compile(
        r'(mutable\s+)?struct\s+(\w+)(?:\s*<:\s*(\w+))?\s*\n(.*?)\nend',
        re.DOTALL
    )
    
    ABSTRACT_PATTERN = re.compile(
        r'abstract\s+type\s+(\w+)(?:\s*<:\s*(\w+))?\s*end'
    )
    
    def extract_from_source(self, source: str, module_name: str = "Main") -> list[TypeInfo]:
        """Extract type definitions from Julia source."""
        types = []
        
        # Concrete structs
        for match in self.STRUCT_PATTERN.finditer(source):
            is_mutable = match.group(1) is not None
            name = match.group(2)
            supertype = match.group(3)
            body = match.group(4)
            
            fields = self._parse_fields(body)
            
            types.append(TypeInfo(
                name=name,
                module=module_name,
                supertype=supertype,
                fields=fields,
            ))
        
        # Abstract types
        for match in self.ABSTRACT_PATTERN.finditer(source):
            name = match.group(1)
            supertype = match.group(2)
            
            types.append(TypeInfo(
                name=name,
                module=module_name,
                supertype=supertype,
                fields=[],  # Abstract types have no fields
            ))
        
        return types
    
    def _parse_fields(self, body: str) -> list[tuple[str, str]]:
        """Parse struct fields from body."""
        fields = []
        
        for line in body.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Pattern: field_name::Type or just field_name
            if "::" in line:
                name, type_str = line.split("::", 1)
                fields.append((name.strip(), type_str.strip()))
            else:
                fields.append((line, "Any"))
        
        return fields


# ---------------------------------------------------------------------------
# Knowledge Synthesizer
# ---------------------------------------------------------------------------


class KnowledgeSynthesizer:
    """
    Synthesizes knowledge context for planning.
    
    Gathers documentation, type info, and method signatures to enrich
    the world state representation before planning.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        embedding_model: Optional[Any] = None,
    ):
        self.cache_dir = cache_dir
        self.embedding_model = embedding_model
        
        self.docstring_extractor = DocStringExtractor()
        self.package_extractor = PackageInfoExtractor()
        self.type_extractor = TypeExtractor()
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def synthesize(
        self,
        repo_path: Path,
        target_symbols: Optional[list[str]] = None,
    ) -> KnowledgeContext:
        """
        Synthesize knowledge context for a repository.
        
        Args:
            repo_path: Path to the Julia repository
            target_symbols: Specific symbols to gather context for
        
        Returns:
            KnowledgeContext with aggregated knowledge
        """
        logger.info(f"Synthesizing knowledge for {repo_path}")
        
        context = KnowledgeContext()
        
        # Extract package info
        context.package_info = self.package_extractor.extract(repo_path)
        
        # Extract from all source files
        src_dir = repo_path / "src"
        if src_dir.exists():
            for julia_file in src_dir.rglob("*.jl"):
                self._process_source_file(julia_file, context)
        
        # Build symbol descriptions
        self._build_symbol_descriptions(context, target_symbols)
        
        # Build related symbols graph
        self._build_related_symbols(context)
        
        # Compute context embedding
        if self.embedding_model is not None:
            context.context_embedding = self._compute_context_embedding(context)
        
        return context
    
    def _process_source_file(self, path: Path, context: KnowledgeContext):
        """Process a single source file."""
        try:
            source = path.read_text()
            module_name = self._infer_module_name(path)
            
            # Extract docstrings
            for doc in self.docstring_extractor.extract_from_source(source, module_name):
                key = f"{doc.module}.{doc.symbol}"
                context.docstrings[key] = doc
            
            # Extract types
            for type_info in self.type_extractor.extract_from_source(source, module_name):
                key = f"{type_info.module}.{type_info.name}"
                context.types[key] = type_info
            
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")
    
    def _infer_module_name(self, path: Path) -> str:
        """Infer module name from file path."""
        # Simple heuristic: use filename without extension
        return path.stem
    
    def _build_symbol_descriptions(
        self,
        context: KnowledgeContext,
        target_symbols: Optional[list[str]] = None,
    ):
        """Build natural language descriptions for symbols."""
        # From docstrings
        for key, doc in context.docstrings.items():
            if target_symbols is None or doc.symbol in target_symbols:
                desc = self._format_symbol_description(doc)
                context.symbol_descriptions[key] = desc
        
        # From types
        for key, type_info in context.types.items():
            if target_symbols is None or type_info.name in target_symbols:
                desc = self._format_type_description(type_info)
                context.symbol_descriptions[key] = desc
    
    def _format_symbol_description(self, doc: DocString) -> str:
        """Format a docstring as a description."""
        parts = []
        
        if doc.signature:
            parts.append(f"Signature: {doc.signature}")
        
        if doc.docstring:
            parts.append(doc.docstring[:200])  # Truncate
        
        if doc.examples:
            parts.append(f"Example: {doc.examples[0]}")
        
        return " | ".join(parts)
    
    def _format_type_description(self, type_info: TypeInfo) -> str:
        """Format a type as a description."""
        parts = [f"Type {type_info.name}"]
        
        if type_info.supertype:
            parts.append(f"extends {type_info.supertype}")
        
        if type_info.fields:
            field_strs = [f"{name}::{typ}" for name, typ in type_info.fields[:5]]
            parts.append(f"with fields {', '.join(field_strs)}")
        
        if type_info.docstring:
            parts.append(type_info.docstring[:100])
        
        return " | ".join(parts)
    
    def _build_related_symbols(self, context: KnowledgeContext):
        """Build graph of related symbols."""
        # From see_also references
        for key, doc in context.docstrings.items():
            related = []
            
            for ref in doc.see_also:
                # Find the full key for the reference
                for other_key in context.docstrings.keys():
                    if other_key.endswith(f".{ref}"):
                        related.append(other_key)
                        break
            
            if related:
                context.related_symbols[key] = related
        
        # From type hierarchy
        for key, type_info in context.types.items():
            if type_info.supertype:
                # Find the supertype
                for other_key, other_type in context.types.items():
                    if other_type.name == type_info.supertype:
                        context.related_symbols.setdefault(key, []).append(other_key)
                        context.related_symbols.setdefault(other_key, []).append(key)
                        break
    
    def _compute_context_embedding(self, context: KnowledgeContext) -> list[float]:
        """Compute embedding for the entire context."""
        # Concatenate all descriptions
        all_text = []
        
        if context.package_info:
            all_text.append(context.package_info.description)
            all_text.append(context.package_info.readme_summary)
        
        for desc in context.symbol_descriptions.values():
            all_text.append(desc)
        
        combined = " ".join(all_text)[:4096]  # Limit length
        
        # Use embedding model (placeholder)
        # In practice, this would call a sentence transformer
        return self.embedding_model.encode(combined).tolist()
    
    def get_context_for_symbol(
        self,
        context: KnowledgeContext,
        symbol: str,
    ) -> dict[str, Any]:
        """Get context relevant to a specific symbol."""
        result = {
            "symbol": symbol,
            "description": None,
            "related": [],
            "type_info": None,
            "docstring": None,
        }
        
        # Find by suffix match
        for key, desc in context.symbol_descriptions.items():
            if key.endswith(f".{symbol}"):
                result["description"] = desc
                break
        
        for key, doc in context.docstrings.items():
            if key.endswith(f".{symbol}"):
                result["docstring"] = doc
                break
        
        for key, type_info in context.types.items():
            if key.endswith(f".{symbol}"):
                result["type_info"] = type_info
                break
        
        # Get related symbols
        for key, related in context.related_symbols.items():
            if key.endswith(f".{symbol}"):
                result["related"] = [r.split(".")[-1] for r in related]
                break
        
        return result


# ---------------------------------------------------------------------------
# Planning Context Enrichment
# ---------------------------------------------------------------------------


class PlanningContextEnricher:
    """
    Enriches planning context with synthesized knowledge.
    
    Used before planning to provide the JEPA model with additional
    semantic context about the codebase.
    """
    
    def __init__(
        self,
        synthesizer: KnowledgeSynthesizer,
        cache_knowledge: bool = True,
    ):
        self.synthesizer = synthesizer
        self.cache_knowledge = cache_knowledge
        self._context_cache: dict[str, KnowledgeContext] = {}
    
    def enrich(
        self,
        repo_path: Path,
        goal: str,
        affected_symbols: list[str],
    ) -> dict[str, Any]:
        """
        Enrich planning context with knowledge.
        
        Args:
            repo_path: Path to the repository
            goal: Natural language goal description
            affected_symbols: Symbols that may be affected by the plan
        
        Returns:
            Enriched context dict for planning
        """
        # Get or synthesize knowledge context
        cache_key = str(repo_path)
        if cache_key in self._context_cache and self.cache_knowledge:
            knowledge = self._context_cache[cache_key]
        else:
            knowledge = self.synthesizer.synthesize(repo_path, affected_symbols)
            if self.cache_knowledge:
                self._context_cache[cache_key] = knowledge
        
        # Build enriched context
        enriched = {
            "goal": goal,
            "package_name": knowledge.package_info.name if knowledge.package_info else None,
            "package_description": knowledge.package_info.description if knowledge.package_info else None,
            "symbol_contexts": {},
            "related_symbols": {},
            "context_embedding": knowledge.context_embedding,
        }
        
        # Add context for each affected symbol
        for symbol in affected_symbols:
            symbol_context = self.synthesizer.get_context_for_symbol(knowledge, symbol)
            if symbol_context["description"]:
                enriched["symbol_contexts"][symbol] = symbol_context
        
        # Add related symbols for discovery
        for symbol in affected_symbols:
            for key, related in knowledge.related_symbols.items():
                if key.endswith(f".{symbol}"):
                    enriched["related_symbols"][symbol] = [r.split(".")[-1] for r in related]
        
        return enriched
