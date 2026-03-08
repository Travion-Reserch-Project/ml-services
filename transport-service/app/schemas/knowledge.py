"""
Knowledge Base and RAG Schemas

This module defines Pydantic models for:
- Knowledge document structure
- Search requests and responses
- PDF extraction
- RAG system interactions
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ================================================
# Knowledge Document Models
# ================================================

class DocumentMetadata(BaseModel):
    """Metadata for knowledge documents"""
    
    # Location information (optional, bilingual support)
    location: Optional[str] = None
    location_en: Optional[str] = None  # English translation
    location_si: Optional[str] = None  # Sinhala translation
    
    # Transport-related fields (generic)
    transport_type: Optional[str] = None  # "bus", "train", "road", "tourist_route", etc.
    route_name: Optional[str] = None
    route_number: Optional[str] = None
    operator: Optional[str] = None
    
    # Document metadata
    language: Literal["en", "si", "ta", "mixed"] = "en"
    source: Optional[str] = None  # e.g., "wikipedia", "facebook", "government_website"
    source_url: Optional[str] = None
    extracted_date: Optional[str] = None
    last_updated: Optional[str] = None
    
    # Categorization
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None  # "road_network", "railway", "traffic_rules", "tourist_route", "tips", etc.
    verified: bool = False


class KnowledgeDocument(BaseModel):
    """Core knowledge document structure"""
    
    id: str = Field(..., description="Unique document identifier")
    type: Literal["road_network", "railway", "traffic_rules", "tourist_route", "transport_tip", "general"] = "general"
    content: str = Field(..., description="Main text content for embedding")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    
    # Optional fields
    embedding: Optional[List[float]] = None  # Vector representation
    created_at: Optional[datetime] = None
    

class DocumentUploadRequest(BaseModel):
    """Request to upload knowledge documents"""
    
    documents: List[KnowledgeDocument] = Field(..., min_length=1)
    collection_name: str = Field(default="transport_knowledge", description="Target collection")
    overwrite: bool = Field(default=False, description="Overwrite existing documents")


class DocumentUploadResponse(BaseModel):
    """Response after uploading documents"""
    
    success: bool
    documents_added: int
    documents_updated: int
    duplicates_skipped: int = Field(default=0, description="Number of duplicate routes skipped")
    failed_ids: List[str] = Field(default_factory=list)
    message: str


# ================================================
# Search & RAG Models
# ================================================

class KnowledgeSearchRequest(BaseModel):
    """Request for semantic search + RAG"""
    
    query: str = Field(..., min_length=1, description="User's question or search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    collection_name: Optional[str] = Field(default=None, description="Specific collection to search")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    use_rag: bool = Field(default=True, description="Generate answer using RAG")
    language: Optional[Literal["en", "si", "ta", "auto"]] = Field(default="auto")


class SearchResult(BaseModel):
    """Single search result"""
    
    id: str
    content: str
    score: float = Field(..., description="Similarity score (0-1)")
    metadata: DocumentMetadata
    distance: Optional[float] = None  # Cosine distance from query


class KnowledgeSearchResponse(BaseModel):
    """Response from knowledge search"""
    
    success: bool
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    
    # RAG-specific fields
    answer: Optional[str] = None  # Generated answer from LLM
    generated_by: Optional[str] = None  # Model used (e.g., "claude-3.5-sonnet")
    
    # Metadata
    query_language: Optional[str] = None
    total_results: int = 0
    search_time_ms: Optional[float] = None
    rag_time_ms: Optional[float] = None


# ================================================
# PDF Extraction Models
# ================================================

class FareData(BaseModel):
    """Structured fare information extracted from PDF"""
    
    origin: str
    destination: str
    distance_km: Optional[float] = None
    fare_normal: Optional[float] = None
    fare_semi_luxury: Optional[float] = None
    fare_luxury: Optional[float] = None
    fare_super_luxury: Optional[float] = None
    route_number: Optional[str] = None
    additional_info: Optional[str] = None


class PDFExtractionRequest(BaseModel):
    """Request to extract fare data from PDF"""
    
    pdf_path: Optional[str] = None  # Local file path
    pdf_url: Optional[str] = None   # Remote URL
    service_type: Optional[Literal["normal", "semi_luxury", "luxury", "super_luxury"]] = None
    extract_tables: bool = Field(default=True)
    save_structured: bool = Field(default=True, description="Save as JSON in data/structured/")


class PDFExtractionResponse(BaseModel):
    """Response from PDF extraction"""
    
    success: bool
    pdf_name: str
    fares_extracted: List[FareData] = Field(default_factory=list)
    total_fares: int = 0
    errors: List[str] = Field(default_factory=list)
    structured_file_path: Optional[str] = None


class DataExtractionResult(BaseModel):
    """Result from web data extraction/scraping"""
    
    success: bool
    documents: List[KnowledgeDocument] = Field(default_factory=list)
    documents_extracted: int = 0
    source_url: str = ""
    error: Optional[str] = None


# ================================================
# Statistics & Health Models
# ================================================

class CollectionStats(BaseModel):
    """Statistics for a single collection"""
    
    name: str
    document_count: int
    last_updated: Optional[str] = None
    avg_document_length: Optional[float] = None


class KnowledgeStatsResponse(BaseModel):
    """Overall knowledge base statistics"""
    
    total_documents: int
    collections: List[CollectionStats] = Field(default_factory=list)
    storage_size_mb: Optional[float] = None
    embedding_model: str
    vector_db_type: str = "chromadb"
    last_sync: Optional[str] = None


# ================================================
# Straico API Models
# ================================================

class StraicoRequest(BaseModel):
    """Request to Straico API"""
    
    model: str
    messages: List[Dict[str, str]]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1000, le=8000)
    stream: bool = False


class StraicoResponse(BaseModel):
    """Response from Straico API"""
    
    content: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


# ================================================
# Error Models
# ================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
