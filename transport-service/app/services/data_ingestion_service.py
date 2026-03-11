"""
Data Ingestion Service

Handles downloading, parsing, and chunking of real Sri Lankan transport datasets:
- GIS/Map data (GeoJSON, Shapefiles)
- Railway data (GTFS, CSV, JSON)
- Bus routes and schedules
- Traffic rules and signs
- Tourism routes
- Historical transport data
- Fare tables

Supports multiple formats: txt, json, csv, md, geojson, pdf
"""

import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from app.schemas.knowledge import KnowledgeDocument, DocumentMetadata

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Service for ingesting real transport datasets into RAG system"""
    
    def __init__(self, kb_base_path: str = "kb_transport"):
        """
        Initialize data ingestion service
        
        Args:
            kb_base_path: Base directory for knowledge base storage
        """
        self.kb_base = Path(kb_base_path)
        self.kb_base.mkdir(parents=True, exist_ok=True)
        
        # Define categories and their subdirectories
        self.categories = {
            "gis": self.kb_base / "gis",
            "roads": self.kb_base / "roads",
            "railway": self.kb_base / "railway",
            "bus": self.kb_base / "bus",
            "fares": self.kb_base / "fares",
            "rules": self.kb_base / "rules",
            "signs": self.kb_base / "signs",
            "tourism": self.kb_base / "tourism",
            "history": self.kb_base / "history",
            "stats": self.kb_base / "stats",
            "maps": self.kb_base / "maps",
            "reports": self.kb_base / "reports",
            "gtfs": self.kb_base / "gtfs",
            "pdf": self.kb_base / "pdf",
            "images": self.kb_base / "images"
        }
        
        logger.info(f"✓ Data ingestion service initialized (KB: {kb_base_path})")
    
    def ingest_geojson(
        self,
        file_path: Path,
        category: str = "gis",
        chunk_features: bool = True
    ) -> List[KnowledgeDocument]:
        """
        Ingest GeoJSON files (roads, railways, boundaries)
        
        Args:
            file_path: Path to GeoJSON file
            category: KB category (gis, roads, railway)
            chunk_features: Whether to create one doc per feature or combine
            
        Returns:
            List of KnowledgeDocument objects
        """
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if chunk_features and 'features' in data:
                # Create one document per feature
                for idx, feature in enumerate(data['features']):
                    properties = feature.get('properties', {})
                    geometry = feature.get('geometry', {})
                    
                    # Build readable content
                    content = self._geojson_feature_to_text(feature, properties)
                    
                    doc_id = f"{category}_geo_{file_path.stem}_{idx}"
                    
                    metadata = DocumentMetadata(
                        transport_type=category,
                        category=category,
                        source=f"geojson_{file_path.name}",
                        source_url=str(file_path),
                        language="en",
                        verified=True,
                        tags=[category, "gis", "geojson"]
                    )
                    
                    doc = KnowledgeDocument(
                        id=doc_id,
                        type="road_network" if category == "roads" else "railway" if category == "railway" else "general",
                        content=content,
                        metadata=metadata,
                        created_at=datetime.now()
                    )
                    documents.append(doc)
            
            logger.info(f"✓ Ingested {len(documents)} features from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to ingest GeoJSON {file_path}: {e}")
        
        return documents
    
    def _geojson_feature_to_text(self, feature: Dict, properties: Dict) -> str:
        """Convert GeoJSON feature to readable text"""
        parts = []
        
        # Add feature type
        geometry_type = feature.get('geometry', {}).get('type', 'Unknown')
        parts.append(f"Feature Type: {geometry_type}")
        
        # Add properties
        if properties:
            parts.append("\nProperties:")
            for key, value in properties.items():
                if value:
                    parts.append(f"  {key}: {value}")
        
        return "\n".join(parts)
    
    def ingest_csv(
        self,
        file_path: Path,
        category: str,
        chunk_rows: bool = True,
        max_rows_per_doc: int = 10
    ) -> List[KnowledgeDocument]:
        """
        Ingest CSV files (stats, fares, routes)
        
        Args:
            file_path: Path to CSV file
            category: KB category
            chunk_rows: Whether to chunk into multiple documents
            max_rows_per_doc: Maximum rows per document
            
        Returns:
            List of KnowledgeDocument objects
        """
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if chunk_rows:
                # Chunk into multiple documents
                for chunk_idx in range(0, len(rows), max_rows_per_doc):
                    chunk = rows[chunk_idx:chunk_idx + max_rows_per_doc]
                    content = self._csv_chunk_to_text(chunk, file_path.stem)
                    
                    doc_id = f"{category}_csv_{file_path.stem}_{chunk_idx}"
                    
                    metadata = DocumentMetadata(
                        transport_type=category,
                        category=category,
                        source=f"csv_{file_path.name}",
                        source_url=str(file_path),
                        language="en",
                        verified=True,
                        tags=[category, "csv", "data"]
                    )
                    
                    doc = KnowledgeDocument(
                        id=doc_id,
                        type="general",
                        content=content,
                        metadata=metadata,
                        created_at=datetime.now()
                    )
                    documents.append(doc)
            
            logger.info(f"✓ Ingested {len(documents)} chunks from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to ingest CSV {file_path}: {e}")
        
        return documents
    
    def _csv_chunk_to_text(self, rows: List[Dict], table_name: str) -> str:
        """Convert CSV chunk to readable text"""
        parts = [f"Data from: {table_name}\n"]
        
        for idx, row in enumerate(rows, 1):
            parts.append(f"\nRecord {idx}:")
            for key, value in row.items():
                if value:
                    parts.append(f"  {key}: {value}")
        
        return "\n".join(parts)
    
    def ingest_json(
        self,
        file_path: Path,
        category: str,
        chunk_key: Optional[str] = None
    ) -> List[KnowledgeDocument]:
        """
        Ingest JSON files (routes, stops, schedules, Wikipedia articles)
        
        Args:
            file_path: Path to JSON file
            category: KB category
            chunk_key: Key to chunk array by (e.g., "routes", "stops")
            
        Returns:
            List of KnowledgeDocument objects
        """
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Infer chunk key for known structured JSON datasets when not provided.
            # This ensures files like city analysis, OSM bus stations, and railway station lists
            # are split into one document per item rather than one huge document.
            if chunk_key is None and isinstance(data, dict):
                for candidate_key in [
                    "cities",
                    "all_bus_locations",
                    "official_railway_lines",
                    "stations",
                    "routes",
                    "stops",
                    "trips",
                    "features",
                ]:
                    if isinstance(data.get(candidate_key), list):
                        chunk_key = candidate_key
                        break
            
            # Check if this is Wikipedia article format (list of articles)
            if isinstance(data, list) and len(data) > 0 and 'title' in data[0] and 'content' in data[0]:
                # Wikipedia articles format
                for article in data:
                    title = article.get('title', 'Unknown')
                    content = article.get('content', '')
                    url = article.get('url', '')
                    transport_type = article.get('transport_type', 'general')
                    article_category = article.get('category', category)
                    
                    if not content:
                        continue
                    
                    # Use article title in document ID
                    safe_title = title.replace(' ', '_').replace('/', '_')[:50]
                    doc_id = f"wiki_{article_category}_{safe_title}"
                    
                    # Format content with title
                    formatted_content = f"# {title}\n\n{content}"
                    
                    metadata = DocumentMetadata(
                        transport_type=transport_type,
                        category=article_category,
                        source="Wikipedia",
                        source_url=url,
                        language="en",
                        verified=True,
                        tags=[article_category, "wikipedia", transport_type]
                    )
                    
                    doc = KnowledgeDocument(
                        id=doc_id,
                        type="general",
                        content=formatted_content,
                        metadata=metadata,
                        created_at=datetime.now()
                    )
                    documents.append(doc)
            
            elif chunk_key and isinstance(data.get(chunk_key), list):
                # Chunk array items
                items = data[chunk_key]
                for idx, item in enumerate(items):
                    content = json.dumps(item, indent=2, ensure_ascii=False)
                    
                    doc_id = f"{category}_json_{file_path.stem}_{idx}"
                    
                    metadata = DocumentMetadata(
                        transport_type=category,
                        category=category,
                        source=f"json_{file_path.name}",
                        source_url=str(file_path),
                        language="en",
                        verified=True,
                        tags=[category, "json"]
                    )
                    
                    doc = KnowledgeDocument(
                        id=doc_id,
                        type="general",
                        content=content,
                        metadata=metadata,
                        created_at=datetime.now()
                    )
                    documents.append(doc)
            else:
                # Single document
                content = json.dumps(data, indent=2, ensure_ascii=False)
                
                doc_id = f"{category}_json_{file_path.stem}"
                
                metadata = DocumentMetadata(
                    transport_type=category,
                    category=category,
                    source=f"json_{file_path.name}",
                    source_url=str(file_path),
                    language="en",
                    verified=True,
                    tags=[category, "json"]
                )
                
                doc = KnowledgeDocument(
                    id=doc_id,
                    type="general",
                    content=content,
                    metadata=metadata,
                    created_at=datetime.now()
                )
                documents.append(doc)
            
            logger.info(f"✓ Ingested {len(documents)} documents from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to ingest JSON {file_path}: {e}")
        
        return documents
    
    def ingest_text(
        self,
        file_path: Path,
        category: str,
        chunk_size: int = 1000
    ) -> List[KnowledgeDocument]:
        """
        Ingest text files (history, rules, guides)
        
        Args:
            file_path: Path to text file
            category: KB category
            chunk_size: Characters per chunk
            
        Returns:
            List of KnowledgeDocument objects
        """
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Simple chunking by size
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            for idx, chunk in enumerate(chunks):
                doc_id = f"{category}_txt_{file_path.stem}_{idx}"
                
                metadata = DocumentMetadata(
                    transport_type=category,
                    category=category,
                    source=f"text_{file_path.name}",
                    source_url=str(file_path),
                    language="en",
                    verified=True,
                    tags=[category, "text"]
                )
                
                doc = KnowledgeDocument(
                    id=doc_id,
                    type="general",
                    content=chunk,
                    metadata=metadata,
                    created_at=datetime.now()
                )
                documents.append(doc)
            
            logger.info(f"✓ Ingested {len(documents)} chunks from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to ingest text {file_path}: {e}")
        
        return documents
    
    def ingest_gtfs(self, gtfs_dir: Path) -> List[KnowledgeDocument]:
        """
        Ingest GTFS (General Transit Feed Specification) data
        
        Standard GTFS files:
        - routes.txt
        - stops.txt
        - trips.txt
        - stop_times.txt
        - calendar.txt
        - shapes.txt
        
        Args:
            gtfs_dir: Directory containing GTFS files
            
        Returns:
            List of KnowledgeDocument objects
        """
        documents = []
        
        gtfs_files = {
            "routes": "routes.txt",
            "stops": "stops.txt",
            "trips": "trips.txt",
            "stop_times": "stop_times.txt",
            "calendar": "calendar.txt",
            "shapes": "shapes.txt"
        }
        
        for file_type, filename in gtfs_files.items():
            file_path = gtfs_dir / filename
            if file_path.exists():
                logger.info(f"Ingesting GTFS {file_type} from {filename}")
                docs = self.ingest_csv(file_path, category="gtfs", chunk_rows=True, max_rows_per_doc=20)
                documents.extend(docs)
        
        logger.info(f"✓ Total GTFS documents ingested: {len(documents)}")
        return documents
    
    def ingest_directory(
        self,
        directory: Path,
        category: str,
        recursive: bool = True
    ) -> List[KnowledgeDocument]:
        """
        Ingest all supported files from a directory
        
        Args:
            directory: Directory to scan
            category: KB category
            recursive: Whether to scan subdirectories
            
        Returns:
            List of KnowledgeDocument objects
        """
        documents = []
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue
            
            ext = file_path.suffix.lower()
            
            if ext == '.geojson':
                docs = self.ingest_geojson(file_path, category)
                documents.extend(docs)
            elif ext == '.json':
                docs = self.ingest_json(file_path, category)
                documents.extend(docs)
            elif ext == '.csv':
                docs = self.ingest_csv(file_path, category)
                documents.extend(docs)
            elif ext in ['.txt', '.md']:
                docs = self.ingest_text(file_path, category)
                documents.extend(docs)
            else:
                logger.debug(f"Skipping unsupported file: {file_path.name}")
        
        logger.info(f"✓ Ingested {len(documents)} documents from {directory}")
        return documents
