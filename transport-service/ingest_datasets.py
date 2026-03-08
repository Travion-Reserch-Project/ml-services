#!/usr/bin/env python
"""
Dataset Ingestion Script

Ingests all downloaded datasets from kb_transport/ into ChromaDB.
Processes multiple formats: GeoJSON, CSV, JSON, TXT, MD

Usage:
    python ingest_datasets.py --category gis
    python ingest_datasets.py --category railway
    python ingest_datasets.py  # ingest all
"""

import logging
import sys
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Ingest datasets into vector database"""
    parser = argparse.ArgumentParser(description="Ingest transport datasets into RAG system")
    parser.add_argument(
        '--category',
        type=str,
        choices=['gis', 'roads', 'railway', 'bus', 'fares', 'rules', 'signs', 
                 'tourism', 'history', 'stats', 'reports', 'gtfs', 'all'],
        default='all',
        help="Category to ingest (default: all)"
    )
    parser.add_argument(
        '--kb-dir',
        type=str,
        default='kb_transport',
        help="Knowledge base directory (default: kb_transport)"
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='transport_knowledge',
        help="ChromaDB collection name (default: transport_knowledge)"
    )
    
    args = parser.parse_args()
    
    try:
        from app.services.data_ingestion_service import DataIngestionService
        from app.services.vector_db_service import VectorDBService
        
        logger.info("=" * 80)
        logger.info("🚀 TRANSPORT DATASET INGESTION")
        logger.info("=" * 80)
        
        # Initialize services
        logger.info("\n📦 Initializing services...")
        ingestion = DataIngestionService(kb_base_path=args.kb_dir)
        vector_db = VectorDBService()
        
        kb_base = Path(args.kb_dir)
        
        # Determine which categories to process
        if args.category == 'all':
            categories = ['gis', 'roads', 'railway', 'bus', 'fares', 'rules', 
                         'signs', 'tourism', 'history', 'stats', 'reports', 'gtfs']
        else:
            categories = [args.category]
        
        all_documents = []
        
        # Process each category
        for category in categories:
            category_dir = kb_base / category
            
            if not category_dir.exists() or not any(category_dir.iterdir()):
                logger.warning(f"⚠️  Category directory empty or missing: {category}")
                logger.info(f"   Download datasets to: {category_dir}")
                continue
            
            logger.info(f"\n📂 Processing category: {category.upper()}")
            logger.info(f"   Directory: {category_dir}")
            
            # Special handling for GTFS
            if category == 'gtfs':
                docs = ingestion.ingest_gtfs(category_dir)
            else:
                docs = ingestion.ingest_directory(category_dir, category, recursive=True)
            
            all_documents.extend(docs)
            logger.info(f"   ✓ Extracted {len(docs)} documents from {category}")
        
        if not all_documents:
            logger.error("❌ No documents extracted. Download datasets first!")
            logger.info("\n📝 Run: python download_datasets.py")
            logger.info("   Then follow instructions to download real datasets")
            return 1
        
        # Add to vector database
        logger.info(f"\n🔄 Adding {len(all_documents)} documents to ChromaDB...")
        logger.info(f"   Collection: {args.collection}")
        
        result = vector_db.add_documents(
            documents=all_documents,
            collection_name=args.collection,
            skip_duplicates=True
        )
        
        logger.info(f"\n✅ Ingestion complete!")
        logger.info(f"   Documents added: {result.get('added', 0)}")
        logger.info(f"   Duplicates skipped: {result.get('duplicates_skipped', 0)}")
        logger.info(f"   Total in collection: {result.get('total', 0)}")
        
        # Get statistics
        logger.info(f"\n📊 Collection statistics:")
        stats = vector_db.get_collection_stats(args.collection)
        logger.info(f"   Total documents: {stats.document_count}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ DATASET INGESTION SUCCESSFUL!")
        logger.info("=" * 80)
        logger.info("\n📝 Next steps:")
        logger.info("   1. Start the API: python app.py")
        logger.info("   2. Test search with real queries")
        logger.info("   3. Query examples:")
        logger.info('      - "What are the main highways in Sri Lanka?"')
        logger.info('      - "Railway routes from Colombo"')
        logger.info('      - "Traffic rules for driving"')
        logger.info('      - "Tourist places along southern coast"')
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
