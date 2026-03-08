#!/usr/bin/env python
"""
Load Transport Knowledge Into Vector Database

Ingests sample transport knowledge from WebScraperService into ChromaDB.
Run this once to populate the database with initial transport data.

Usage:
    python load_transport_knowledge.py
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Load transport knowledge into database"""
    try:
        from app.services.web_scraper_service import WebScraperService
        from app.services.vector_db_service import VectorDBService
        
        logger.info("=" * 80)
        logger.info("🚀 TRANSPORT KNOWLEDGE LOADER")
        logger.info("=" * 80)
        
        # Initialize services
        logger.info("\n📦 Initializing services...")
        scraper = WebScraperService()
        vector_db = VectorDBService()
        
        # Get sample data
        logger.info("\n📚 Fetching general transport knowledge...")
        general_docs = scraper.get_sample_transport_data()
        
        logger.info("📚 Fetching bilingual transport knowledge...")
        bilingual_docs = scraper.get_bilingual_data()
        
        all_docs = general_docs + bilingual_docs
        logger.info(f"✓ Fetched {len(all_docs)} documents total")
        
        # Add documents to collection
        collection_name = "transport_knowledge"
        logger.info(f"\n🔄 Adding {len(all_docs)} documents to collection: {collection_name}...")
        
        result = vector_db.add_documents(
            documents=all_docs,
            collection_name=collection_name,
            skip_duplicates=False
        )
        
        logger.info(f"\n✅ Successfully added {result.get('added', 0)} documents")
        logger.info(f"   Total in collection: {result.get('total', 0)}")
        
        # Verify
        logger.info(f"\n📊 Collection statistics:")
        stats = vector_db.get_collection_stats(collection_name)
        logger.info(f"   Documents: {stats.document_count}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Transport knowledge loaded successfully!")
        logger.info("=" * 80)
        logger.info("\n📝 Next steps:")
        logger.info("   1. Start the API: python app.py")
        logger.info("   2. Test search: curl -X POST http://localhost:8001/api/knowledge/search \\")
        logger.info('      -d \'{"query": "How do I get from Colombo to Galle?"}\' \\')
        logger.info("      -H 'Content-Type: application/json'")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to load transport knowledge: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
