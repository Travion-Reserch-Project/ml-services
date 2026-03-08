#!/usr/bin/env python3
"""
Automated Transport Data Scraper
Scrapes data from OpenStreetMap and Wikipedia, then ingests into ChromaDB
"""

import sys
import logging
from pathlib import Path
import argparse

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.osm_scraper_service import OSMScraperService
from app.services.wikipedia_scraper_service import WikipediaScraperService
from app.services.data_ingestion_service import DataIngestionService
from app.services.vector_db_service import VectorDBService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scrape_osm_data(kb_dir: Path) -> dict:
    """Scrape OpenStreetMap data"""
    logger.info("=" * 60)
    logger.info("🗺️  STEP 1: SCRAPING OPENSTREETMAP DATA")
    logger.info("=" * 60)
    
    scraper = OSMScraperService(output_dir=kb_dir)
    results = scraper.scrape_all()
    
    # Save to GeoJSON files
    saved_files = {}
    for category, features in results.items():
        if features:
            filepath = scraper.save_geojson(features, f"{category}.geojson", "gis")
            saved_files[category] = filepath
    
    logger.info(f"\n✅ OSM scraping complete! Saved {len(saved_files)} files")
    return saved_files


def scrape_wikipedia_data(kb_dir: Path) -> Path:
    """Scrape Wikipedia articles"""
    logger.info("\n" + "=" * 60)
    logger.info("📚 STEP 2: SCRAPING WIKIPEDIA ARTICLES")
    logger.info("=" * 60)
    
    scraper = WikipediaScraperService(output_dir=kb_dir)
    articles = scraper.scrape_all()
    
    # Save all articles
    filepath = scraper.save_articles(articles)
    
    # Also save by category
    scraper.save_by_category(articles)
    
    logger.info(f"\n✅ Wikipedia scraping complete! Scraped {len(articles)} articles")
    return filepath


def ingest_data(kb_dir: Path, collection_name: str = "transport_knowledge") -> dict:
    """Ingest scraped data into ChromaDB"""
    logger.info("\n" + "=" * 60)
    logger.info("💾 STEP 3: INGESTING DATA INTO CHROMADB")
    logger.info("=" * 60)
    
    ingestion = DataIngestionService(kb_base_path=kb_dir)
    vector_db = VectorDBService()
    
    all_documents = []
    
    # Ingest GIS data (GeoJSON files)
    logger.info("\n📍 Ingesting GIS data...")
    gis_dir = kb_dir / "gis"
    if gis_dir.exists():
        gis_docs = ingestion.ingest_directory(gis_dir, category="gis")
        all_documents.extend(gis_docs)
        logger.info(f"✅ Processed {len(gis_docs)} GIS documents")
    
    # Ingest Wikipedia articles (JSON files)
    logger.info("\n📝 Ingesting Wikipedia articles...")
    for category_dir in ["documents", "roads", "railway", "bus", "tourism", "rules", "airports_ports", "general", "maps"]:
        dir_path = kb_dir / category_dir
        if dir_path.exists():
            json_docs = ingestion.ingest_directory(dir_path, category=category_dir)
            all_documents.extend(json_docs)
            logger.info(f"✅ Processed {len(json_docs)} documents from {category_dir}")
    
    # Add to ChromaDB
    logger.info(f"\n🔄 Adding {len(all_documents)} documents to ChromaDB...")
    result = vector_db.add_documents(all_documents, collection_name=collection_name)
    
    logger.info(f"\n✅ Ingestion complete!")
    logger.info(f"   Total documents ingested: {len(all_documents)}")
    logger.info(f"   Documents added to ChromaDB: {result.get('added', 0)}")
    logger.info(f"   Duplicates skipped: {result.get('duplicates_skipped', 0)}")
    
    # Get collection stats
    stats = vector_db.get_collection_stats(collection_name)
    logger.info(f"\n📊 ChromaDB Collection Stats:")
    logger.info(f"   Total documents in collection: {stats.document_count}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Scrape and ingest Sri Lankan transport data")
    parser.add_argument(
        '--kb-dir',
        type=Path,
        default=Path('kb_transport'),
        help='Knowledge base directory (default: kb_transport)'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='transport_knowledge',
        help='ChromaDB collection name (default: transport_knowledge)'
    )
    parser.add_argument(
        '--skip-osm',
        action='store_true',
        help='Skip OpenStreetMap scraping'
    )
    parser.add_argument(
        '--skip-wikipedia',
        action='store_true',
        help='Skip Wikipedia scraping'
    )
    parser.add_argument(
        '--ingest-only',
        action='store_true',
        help='Only ingest existing data (skip scraping)'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 STARTING AUTOMATED TRANSPORT DATA SCRAPING")
        logger.info(f"Knowledge Base Directory: {args.kb_dir}")
        logger.info(f"ChromaDB Collection: {args.collection}")
        
        # Step 1: Scrape OpenStreetMap
        if not args.ingest_only and not args.skip_osm:
            scrape_osm_data(args.kb_dir)
        else:
            logger.info("\n⏭️  Skipping OSM scraping")
        
        # Step 2: Scrape Wikipedia
        if not args.ingest_only and not args.skip_wikipedia:
            scrape_wikipedia_data(args.kb_dir)
        else:
            logger.info("\n⏭️  Skipping Wikipedia scraping")
        
        # Step 3: Ingest data
        stats = ingest_data(args.kb_dir, args.collection)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL DONE!")
        logger.info("=" * 60)
        logger.info(f"\n✅ Successfully built Sri Lankan transport knowledge base")
        logger.info(f"   Total documents: {stats.document_count}")
        logger.info(f"\n🚀 Ready to use! Start the API with: python app.py")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️ Scraping interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
