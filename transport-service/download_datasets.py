"""
Dataset Downloader

Downloads real Sri Lankan transport datasets from various sources:
- GIS/Map data from NSDI
- Railway statistics
- GTFS data
- Transport reports
- Traffic sign datasets
- Tourism data

Run this script to populate the kb_transport directory with real data.
"""

import logging
import sys
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Downloads and manages Sri Lankan transport datasets"""
    
    def __init__(self, kb_base: str = "kb_transport"):
        self.kb_base = Path(kb_base)
        self.kb_base.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and metadata
        self.datasets = {
            "gis": {
                "name": "National Spatial Data Infrastructure - Transport Layers",
                "urls": [
                    "https://data.gov.lk/dataset/national-spatial-data-infrastructure-transport",
                    "https://www.survey.gov.lk/sdiv/geonetwork/srv/api"
                ],
                "description": "Roads, railways, airports, ports, transport layers",
                "format": "WMS / GeoJSON / Shapefile",
                "manual_download": True,
                "instructions": "Visit NSDI portal and download transport WMS layers"
            },
            
            "roads": {
                "name": "World Bank Road Network Shapefile",
                "urls": [
                    "https://datacatalog.worldbank.org/search/dataset/sri-lanka-roads",
                    "https://energydata.info/dataset/road-network-of-sri-lanka"
                ],
                "description": "Class A/B roads, national road master plan",
                "format": "Shapefile / GeoJSON",
                "manual_download": True
            },
            
            "gis_free": {
                "name": "Free Sri Lanka GIS Data",
                "urls": [
                    "https://github.com/theagiletekproj/srilanka-geojson",
                    "https://data.humdata.org/dataset/sri-lanka-administrative-boundaries"
                ],
                "description": "Admin boundaries, provinces, districts",
                "format": "GeoJSON",
                "manual_download": False,
                "note": "Can clone GitHub repo directly"
            },
            
            "railway_stats": {
                "name": "Transport Statistics Dataset",
                "urls": [
                    "https://data.gov.lk/dataset/transport-statistics",
                    "https://data.gov.lk/dataset/department-of-railway-statistics"
                ],
                "description": "Rail track length, passenger data, bus data",
                "format": "CSV / Excel",
                "manual_download": True
            },
            
            "gtfs": {
                "name": "GTFS Research Dataset",
                "urls": [
                    "https://zenodo.org/record/3630623"
                ],
                "description": "Stops, routes, trips, timetables, transfers, shapes",
                "format": "GTFS (txt files)",
                "manual_download": True,
                "note": "Research dataset - may need academic access"
            },
            
            "transport_reports": {
                "name": "Public Transport Reports",
                "urls": [
                    "https://data.gov.lk/dataset/public-transport-report",
                    "https://www.transport.gov.lk/web/"
                ],
                "description": "Bus operational data, transport sector statistics",
                "format": "PDF / CSV",
                "manual_download": True
            },
            
            "traffic_signs": {
                "name": "Sri Lanka Traffic Sign Dataset",
                "urls": [
                    "https://www.kaggle.com/datasets/sujithknn/sri-lanka-traffic-sign-dataset"
                ],
                "description": "70 traffic sign classes, 10k+ images",
                "format": "Images + JSON",
                "manual_download": True,
                "note": "Kaggle dataset - need API key"
            },
            
            "documents": {
                "name": "Large Sri Lanka Document Dataset",
                "urls": [
                    "https://huggingface.co/datasets/sinhala-nlp/sinhala_dataset"
                ],
                "description": "200k+ documents - tourism, law, policy, news",
                "format": "Text / JSON",
                "manual_download": True,
                "note": "Filter for transport/tourism related only"
            }
        }
    
    def show_datasets(self):
        """Display all available datasets with download instructions"""
        logger.info("=" * 80)
        logger.info("📦 AVAILABLE SRI LANKAN TRANSPORT DATASETS")
        logger.info("=" * 80)
        
        for category, info in self.datasets.items():
            logger.info(f"\n📋 {category.upper()}: {info['name']}")
            logger.info(f"   Description: {info['description']}")
            logger.info(f"   Format: {info['format']}")
            logger.info(f"   Manual Download: {'Yes' if info.get('manual_download') else 'No'}")
            
            logger.info(f"   URLs:")
            for url in info['urls']:
                logger.info(f"     - {url}")
            
            if info.get('note'):
                logger.info(f"   Note: {info['note']}")
            
            if info.get('instructions'):
                logger.info(f"   Instructions: {info['instructions']}")
    
    def create_dataset_manifest(self):
        """Create a JSON manifest file with all dataset information"""
        manifest_path = self.kb_base / "dataset_manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(self.datasets, f, indent=2)
        
        logger.info(f"✓ Dataset manifest created: {manifest_path}")
    
    def create_download_instructions(self):
        """Create markdown file with detailed download instructions"""
        instructions_path = self.kb_base / "DOWNLOAD_INSTRUCTIONS.md"
        
        content = [
            "# Sri Lankan Transport Dataset Download Instructions\n",
            "This guide explains how to download and prepare real datasets for the RAG system.\n",
            "---\n"
        ]
        
        for category, info in self.datasets.items():
            content.append(f"\n## {category.upper()}: {info['name']}\n")
            content.append(f"**Description:** {info['description']}\n")
            content.append(f"**Format:** {info['format']}\n")
            content.append(f"**Target Directory:** `kb_transport/{category}/`\n")
            
            content.append("\n**Download URLs:**\n")
            for url in info['urls']:
                content.append(f"- {url}\n")
            
            if info.get('manual_download'):
                content.append("\n**Download Steps:**\n")
                content.append("1. Visit the URL above\n")
                content.append("2. Download the dataset (may require registration)\n")
                content.append(f"3. Extract files to `kb_transport/{category}/`\n")
                content.append("4. Run ingestion script\n")
            
            if info.get('note'):
                content.append(f"\n**Note:** {info['note']}\n")
            
            if info.get('instructions'):
                content.append(f"\n**Special Instructions:** {info['instructions']}\n")
            
            content.append("\n---\n")
        
        with open(instructions_path, 'w') as f:
            f.write("".join(content))
        
        logger.info(f"✓ Download instructions created: {instructions_path}")
    
    def create_sample_data_structure(self):
        """Create example file structure to show expected format"""
        examples = {
            "gis/roads.geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "road_name": "A6 Highway",
                            "road_class": "A",
                            "length_km": 430
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[80.0, 7.0], [80.1, 7.1]]
                        }
                    }
                ]
            },
            
            "railway/routes.json": {
                "routes": [
                    {
                        "route_id": "main_line_001",
                        "route_name": "Colombo - Matara Main Line",
                        "distance_km": 264
                    }
                ]
            },
            
            "bus/routes.csv": "route_number,origin,destination,operator\\n138,Colombo,Matara,SLTB\\n",
            
            "rules/road_rules.txt": "Traffic Rules for Sri Lanka:\\n1. Drive on the left side\\n2. Speed limit: 50km/h in cities...",
            
            "tourism/scenic_routes.md": "# Scenic Train Routes\\n## Kandy to Ella\\nOne of the most beautiful train journeys..."
        }
        
        for file_path, content in examples.items():
            full_path = self.kb_base / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(content, dict):
                with open(full_path, 'w') as f:
                    json.dump(content, f, indent=2)
            else:
                with open(full_path, 'w') as f:
                    f.write(content)
        
        logger.info(f"✓ Sample data structure created in {self.kb_base}")


def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("🚀 SRI LANKAN TRANSPORT DATASET DOWNLOADER")
    logger.info("=" * 80)
    
    downloader = DatasetDownloader()
    
    # Show all available datasets
    downloader.show_datasets()
    
    # Create helpful files
    logger.info("\n📝 Creating helper files...")
    downloader.create_dataset_manifest()
    downloader.create_download_instructions()
    downloader.create_sample_data_structure()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ SETUP COMPLETE!")
    logger.info("=" * 80)
    logger.info("\n📋 Next steps:")
    logger.info("1. Read: kb_transport/DOWNLOAD_INSTRUCTIONS.md")
    logger.info("2. Download datasets from the URLs provided")
    logger.info("3. Place files in appropriate kb_transport/ subdirectories")
    logger.info("4. Run: python ingest_datasets.py")
    logger.info("\n💡 Tip: Start with GIS roads and railway data for best results")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
