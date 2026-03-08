"""
OpenStreetMap Scraper Service
Fetches transport data from OpenStreetMap using Overpass API
- Roads, highways, railways
- Bus stops, train stations
- Airports, ports
- Routes and ways
"""

import requests
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OSMScraperService:
    """Scrape transport data from OpenStreetMap for Sri Lanka"""
    
    # Overpass API endpoint (public, no API key needed)
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    
    # Sri Lanka bounding box: [south, west, north, east]
    SRI_LANKA_BBOX = [5.9, 79.5, 9.9, 81.9]
    
    def __init__(self, output_dir: Path = Path("kb_transport")):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Transport-RAG-System/1.0'
        })
    
    def _query_overpass(self, query: str, timeout: int = 180) -> Dict:
        """Execute Overpass API query"""
        try:
            response = self.session.post(
                self.OVERPASS_URL,
                data={'data': query},
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Overpass API error: {e}")
            return {"elements": []}
    
    def scrape_highways(self) -> List[Dict]:
        """Scrape major highways and roads"""
        logger.info("🛣️ Scraping highways from OpenStreetMap...")
        
        bbox = ",".join(map(str, self.SRI_LANKA_BBOX))
        query = f"""
        [out:json][timeout:180];
        (
          way["highway"="motorway"]({bbox});
          way["highway"="trunk"]({bbox});
          way["highway"="primary"]({bbox});
          way["highway"="secondary"]({bbox});
        );
        out body;
        >;
        out skel qt;
        """
        
        result = self._query_overpass(query)
        
        # Convert to GeoJSON format
        features = []
        ways = {elem['id']: elem for elem in result.get('elements', []) if elem['type'] == 'way'}
        nodes = {elem['id']: elem for elem in result.get('elements', []) if elem['type'] == 'node'}
        
        for way_id, way in ways.items():
            coordinates = []
            for node_id in way.get('nodes', []):
                if node_id in nodes:
                    node = nodes[node_id]
                    coordinates.append([node['lon'], node['lat']])
            
            if coordinates:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "osm_id": way_id,
                        "highway": way.get('tags', {}).get('highway', 'unknown'),
                        "name": way.get('tags', {}).get('name', 'Unnamed Road'),
                        "ref": way.get('tags', {}).get('ref', ''),
                        "surface": way.get('tags', {}).get('surface', ''),
                        "lanes": way.get('tags', {}).get('lanes', '')
                    }
                }
                features.append(feature)
        
        logger.info(f"✅ Scraped {len(features)} highway features")
        return features
    
    def scrape_railways(self) -> List[Dict]:
        """Scrape railway lines and stations"""
        logger.info("🚂 Scraping railways from OpenStreetMap...")
        
        bbox = ",".join(map(str, self.SRI_LANKA_BBOX))
        query = f"""
        [out:json][timeout:180];
        (
          way["railway"="rail"]({bbox});
          node["railway"="station"]({bbox});
          node["railway"="halt"]({bbox});
        );
        out body;
        >;
        out skel qt;
        """
        
        result = self._query_overpass(query)
        
        features = []
        ways = {elem['id']: elem for elem in result.get('elements', []) if elem['type'] == 'way'}
        nodes_dict = {elem['id']: elem for elem in result.get('elements', []) if elem['type'] == 'node'}
        
        # Railway lines
        for way_id, way in ways.items():
            coordinates = []
            for node_id in way.get('nodes', []):
                if node_id in nodes_dict:
                    node = nodes_dict[node_id]
                    coordinates.append([node['lon'], node['lat']])
            
            if coordinates:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "osm_id": way_id,
                        "type": "railway_line",
                        "railway": way.get('tags', {}).get('railway', 'rail'),
                        "name": way.get('tags', {}).get('name', 'Unnamed Railway'),
                        "usage": way.get('tags', {}).get('usage', 'main')
                    }
                }
                features.append(feature)
        
        # Railway stations
        for node in result.get('elements', []):
            if node['type'] == 'node' and 'tags' in node:
                if 'railway' in node['tags']:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [node['lon'], node['lat']]
                        },
                        "properties": {
                            "osm_id": node['id'],
                            "type": "railway_station",
                            "railway": node['tags'].get('railway', 'station'),
                            "name": node['tags'].get('name', 'Unnamed Station'),
                            "operator": node['tags'].get('operator', '')
                        }
                    }
                    features.append(feature)
        
        logger.info(f"✅ Scraped {len(features)} railway features")
        return features
    
    def scrape_bus_stops(self) -> List[Dict]:
        """Scrape bus stops and terminals"""
        logger.info("🚌 Scraping bus stops from OpenStreetMap...")
        
        bbox = ",".join(map(str, self.SRI_LANKA_BBOX))
        query = f"""
        [out:json][timeout:180];
        (
          node["highway"="bus_stop"]({bbox});
          node["amenity"="bus_station"]({bbox});
        );
        out body;
        """
        
        result = self._query_overpass(query)
        
        features = []
        for node in result.get('elements', []):
            if node['type'] == 'node':
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [node['lon'], node['lat']]
                    },
                    "properties": {
                        "osm_id": node['id'],
                        "type": "bus_stop",
                        "name": node.get('tags', {}).get('name', 'Unnamed Bus Stop'),
                        "network": node.get('tags', {}).get('network', ''),
                        "operator": node.get('tags', {}).get('operator', '')
                    }
                }
                features.append(feature)
        
        logger.info(f"✅ Scraped {len(features)} bus stop features")
        return features
    
    def scrape_airports(self) -> List[Dict]:
        """Scrape airports and airfields"""
        logger.info("✈️ Scraping airports from OpenStreetMap...")
        
        bbox = ",".join(map(str, self.SRI_LANKA_BBOX))
        query = f"""
        [out:json][timeout:180];
        (
          way["aeroway"="aerodrome"]({bbox});
          node["aeroway"="aerodrome"]({bbox});
        );
        out body;
        >;
        out skel qt;
        """
        
        result = self._query_overpass(query)
        
        features = []
        for elem in result.get('elements', []):
            if elem['type'] == 'node' and 'tags' in elem:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [elem['lon'], elem['lat']]
                    },
                    "properties": {
                        "osm_id": elem['id'],
                        "type": "airport",
                        "name": elem['tags'].get('name', 'Unnamed Airport'),
                        "iata": elem['tags'].get('iata', ''),
                        "icao": elem['tags'].get('icao', '')
                    }
                }
                features.append(feature)
        
        logger.info(f"✅ Scraped {len(features)} airport features")
        return features
    
    def scrape_all(self) -> Dict[str, List[Dict]]:
        """Scrape all transport data with rate limiting"""
        logger.info("🌍 Starting comprehensive OSM data scraping...")
        
        results = {}
        
        # Rate limiting between requests (Overpass API fair use)
        categories = [
            ('highways', self.scrape_highways),
            ('railways', self.scrape_railways),
            ('bus_stops', self.scrape_bus_stops),
            ('airports', self.scrape_airports)
        ]
        
        for category, scraper_func in categories:
            try:
                results[category] = scraper_func()
                time.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"Error scraping {category}: {e}")
                results[category] = []
        
        return results
    
    def save_geojson(self, features: List[Dict], filename: str, category_dir: str):
        """Save features as GeoJSON file"""
        output_path = self.output_dir / category_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "source": "OpenStreetMap",
                "license": "ODbL",
                "url": "https://www.openstreetmap.org/copyright"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(features)} features to {output_path}")
        return output_path


if __name__ == "__main__":
    # Test scraping
    logging.basicConfig(level=logging.INFO)
    
    scraper = OSMScraperService()
    
    # Scrape all data
    results = scraper.scrape_all()
    
    # Save to files
    for category, features in results.items():
        if features:
            scraper.save_geojson(features, f"{category}.geojson", "gis")
    
    print("\n✅ OSM scraping complete!")
    print(f"Total highways: {len(results.get('highways', []))}")
    print(f"Total railways: {len(results.get('railways', []))}")
    print(f"Total bus stops: {len(results.get('bus_stops', []))}")
    print(f"Total airports: {len(results.get('airports', []))}")
