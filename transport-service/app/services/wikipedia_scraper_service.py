"""
Wikipedia Scraper Service
Fetches transport-related articles from Wikipedia
- Roads, highways, expressways
- Railways and stations
- Bus transport
- Airports and seaports
- Tourism routes
"""

import requests
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WikipediaScraperService:
    """Scrape transport data from Wikipedia"""
    
    WIKI_API = "https://en.wikipedia.org/w/api.php"
    WIKI_BASE = "https://en.wikipedia.org/wiki/"
    
    # Transport-related Wikipedia articles for Sri Lanka
    TRANSPORT_ARTICLES = [
        # Roads and Highways
        "Transport_in_Sri_Lanka",
        "Road_signs_in_Sri_Lanka",
        "Roads_in_Sri_Lanka",
        "Expressways_in_Sri_Lanka",
        "Southern_Expressway_(Sri_Lanka)",
        "Colombo–Katunayake_Expressway",
        "Outer_Circular_Expressway_(Sri_Lanka)",
        "Central_Expressway_(Sri_Lanka)",
        
        # Railways
        "Sri_Lanka_Railways",
        "List_of_railway_stations_in_Sri_Lanka",
        "Railway_stations_in_Sri_Lanka",
        "Main_Line_(Sri_Lanka)",
        "Coastal_Line_(Sri_Lanka)",
        "Northern_Line_(Sri_Lanka)",
        "Puttalam_Line",
        "Kelani_Valley_Line",
        
        # Bus Transport
        "Sri_Lanka_Transport_Board",
        "National_Transport_Commission",
        "Bus_transport_in_Sri_Lanka",
        
        # Airports and Ports
        "Bandaranaike_International_Airport",
        "Ratmalana_Airport",
        "Mattala_Rajapaksa_International_Airport",
        "Port_of_Colombo",
        "Hambantota_Port",
        
        # Cities and Tourism
        "Colombo",
        "Kandy",
        "Galle",
        "Anuradhapura",
        "Sigiriya",
        
        # Rules and Regulations
        "Road_safety_in_Sri_Lanka",
        "Department_of_Motor_Traffic_(Sri_Lanka)",
    ]
    
    def __init__(self, output_dir: Path = Path("kb_transport")):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Transport-RAG-System/1.0 (Educational)'
        })
    
    def get_article_content(self, title: str) -> Optional[Dict]:
        """Fetch Wikipedia article content"""
        try:
            # Get article extract (plain text)
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts|info',
                'explaintext': True,
                'inprop': 'url'
            }
            
            response = self.session.get(self.WIKI_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            page = next(iter(pages.values()))
            
            if 'extract' not in page:
                logger.warning(f"No content found for: {title}")
                return None
            
            return {
                'title': page.get('title', title),
                'content': page.get('extract', ''),
                'url': page.get('fullurl', f"{self.WIKI_BASE}{title}"),
                'pageid': page.get('pageid', 0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching article '{title}': {e}")
            return None
    
    def get_article_sections(self, title: str) -> List[Dict]:
        """Get article broken into sections"""
        try:
            params = {
                'action': 'parse',
                'format': 'json',
                'page': title,
                'prop': 'sections|text',
                'disablelimitreport': True,
                'disableeditsection': True
            }
            
            response = self.session.get(self.WIKI_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'parse' not in data:
                return []
            
            sections = []
            html_content = data['parse']['text']['*']
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'sup', 'table']):
                tag.decompose()
            
            # Extract text by headings
            current_section = {'title': title, 'heading': 'Introduction', 'content': ''}
            
            for elem in soup.find_all(['h2', 'h3', 'p']):
                if elem.name in ['h2', 'h3']:
                    if current_section['content'].strip():
                        sections.append(current_section)
                    current_section = {
                        'title': title,
                        'heading': elem.get_text().strip(),
                        'content': ''
                    }
                elif elem.name == 'p':
                    text = elem.get_text().strip()
                    if text:
                        current_section['content'] += text + '\n\n'
            
            if current_section['content'].strip():
                sections.append(current_section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing sections for '{title}': {e}")
            return []
    
    def scrape_article(self, title: str) -> Optional[Dict]:
        """Scrape a single article with metadata"""
        logger.info(f"📄 Scraping: {title}")
        
        article = self.get_article_content(title)
        if not article:
            return None
        
        # Categorize article
        category = self._categorize_article(title)
        
        return {
            'title': article['title'],
            'content': article['content'],
            'url': article['url'],
            'source': 'Wikipedia',
            'category': category,
            'transport_type': self._get_transport_type(title),
            'language': 'en'
        }
    
    def _categorize_article(self, title: str) -> str:
        """Categorize article by topic"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['road', 'highway', 'expressway']):
            return 'roads'
        elif any(word in title_lower for word in ['railway', 'train', 'station', 'line']):
            return 'railway'
        elif any(word in title_lower for word in ['bus', 'transport board']):
            return 'bus'
        elif any(word in title_lower for word in ['airport', 'port', 'seaport']):
            return 'airports_ports'
        elif any(word in title_lower for word in ['safety', 'motor', 'traffic']):
            return 'rules'
        elif any(word in title_lower for word in ['colombo', 'kandy', 'galle', 'tourism']):
            return 'tourism'
        else:
            return 'general'
    
    def _get_transport_type(self, title: str) -> str:
        """Get transport type from title"""
        title_lower = title.lower()
        
        if 'road' in title_lower or 'highway' in title_lower:
            return 'road'
        elif 'rail' in title_lower or 'train' in title_lower:
            return 'railway'
        elif 'bus' in title_lower:
            return 'bus'
        elif 'airport' in title_lower or 'flight' in title_lower:
            return 'air'
        elif 'port' in title_lower or 'ship' in title_lower:
            return 'sea'
        else:
            return 'general'
    
    def scrape_all(self) -> List[Dict]:
        """Scrape all transport articles"""
        logger.info(f"📚 Starting Wikipedia scraping ({len(self.TRANSPORT_ARTICLES)} articles)...")
        
        articles = []
        for i, title in enumerate(self.TRANSPORT_ARTICLES, 1):
            try:
                article = self.scrape_article(title)
                if article:
                    articles.append(article)
                    logger.info(f"✅ [{i}/{len(self.TRANSPORT_ARTICLES)}] {title}")
                else:
                    logger.warning(f"⚠️ [{i}/{len(self.TRANSPORT_ARTICLES)}] Failed: {title}")
                
                # Rate limiting (be respectful to Wikipedia)
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error scraping '{title}': {e}")
                continue
        
        logger.info(f"✅ Scraped {len(articles)} Wikipedia articles")
        return articles
    
    def save_articles(self, articles: List[Dict], filename: str = "wikipedia_articles.json"):
        """Save articles to JSON file"""
        output_path = self.output_dir / "documents" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(articles)} articles to {output_path}")
        return output_path
    
    def save_by_category(self, articles: List[Dict]):
        """Save articles grouped by category"""
        # Group by category
        by_category = {}
        for article in articles:
            category = article['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(article)
        
        # Save each category
        for category, cat_articles in by_category.items():
            filename = f"wikipedia_{category}.json"
            output_path = self.output_dir / category / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cat_articles, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved {len(cat_articles)} articles to {category}/{filename}")


if __name__ == "__main__":
    # Test scraping
    logging.basicConfig(level=logging.INFO)
    
    scraper = WikipediaScraperService()
    
    # Scrape all articles
    articles = scraper.scrape_all()
    
    # Save all articles
    scraper.save_articles(articles)
    
    # Save by category
    scraper.save_by_category(articles)
    
    print("\n✅ Wikipedia scraping complete!")
    print(f"Total articles: {len(articles)}")
    
    # Summary by category
    by_cat = {}
    for article in articles:
        cat = article['category']
        by_cat[cat] = by_cat.get(cat, 0) + 1
    
    print("\nArticles by category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")
