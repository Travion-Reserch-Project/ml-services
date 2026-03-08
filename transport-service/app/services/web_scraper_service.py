"""
Web Scraper Service

Fetches general transport knowledge from web sources:
- Wikipedia articles on Sri Lankan roads, railways, transport
- Government websites for traffic rules and regulations
- Tourism websites for popular routes
- Social media data for transport tips

Uses BeautifulSoup for HTML parsing and httpx for async requests.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from app.schemas.knowledge import KnowledgeDocument, DocumentMetadata

logger = logging.getLogger(__name__)


class WebScraperService:
    """Service for scraping and processing web-based transport knowledge"""
    
    def __init__(self):
        """Initialize web scraper service"""
        logger.info("✓ Web scraper service initialized")
        
        # Predefined sources for transport data
        self.sources = {
            "wikipedia_roads": {
                "base_url": "https://en.wikipedia.org/wiki/",
                "articles": [
                    "Highways_of_Sri_Lanka",
                    "Road_network_of_Sri_Lanka",
                    "A6_road_(Sri_Lanka)",
                    "Rail_transport_in_Sri_Lanka",
                    "Transport_in_Sri_Lanka"
                ],
                "category": "road_network"
            },
            "wikipedia_tourism": {
                "base_url": "https://en.wikipedia.org/wiki/",
                "articles": [
                    "Tourism_in_Sri_Lanka",
                    "Tourist_attractions_in_Sri_Lanka"
                ],
                "category": "tourist_route"
            }
        }
    
    def get_sample_transport_data(self) -> List[KnowledgeDocument]:
        """
        Get curated transport knowledge - starting with sample data
        In production, this would scrape Wikipedia and other sources
        
        Returns:
            List of KnowledgeDocument objects with transport information
        """
        documents = []
        
        # Sample data for Sri Lankan transport system
        sample_data = [
            {
                "id": "road_network_001",
                "type": "road_network",
                "title": "Main Highway Networks in Sri Lanka",
                "content": """
Sri Lanka has an extensive road network with major highways:

A6 Highway (Colombo-Jaffna): The main north-south highway, approximately 430 km 
connecting Colombo in the southwest to Jaffna in the north. Passes through major 
cities including Kandy, Matara, and Galle.

A7 Highway (Southern Expressway): A modern expressway connecting Colombo to Matara, 
reducing travel time from 4-5 hours to about 2.5 hours. Important for southern tourism.

Colombo-Katunayake Highway: Connects Colombo airport to the city, facilitating 
international tourist and business travel.

Colombo-Negombo Road: Connects the capital to the beach town, popular with tourists.

Key characteristics:
- Total road network: ~120,000 km
- Left-hand traffic (British colonial influence)
- Monsoon seasons affect road conditions (Southwest: May-July, Northeast: December-February)
- Variable road quality in rural areas; improving in urban corridors
- Tolls on expressways
                """,
                "category": "road_network"
            },
            {
                "id": "railway_001",
                "type": "railway",
                "title": "Sri Lankan Railway System",
                "content": """
Sri Lanka Railways operates an old but extensive network:

Main Railway Lines:
1. Main Line: Colombo to Matara (264 km) - southernmost line, scenic coastal views
2. Northern Line: Colombo to Jaffna (354 km) 
3. Central Line: Colombo to Kandy (121 km) - important for central highlands

Railway Characteristics:
- Total network: ~1,500 km
- Narrow gauge (3 ft 6 in) except some sections
- Scenic routes: Kandy-Nuwara Eliya offers tea plantation views
- Trains are slower than buses but more affordable for long distances
- Popular tourist attraction: Kandy-Ella scenic mountain railway
- Dining and sleeping cars available on main routes

Travel Tips:
- Book in advance during peak seasons (December-February, June-August)
- Trains often run late; plan accordingly
- First-class has air conditioning; second/third class are Budget options
- Platform 1 Colombo is the main railway station
                """,
                "category": "railway"
            },
            {
                "id": "tourist_routes_001",
                "type": "tourist_route",
                "title": "Popular Tourist Routes in Sri Lanka",
                "content": """
Classic Sri Lankan tourist circuits:

1. Western Route (3-4 days):
   Colombo → Negombo → Mount Lavinia → Galle Fort → Matara
   Best for beaches and colonial heritage

2. Southern Coast (2-3 days):
   Galle → Mirissa → Unawatuna → Weligama
   Whale watching (November-April), stunning beaches, water sports

3. Cultural Triangle (3-4 days):
   Colombo → Kandy → Sigiriya → Polonnaruwa → Dambulla
   Ancient temples, cultural sites, historical monuments

4. Central Highlands (2-3 days):
   Kandy → Nuwara Eliya → Ella → Haputale
   Tea plantations, scenic mountain views, cool climate

5. East Coast (2-3 days):
   Arugam Bay → Trincomalee → Batticaloa
   Less crowded beaches, surfing, water sports

Transportation Options:
- Private taxi/car with driver: Most expensive but flexible
- Inter-city buses: Budget-friendly, frequent departures
- Train: Slower but scenic and economical
- Tuk-tuk (auto-rickshaw): Short distances within cities
                """,
                "category": "tourist_route"
            },
            {
                "id": "traffic_rules_001",
                "type": "traffic_rules",
                "title": "Sri Lankan Traffic Rules and Regulations",
                "content": """
Key traffic rules for driving in Sri Lanka:

Driving Rules:
- Drive on the LEFT (British legacy)
- Speed limits: 50 km/h in cities, 70 km/h on highways, 60 km/h on toll roads
- Seatbelts mandatory for drivers and front passengers
- No mobile phone use while driving
- Strictly no drink driving
- Horns are used frequently; it's normal

Motorcycle/Scooter Rules:
- Helmets mandatory for all riders
- Don't ride against traffic
- Watch for loose animals on rural roads
- Be cautious of potholes and uneven surfaces

Road Conditions and Hazards:
- Monsoon rains cause landslides and poor visibility
- Many vehicles lack working headlights/taillights
- Stray animals common on rural roads
- Pedestrian crossings not always marked
- Traffic lights presence varies by city

Common Violations and Fines:
- No documents: Heavy penalty
- Speeding: Police issue on-the-spot fines
- Without helmet: Fine Rs. 500-1000
- Without seatbelt: Fine Rs. 1500+

Useful Contacts:
- Police: 119
- Traffic Police: 011-2 423423
- Road accidents: Call nearest police station
                """,
                "category": "traffic_rules"
            },
            {
                "id": "transport_tips_001",
                "type": "transport_tip",
                "title": "Practical Transport Tips for Travelers",
                "content": """
Helpful tips for getting around Sri Lanka:

Bus Travel:
- Government and private buses operate throughout the island
- Private (intercity) buses faster than government buses; slightly higher cost
- Long-distance buses often have air conditioning and onboard refreshments
- Best to book seats in advance during peak tourist season
- Timetables are suggestions; buses leave when full
- Keep belongings close in crowded buses

Train Travel:
- Book reserved seats in advance if possible
- Arrive 30 minutes early; platforms announced shortly before departure
- Scenic routes popular - book window seats
- Tea/snacks available on most trains
- Night trains have sleeping berths; book well ahead

Taxi and Tuk-tuk:
- Tuk-tuk fares should be negotiated before starting journey in many areas
- Uber/Grab operate in Colombo and major cities
- Pre-arranged hotel taxis are safer for solo travelers
- Avoid unmarked taxis
- Always note registration number

Renting Vehicles:
- International driving permit recommended (sometimes mandatory)
- Rental companies available in tourist areas
- Insurance check important due to road conditions
- Fuel stations readily available in major towns
- Remember: LEFT side driving

Safety Tips:
- Travel during daylight hours on unfamiliar routes
- Check security advisories for current regions
- Keep travel documents separate
- Travel in small groups when possible
- Trust your instincts
                """,
                "category": "transport_tip"
            },
            {
                "id": "colombo_transport_001",
                "type": "general",
                "title": "Getting Around Colombo City",
                "content": """
Transportation options in Colombo, Sri Lanka's capital:

City Buses:
- Red buses run extensive network across Colombo
- Affordable (Rs. 20-100 depending on distance)
- Frequent during rush hours; less so after 8 PM
- Crowded in peak times; pickpocketing risk on packed buses

Tuk-tuk (Auto-rickshaw):
- Three-wheeled taxis, abundant throughout the city
- Negotiate fare before boarding (or use meter if available)
- Typical fares: Rs. 100-300 for short distances
- Colorfully decorated; common form of local transport

Taxi:
- Modern taxi services like Uber and Grab available
- Traditional black taxis less reliable; price varies
- Uber typically Rs. 100-500 depending on distance
- Grab sometimes cheaper than Uber

Walking:
- Central Colombo can be navigated on foot
- Foot pavements sometimes narrow/uneven
- Heat and humidity make walking tiring for tourists
- Best during early morning coolness

Recent Development:
- Colombo bus rapid transit (BRT) system in planning stages
- Colombo metro concept being explored
- Website: www.transport.gov.lk for planning updates

Key Locations:
- Fort Station: Main railway hub
- Central Bus Stand: Intercity buses
- Pettah: Busy commercial area with congestion
- Galle Face: Waterfront promenade
- Taj Samudra and surrounding hotels: Tourist area
                """,
                "category": "general"
            }
        ]
        
        # Convert to KnowledgeDocument objects
        for item in sample_data:
            metadata = DocumentMetadata(
                transport_type="multi",
                category=item.get("category"),
                language="en",
                source="curated_knowledge",
                last_updated=datetime.now().isoformat(),
                verified=True,
                tags=[item.get("category", "transport")]
            )
            
            doc = KnowledgeDocument(
                id=item["id"],
                type=item.get("type", "general"),
                content=item["content"].strip(),
                metadata=metadata,
                created_at=datetime.now()
            )
            documents.append(doc)
        
        logger.info(f"✓ Generated {len(documents)} sample transport documents")
        return documents
    
    def get_bilingual_data(self) -> List[KnowledgeDocument]:
        """
        Get bilingual transport knowledge (English + Sinhala)
        Useful for local travelers
        
        Returns:
            List of bilingual KnowledgeDocument objects
        """
        documents = []
        
        # Sample bilingual content
        bilingual_data = [
            {
                "id": "common_phrases_001",
                "type": "general",
                "title_en": "Common Transportation Phrases",
                "title_si": "ගතිකරණ සම්මතයි",
                "content": """
Common Transportation Phrases:

How to ask for directions?
- English: "How do I get to the railway station?"
- Sinhala: "පිලිවෙත් ස්ටේෂනයට මම එයා ගියුත් ඕනෑ?" (Pilivethe steshane yata mama eya giyuth oina?)

How much is the fare?
- English: "How much is the fare to Galle?"
- Sinhala: "ගාල්ලට තිස්සේ කීයද?" (Galleta tisse kiyada?)

Where is the bus station?
- English: "Where is the intercity bus station?"
- Sinhala: "නගර අතර බස් නැවතුම් ස්ථානය කොතැනින්ද?" (Nagara atra bas navetum sthanaya kotaninida?)

Common directions:
- Straight: ඉកුත්ව (Ikuthva)
- Turn right: දකුණට돌 (Dakunta dol)
- Turn left: වමට돌 (Vamat dol)
- Near: ළඟින් (Lagin)
- Far: දුරින් (Durin)
                """,
                "category": "transport_tip"
            },
            {
                "id": "city_names_001",
                "type": "general",
                "title_en": "Major Cities and Towns",
                "title_si": "ප්‍රධාන නගර සහ නsettlement",
                "content": """
Major Cities and Their Names:

English ↔ Sinhala:
- Colombo (කොළඹ - Kolumba)
- Kandy (මහනුවර - Mahanuwara) 
- Galle (ගාල්ල - Galla)
- Matara (මාතර - Matara)
- Negombo (නෙගොම්බ - Negomba)
- Jaffna (යාපනය - Yapanaya)
- Trincomalee (ත්‍රිකුණාමලේ - Trikunamale)
- Kurunegala (කුරුණෑගල - Kurunaegala)
- Anuradhapura (අනුරාධපුරය - Anuradhaputaya)
- Ratnapura (රත්නපුර - Ratnapura)

Quick Reference for Travelers:
Use these names when asking for directions or booking transport.
                """,
                "category": "general"
            }
        ]
        
        for item in bilingual_data:
            content = item.get("content", "").strip()
            metadata = DocumentMetadata(
                transport_type="multi",
                category=item.get("category"),
                language="mixed",
                source="curated_knowledge",
                last_updated=datetime.now().isoformat(),
                verified=True,
                tags=["bilingual", item.get("category", "transport")]
            )
            
            doc = KnowledgeDocument(
                id=item["id"],
                type=item.get("type", "general"),
                content=content,
                metadata=metadata,
                created_at=datetime.now()
            )
            documents.append(doc)
        
        logger.info(f"✓ Generated {len(documents)} bilingual transport documents")
        return documents
