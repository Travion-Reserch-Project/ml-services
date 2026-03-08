#!/usr/bin/env python3
"""
Seed script to populate the knowledge base with initial bus fare data
"""

import requests
import json

API_URL = "http://localhost:8001/api/knowledge"

# Sample bus fare data for Sri Lanka
bus_fares = [
    {
        "id": "fare_colombo_galle_001",
        "type": "bus_fare",
        "content": "Bus fare from Colombo to Galle is LKR 250 for normal service, LKR 350 for semi-luxury, and LKR 450 for luxury service. The journey is approximately 119 kilometers and takes about 2-3 hours depending on traffic. (කොළඹ සිට ගාල්ල දක්වා)",
        "metadata": {
            "origin": "Colombo",
            "destination": "Galle",
            "origin_en": "Colombo",
            "destination_en": "Galle",
            "origin_si": "කොළඹ",
            "destination_si": "ගාල්ල",
            "fare_normal": 250.0,
            "fare_semi_luxury": 350.0,
            "fare_luxury": 450.0,
            "distance_km": 119.0,
            "duration_hours": 2.5,
            "language": "en",
            "source": "ntc_2026",
            "category": "fare",
            "verified": True
        }
    },
    {
        "id": "fare_colombo_galle_002_si",
        "type": "bus_fare",
        "content": "කොළඹ සිට ගාල්ල දක්වා බස් ගාස්තුව: සාමාන්‍ය සේවාව රුපියල් 250, අර්ධ සුඛෝපභෝගී රුපියල් 350, සුඛෝපභෝගී සේවාව රුපියල් 450. දුර කිලෝමීටර 119 ක් වන අතර ගමන් කාලය පැය 2-3 අතර වේ.",
        "metadata": {
            "origin": "කොළඹ",
            "destination": "ගාල්ල",
            "origin_en": "Colombo",
            "destination_en": "Galle",
            "origin_si": "කොළඹ",
            "destination_si": "ගාල්ල",
            "fare_normal": 250.0,
            "fare_semi_luxury": 350.0,
            "fare_luxury": 450.0,
            "distance_km": 119.0,
            "duration_hours": 2.5,
            "language": "si",
            "source": "ntc_2026",
            "category": "fare",
            "verified": True
        }
    },
    {
        "id": "fare_colombo_kandy_001",
        "type": "bus_fare",
        "content": "Bus fare from Colombo to Kandy is LKR 180 for normal service, LKR 280 for semi-luxury, and LKR 380 for luxury service. Distance is 115 kilometers, travel time is approximately 3-4 hours via Kadugannawa route. (කොළඹ සිට මහනුවර දක්වා)",
        "metadata": {
            "origin": "Colombo",
            "destination": "Kandy",
            "origin_en": "Colombo",
            "destination_en": "Kandy",
            "origin_si": "කොළඹ",
            "destination_si": "මහනුවර",
            "fare_normal": 180.0,
            "fare_semi_luxury": 280.0,
            "fare_luxury": 380.0,
            "distance_km": 115.0,
            "duration_hours": 3.5,
            "language": "en",
            "source": "ntc_2026",
            "category": "fare",
            "verified": True
        }
    },
    {
        "id": "fare_colombo_kandy_002_si",
        "type": "bus_fare",
        "content": "කොළඹ සිට මහනුවර දක්වා බස් ගාස්තුව: සාමාන්‍ය සේවාව රුපියල් 180, අර්ධ සුඛෝපභෝගී රුපියල් 280, සුඛෝපභෝගී සේවාව රුපියල් 380. කදුගන්නාව මාර්ගය හරහා කිලෝමීටර 115 ක දුරක් පැය 3-4 කින් ගමන් කරයි.",
        "metadata": {
            "origin": "කොළඹ",
            "destination": "මහනුවර",
            "origin_en": "Colombo",
            "destination_en": "Kandy",
            "origin_si": "කොළඹ",
            "destination_si": "මහනුවර",
            "fare_normal": 180.0,
            "fare_semi_luxury": 280.0,
            "fare_luxury": 380.0,
            "distance_km": 115.0,
            "duration_hours": 3.5,
            "language": "si",
            "source": "ntc_2026",
            "category": "fare",
            "verified": True
        }
    },
    {
        "id": "fare_colombo_jaffna_001",
        "type": "bus_fare",
        "content": "Bus fare from Colombo to Jaffna is LKR 850 for normal service, LKR 1200 for semi-luxury, and LKR 1500 for luxury service. Distance is approximately 396 kilometers, travel time is 8-10 hours via A9 highway. (කොළඹ සිට යාපනය දක්වා)",
        "metadata": {
            "origin": "Colombo",
            "destination": "Jaffna",
            "origin_en": "Colombo",
            "destination_en": "Jaffna",
            "origin_si": "කොළඹ",
            "destination_si": "යාපනය",
            "fare_normal": 850.0,
            "fare_semi_luxury": 1200.0,
            "fare_luxury": 1500.0,
            "distance_km": 396.0,
            "duration_hours": 9.0,
            "language": "en",
            "source": "ntc_2026",
            "category": "fare",
            "verified": True
        }
    },
    {
        "id": "fare_kandy_nuwara_eliya_001",
        "type": "bus_fare",
        "content": "Bus fare from Kandy to Nuwara Eliya is LKR 120 for normal service, LKR 180 for semi-luxury. Distance is 77 kilometers through scenic hill country, travel time is approximately 2.5-3 hours. (මහනුවර සිට නුවර එළිය දක්වා)",
        "metadata": {
            "origin": "Kandy",
            "destination": "Nuwara Eliya",
            "origin_en": "Kandy",
            "destination_en": "Nuwara Eliya",
            "origin_si": "මහනුවර",
            "destination_si": "නුවර එළිය",
            "fare_normal": 120.0,
            "fare_semi_luxury": 180.0,
            "distance_km": 77.0,
            "duration_hours": 2.75,
            "language": "en",
            "source": "ntc_2026",
            "category": "fare",
            "verified": True
        }
    },
    {
        "id": "fare_colombo_negombo_001",
        "type": "bus_fare",
        "content": "Bus fare from Colombo to Negombo is LKR 80 for normal service, LKR 120 for semi-luxury. Distance is 37 kilometers via Colombo-Katunayake Expressway or coastal road, travel time is 45 minutes to 1 hour. (කොළඹ සිට මීගමුව දක්වා)",
        "metadata": {
            "origin": "Colombo",
            "destination": "Negombo",
            "origin_en": "Colombo",
            "destination_en": "Negombo",
            "origin_si": "කොළඹ",
            "destination_si": "මීගමුව",
            "fare_normal": 80.0,
            "fare_semi_luxury": 120.0,
            "distance_km": 37.0,
            "duration_hours": 0.75,
            "language": "en",
            "source": "ntc_2026",
            "category": "fare",
            "verified": True
        }
    },
    {
        "id": "tip_peak_hours_001",
        "type": "travel_tip",
        "content": "Peak hours for intercity buses in Sri Lanka are 6-9 AM and 4-7 PM on weekdays. During these times, buses are more crowded and may take longer due to traffic. Consider traveling during off-peak hours for a more comfortable journey.",
        "metadata": {
            "language": "en",
            "source": "travel_guide",
            "category": "tip",
            "verified": True
        }
    },
    {
        "id": "tip_booking_advance_001",
        "type": "travel_tip",
        "content": "For long-distance luxury and semi-luxury buses (over 100km), it's recommended to book tickets 1-2 days in advance, especially during weekends and public holidays. Normal buses operate on a first-come, first-served basis.",
        "metadata": {
            "language": "en",
            "source": "travel_guide",
            "category": "tip",
            "verified": True
        }
    }
]

def upload_data():
    """Upload bus fare data to knowledge base"""
    print("=" * 60)
    print("🚀 Seeding Knowledge Base")
    print("=" * 60)
    
    # Upload documents
    try:
        response = requests.post(
            f"{API_URL}/upload",
            json={"documents": bus_fares},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Successfully uploaded {result['documents_added']} documents")
            print(f"  Collection: {result['collection']}")
            print(f"  Failed: {result['failed']}")
        else:
            print(f"\n✗ Upload failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False
    
    # Get stats
    try:
        response = requests.get(f"{API_URL}/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"\n📊 Knowledge Base Stats:")
            print(f"  Total documents: {stats['total_documents']}")
            print(f"  Collections: {len(stats['collections'])}")
            for col in stats['collections']:
                print(f"    - {col['name']}: {col['document_count']} docs")
        
    except Exception as e:
        print(f"\n⚠ Could not fetch stats: {e}")
    
    # Test search
    print(f"\n🔍 Testing search...")
    test_queries = [
        "What is the bus fare from Colombo to Galle?",
        "කොළඹ සිට ගාල්ලේ බස් ගාස්තුව කීයද?"
    ]
    
    for query in test_queries:
        try:
            response = requests.post(
                f"{API_URL}/search",
                json={"query": query, "top_k": 2, "use_rag": False},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n  Query: {query[:50]}...")
                print(f"  Results: {result['total_results']}")
                print(f"  Language: {result['query_language']}")
        except Exception as e:
            print(f"  ✗ Search failed: {e}")
    
    print(f"\n{'=' * 60}")
    print("✓ Seeding completed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    upload_data()
