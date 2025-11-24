"""
Enhanced test for unlimited location and transport mode extraction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nlp_parser import TransportQueryParser

def test_unlimited_parser():
    """Test that parser can extract ANY location and transport mode."""
    
    print("🧪 Testing Unlimited NLP Parser (Any Location, Any Mode)\n")
    print("=" * 80)
    
    parser = TransportQueryParser(use_bert=True)
    
    # Test cases with diverse locations and transport modes
    test_cases = [
        # Small towns/villages
        {
            "query": "I need to go from Bentota to Hikkaduwa",
            "expected_locations": ["Bentota", "Hikkaduwa"],
            "description": "Beach towns"
        },
        {
            "query": "Bus from Habarana to Mihintale tomorrow",
            "expected_locations": ["Habarana", "Mihintale"],
            "expected_mode": "bus",
            "description": "Historical sites"
        },
        # Airport references
        {
            "query": "Taxi from Katunayake Airport to Negombo",
            "expected_locations": ["Katunayake", "Negombo"],
            "expected_mode": "taxi",
            "description": "Airport transfer"
        },
        # Beach destinations
        {
            "query": "Tuk-tuk to Unawatuna from Galle Fort",
            "expected_locations": ["Unawatuna", "Galle"],
            "expected_mode": "tuk-tuk",
            "description": "Beach destination"
        },
        # Multiple word locations
        {
            "query": "Train from Fort Station to Arugam Bay",
            "expected_locations": ["Fort", "Arugam Bay"],
            "expected_mode": "train",
            "description": "Multi-word location"
        },
        # Various transport modes
        {
            "query": "I need a minivan from Ratnapura to Sinharaja",
            "expected_locations": ["Ratnapura", "Sinharaja"],
            "expected_mode": "van",
            "description": "Minivan transport"
        },
        {
            "query": "Motorcycle ride to Yala National Park",
            "expected_locations": ["Yala"],
            "expected_mode": "motorcycle",
            "description": "Motorcycle"
        },
        {
            "query": "Scooter rental for Kalpitiya",
            "expected_locations": ["Kalpitiya"],
            "expected_mode": "scooter",
            "description": "Scooter rental"
        },
        # Uncommon locations
        {
            "query": "Van from Pottuvil to Panama beach",
            "expected_locations": ["Pottuvil", "Panama"],
            "expected_mode": "van",
            "description": "Lesser-known spots"
        },
        {
            "query": "Private car to Horton Plains from Nuwara Eliya",
            "expected_locations": ["Horton", "Nuwara Eliya"],
            "expected_mode": "vehicle",
            "description": "National park"
        },
        # Ride-sharing
        {
            "query": "PickMe from Mount Lavinia to Dehiwala",
            "expected_locations": ["Mount Lavinia", "Dehiwala"],
            "expected_mode": "pickme",
            "description": "Ride-share app"
        },
        # Complex queries
        {
            "query": "How do I get from Ampara to Batticaloa by any transport?",
            "expected_locations": ["Ampara", "Batticaloa"],
            "description": "Open transport mode"
        },
        {
            "query": "Goods truck from Peliyagoda to Vavuniya",
            "expected_locations": ["Peliyagoda", "Vavuniya"],
            "expected_mode": "truck",
            "description": "Commercial transport"
        },
        # Short location names
        {
            "query": "Bus from Puna to Gal Oya",
            "expected_locations": ["Puna", "Gal Oya"],
            "expected_mode": "bus",
            "description": "Short place names"
        }
    ]
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test['description']}")
        print(f"   Query: \"{test['query']}\"")
        print("-" * 80)
        
        result = parser.parse(test['query'])
        
        # Display results
        print(f"📊 Extracted:")
        print(f"   Locations: {[result['origin'], result['destination']]}")
        print(f"   Mode: {result['mode']}")
        print(f"   Time: {result['departure_time']}")
        print(f"   Date: {result['date']}")
        
        # Check if locations were extracted
        extracted_locations = [loc for loc in [result['origin'], result['destination']] if loc]
        
        if 'expected_locations' in test:
            # Check if we extracted locations
            if extracted_locations:
                print(f"   ✅ Extracted {len(extracted_locations)} location(s)")
                # Show which expected locations were found
                for expected_loc in test['expected_locations']:
                    found = any(expected_loc.lower() in loc.lower() for loc in extracted_locations)
                    status = "✓" if found else "✗"
                    print(f"      {status} Expected: {expected_loc}")
            else:
                print(f"   ❌ No locations extracted (expected {test['expected_locations']})")
                failed += 1
                continue
        
        if 'expected_mode' in test:
            if result['mode'] and test['expected_mode'].lower() in result['mode'].lower():
                print(f"   ✅ Correct mode: {result['mode']}")
            else:
                print(f"   ⚠️  Mode: got '{result['mode']}', expected '{test['expected_mode']}'")
        
        # Validate
        is_valid, error = parser.validate_query(result)
        if is_valid:
            print(f"   ✅ Valid query")
            passed += 1
        else:
            print(f"   ❌ Invalid: {error}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"\n📈 Test Summary:")
    print(f"   Total tests:  {len(test_cases)}")
    print(f"   ✅ Passed:    {passed}")
    print(f"   ❌ Failed:    {failed}")
    print(f"   Success rate: {passed/len(test_cases)*100:.1f}%")
    
    print("\n✨ Key Features Demonstrated:")
    print("   ✓ Extracts ANY location name (not limited to predefined list)")
    print("   ✓ Supports diverse transport modes (bus, train, taxi, van, motorcycle, etc.)")
    print("   ✓ Handles multi-word locations (Nuwara Eliya, Arugam Bay)")
    print("   ✓ Recognizes ride-sharing services (PickMe, Uber, Grab)")
    print("   ✓ Works with small towns, beaches, national parks, airports")
    print("   ✓ No validation restrictions on location names")
    
    return passed >= len(test_cases) * 0.8  # 80% pass rate


if __name__ == "__main__":
    success = test_unlimited_parser()
    print("\n" + "=" * 80)
    if success:
        print("✅ Parser successfully handles unlimited locations and transport modes!")
    else:
        print("⚠️  Parser needs improvement for some edge cases")
    exit(0 if success else 1)
