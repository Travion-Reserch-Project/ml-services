"""
Test script for NLP Parser
Run this to verify BERT integration is working
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nlp_parser import TransportQueryParser

def test_parser():
    """Test the NLP parser with various queries."""
    
    print("🧪 Testing Transport NLP Parser with BERT\n")
    print("=" * 80)
    
    # Initialize parser
    print("\n📥 Initializing parser...")
    parser = TransportQueryParser(use_bert=True)
    print("✅ Parser initialized\n")
    
    # Test queries
    test_cases = [
        {
            "query": "I want to go from Kandy to Colombo at 2pm",
            "expected": {
                "origin": "Kandy",
                "destination": "Colombo",
                "time": 14.0
            }
        },
        {
            "query": "Bus from Galle to Colombo tomorrow morning",
            "expected": {
                "origin": "Galle",
                "destination": "Colombo",
                "mode": "bus",
                "time": 8.0
            }
        },
        {
            "query": "Train tickets from Colombo to Ella leaving after 3pm",
            "expected": {
                "origin": "Colombo",
                "destination": "Ella",
                "mode": "train",
                "time": 15.0
            }
        },
        {
            "query": "How do I get to Kandy from Negombo?",
            "expected": {
                "origin": "Negombo",
                "destination": "Kandy"
            }
        },
        {
            "query": "Show me buses to Galle this evening",
            "expected": {
                "destination": "Galle",
                "mode": "bus",
                "time": 18.0
            }
        },
        {
            "query": "Tuk-tuk to the airport from Colombo at 6:30am",
            "expected": {
                "origin": "Colombo",
                "mode": "tuk-tuk",
                "time": 6.5
            }
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test['query']}")
        print("-" * 80)
        
        # Parse query
        result = parser.parse(test['query'])
        
        # Display results
        print(f"📊 Parsed Results:")
        print(f"   Origin:         {result['origin']}")
        print(f"   Destination:    {result['destination']}")
        print(f"   Departure Time: {result['departure_time']}")
        print(f"   Mode:           {result['mode']}")
        print(f"   Date:           {result['date']}")
        
        # Validate
        is_valid, error = parser.validate_query(result)
        if is_valid:
            print(f"✅ Valid query")
            passed += 1
        else:
            print(f"❌ Invalid: {error}")
            failed += 1
        
        # Check expected values
        expected = test['expected']
        matches = []
        mismatches = []
        
        for key, expected_value in expected.items():
            actual_value = result.get(key)
            if actual_value == expected_value:
                matches.append(f"✓ {key}")
            else:
                mismatches.append(f"✗ {key}: expected '{expected_value}', got '{actual_value}'")
        
        if matches:
            print(f"   Matches: {', '.join(matches)}")
        if mismatches:
            print(f"   Mismatches: {', '.join(mismatches)}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"\n📈 Test Summary:")
    print(f"   Total tests:  {len(test_cases)}")
    print(f"   ✅ Passed:    {passed}")
    print(f"   ❌ Failed:    {failed}")
    print(f"   Success rate: {passed/len(test_cases)*100:.1f}%")
    
    return passed == len(test_cases)


if __name__ == "__main__":
    success = test_parser()
    exit(0 if success else 1)
