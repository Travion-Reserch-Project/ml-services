"""
Quick test for the specific edge cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nlp_parser import TransportQueryParser

parser = TransportQueryParser(use_bert=True)

edge_cases = [
    "Motorcycle ride to Yala National Park",
    "Private car to Horton Plains from Nuwara Eliya",
]

print("\n🔍 Testing Edge Cases:\n")
for query in edge_cases:
    print(f"Query: '{query}'")
    result = parser.parse(query)
    print(f"  Locations: {[result['origin'], result['destination']]}")
    print(f"  Mode: {result['mode']}")
    print()
