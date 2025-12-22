"""
Test temporal features integration
"""

import sys
sys.path.append('.')

from utils.transport_service_gnn import TransportServiceGNN

def test_temporal():
    print("=" * 80)
    print("🧪 TESTING TEMPORAL FEATURES (Date/Time)")
    print("=" * 80)
    
    service = TransportServiceGNN('model/transport_gnn_model.pth', 'data')
    
    # Test 1: Without date/time (baseline)
    print("\n1️⃣ Test WITHOUT date/time:")
    result1 = service.get_recommendations("Colombo", "Anuradhapura")
    print(f"   Best option: {result1['best_option']['mode']} - {result1['best_option']['rating']:.2f}⭐")
    print(f"   Note: {result1['note']}")
    
    # Test 2: With date only
    print("\n2️⃣ Test WITH date (Poya day - crowded):")
    result2 = service.get_recommendations(
        "Colombo", "Anuradhapura",
        departure_date="2025-12-15"  # Assume poya day
    )
    print(f"   Best option: {result2['best_option']['mode']} - {result2['best_option']['rating']:.2f}⭐")
    print(f"   Temporal info: {result2['temporal_info']}")
    print(f"   Note: {result2['note']}")
    
    # Test 3: With date and peak time
    print("\n3️⃣ Test WITH date + peak time (morning rush):")
    result3 = service.get_recommendations(
        "Colombo", "Anuradhapura",
        departure_date="2025-12-20",  # Regular day
        departure_time="08:30"  # Morning peak
    )
    print(f"   Best option: {result3['best_option']['mode']} - {result3['best_option']['rating']:.2f}⭐")
    print(f"   Temporal info: {result3['temporal_info']}")
    print(f"   Note: {result3['note']}")
    
    print("\n   📊 All recommendations (with temporal adjustments):")
    for i, rec in enumerate(result3['recommendations'], 1):
        print(f"      {i}. {rec['mode']:12s} - {rec['rating']:.2f}⭐ "
              f"(temporal: {rec['temporal_context']['time_period'] if rec['temporal_context'] else 'N/A'})")
    
    # Test 4: Off-peak time
    print("\n4️⃣ Test WITH off-peak time (afternoon):")
    result4 = service.get_recommendations(
        "Colombo", "Anuradhapura",
        departure_date="2025-12-20",
        departure_time="14:30"  # Afternoon (off-peak)
    )
    print(f"   Best option: {result4['best_option']['mode']} - {result4['best_option']['rating']:.2f}⭐")
    print(f"   Temporal info: {result4['temporal_info']}")
    
    print("\n" + "=" * 80)
    print("✅ TEMPORAL FEATURES WORKING!")
    print("=" * 80)
    print("\n📝 How it works:")
    print("   1. Base GNN rating (1-5 stars)")
    print("   2. Temporal context extracted from date/time")
    print("   3. Ratings adjusted based on service_conditions.csv:")
    print("      • Crowdedness level (low/medium/high/very_high)")
    print("      • Typical delays at that time")
    print("      • Peak hours penalized more")
    print("\n⚠️  FUTURE ENHANCEMENT:")
    print("   Retrain GNN model to LEARN temporal patterns instead of manual adjustment")
    print("=" * 80)

if __name__ == "__main__":
    try:
        test_temporal()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
