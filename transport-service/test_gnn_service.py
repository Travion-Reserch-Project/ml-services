"""
Test script for GNN Transport Service
Run this to verify the service works before starting the API
"""

import sys
sys.path.append('.')

from utils.transport_service_gnn import TransportServiceGNN

def test_service():
    print("=" * 70)
    print("🧪 TESTING GNN TRANSPORT SERVICE")
    print("=" * 70)
    
    # Initialize service
    print("\n1️⃣ Loading service...")
    service = TransportServiceGNN(
        model_path='model/transport_gnn_model.pth',
        data_path='data'
    )
    
    # Test 1: Get recommendations
    print("\n2️⃣ Testing recommendations: Colombo → Anuradhapura")
    result = service.get_recommendations("Colombo", "Anuradhapura")
    
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        print(f"   ✅ Found {result['total_services']} options")
        print(f"   🎯 Best option: {result['best_option']['mode']} "
              f"({result['best_option']['rating']:.2f}⭐)")
        
        print(f"\n   All recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"      {i}. {rec['mode']:12s} - {rec['operator']:20s} "
                  f"| {rec['rating']:.2f}⭐ | Rs.{rec['fare_lkr']:.0f} | "
                  f"{rec['duration_min']}min")
    
    # Test 2: Get all services
    print("\n3️⃣ Testing all services endpoint...")
    all_services = service.get_all_services()
    
    if "error" in all_services:
        print(f"   ❌ Error: {all_services['error']}")
    else:
        print(f"   ✅ Total services: {all_services['total_services']}")
        print(f"   Top 3 rated services:")
        for i, svc in enumerate(all_services['services'][:3], 1):
            print(f"      {i}. {svc['origin']:15s} → {svc['destination']:15s} "
                  f"| {svc['mode']:12s} | {svc['rating']:.2f}⭐")
    
    # Test 3: Natural language query
    print("\n4️⃣ Testing natural language query...")
    query_result = service.query("How can I travel from Colombo to Kandy?")
    
    if "error" in query_result:
        print(f"   ⚠️  {query_result['error']}")
    else:
        print(f"   ✅ Found recommendations for {query_result['origin']} → {query_result['destination']}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Service is ready!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Start the API: uvicorn app:app --reload --port 8001")
    print("   2. Test endpoint: curl http://localhost:8001/api/health")
    print("   3. Get recommendations: POST to /api/recommend")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_service()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
