"""
Lightweight API tests for CI/CD pipeline
Tests API structure without loading ML models
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that core modules can be imported"""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_app_structure():
    """Test that app.py can be read and has expected structure"""
    app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
    assert os.path.exists(app_path), "app.py not found"
    
    with open(app_path, 'r') as f:
        content = f.read()
        assert 'FastAPI' in content, "FastAPI not found in app.py"
        assert '/api/query' in content, "Query endpoint not found"
        assert '/health' in content, "Health endpoint not found"

def test_requirements():
    """Test that requirements.txt exists and has expected packages"""
    req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    assert os.path.exists(req_path), "requirements.txt not found"
    
    with open(req_path, 'r') as f:
        content = f.read()
        assert 'fastapi' in content.lower(), "FastAPI not in requirements"
        assert 'pydantic' in content.lower(), "Pydantic not in requirements"

if __name__ == "__main__":
    print("🧪 Running lightweight API tests\n")
    print("=" * 60)
    
    tests = [test_imports, test_app_structure, test_requirements]
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}: PASSED")
        except AssertionError as e:
            print(f"❌ {test.__name__}: FAILED - {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ {test.__name__}: ERROR - {e}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
