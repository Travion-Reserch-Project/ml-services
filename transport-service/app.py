from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="Transport Service API",
    description="AI-powered transport query service using BERT NLP + GNN predictions",
    version="1.0.0"
)

# Lazy loading of heavy dependencies
nlp_parser = None
transport_service = None

def get_nlp_parser():
    """Lazy load NLP parser."""
    global nlp_parser
    if nlp_parser is None:
        from utils.nlp_parser import TransportQueryParser
        nlp_parser = TransportQueryParser(use_bert=True)
    return nlp_parser

def get_transport_service():
    """Lazy load transport service."""
    global transport_service
    if transport_service is None:
        from utils.transport_service import TransportService
        model_path = 'model/transport_gnn_model.pth'
        data_path = 'data'
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}")
            return None
            
        transport_service = TransportService(model_path, data_path)
    return transport_service


class QueryRequest(BaseModel):
    """Request model for natural language queries."""
    query: str
    user_location: Optional[str] = None


class ServiceQuery(BaseModel):
    """Request model for structured service queries."""
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_time: Optional[float] = None
    mode: Optional[str] = None


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "Transport Service API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "parse": "/api/parse",
            "services": "/api/services"
        }
    }


@app.post("/api/query")
def process_natural_language_query(request: QueryRequest):
    """
    Process a natural language query and return transport options.
    
    Example queries:
    - "I want to go from Kandy to Colombo at 2pm"
    - "Bus from Galle to Colombo tomorrow morning"
    - "Train to Ella leaving after 3pm"
    """
    try:
        service = get_transport_service()
        
        if service is None:
            return {
                "success": False,
                "error": "Transport service not available (model not loaded)"
            }
        
        result = service.process_query(request.query)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse")
def parse_query(request: QueryRequest):
    """
    Parse a natural language query without fetching services.
    Useful for testing the NLP parser.
    
    Returns extracted entities: origin, destination, time, mode, date.
    """
    try:
        parser = get_nlp_parser()
        parsed = parser.parse(request.query)
        is_valid, error = parser.validate_query(parsed)
        
        return {
            "success": is_valid,
            "parsed": parsed,
            "error": error if not is_valid else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/services")
def get_services(query: ServiceQuery):
    """
    Get services using structured parameters (no NLP parsing).
    
    Parameters:
    - origin: Starting city
    - destination: Destination city
    - departure_time: Time in hours (e.g., 14.5 for 2:30pm)
    - mode: Transport mode ('bus', 'train', 'tuk-tuk')
    """
    try:
        service = get_transport_service()
        
        if service is None:
            return {
                "success": False,
                "error": "Transport service not available"
            }
        
        # Convert ServiceQuery to dict format expected by service
        parsed = {
            'origin': query.origin,
            'destination': query.destination,
            'departure_time': query.departure_time,
            'mode': query.mode,
            'date': 'today',
            'raw_query': ''
        }
        
        services = service._filter_services(parsed)
        
        return {
            "success": True,
            "services": services,
            "count": len(services)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    """Detailed health check with model and data status."""
    parser = get_nlp_parser()
    service = get_transport_service()
    
    # Check if model file exists
    model_exists = os.path.exists('model/transport_gnn_model.pth')
    
    return {
        "status": "healthy",
        "nlp_parser": "loaded" if parser else "not loaded",
        "gnn_model": "loaded" if (service and service.model) else ("not found" if not model_exists else "not loaded"),
        "data": "loaded" if (service and service.services_data is not None) else "not loaded",
        "note": "GNN model will be added via Git LFS after training" if not model_exists else None
    }


# Run with: uvicorn app:app --reload --port 8001
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
