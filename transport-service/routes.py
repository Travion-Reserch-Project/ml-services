import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.temporal_features import get_reliability_score, get_crowding_score

logger = logging.getLogger(__name__)

router = APIRouter()

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
    """Lazy load refactored temporal-aware GNN transport service."""
    global transport_service
    if transport_service is None:
        from utils.transport_service_gnn_refactored import TransportServiceGNNRefactored
        # Optionally fetch via MLflow registry at runtime
        model_source = os.getenv('MODEL_SOURCE', '').strip().lower()
        if model_source == 'mlflow':
            try:
                # Try to ensure the model checkpoint exists locally via MLflow
                from download_model import ensure_model_available_via_mlflow
                print("🌐 MODEL_SOURCE=mlflow: attempting to resolve model from registry...")
                ensure_model_available_via_mlflow()
            except Exception as e:
                print(f"⚠️  MLflow registry resolution failed: {e}")
        model_path = 'model/transport_gnn_routing.pth'
        data_path = 'data'
        # Always instantiate the service; it can operate without a model
        # (uses survey-based rules) and will still load CSV data.
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}. Service will use survey-based rules.")

        transport_service = TransportServiceGNNRefactored(model_path, data_path)
    return transport_service


class QueryRequest(BaseModel):
    """Request model for natural language queries."""
    query: str
    user_location: Optional[str] = None


class ServiceQuery(BaseModel):
    """Request model for structured service queries."""
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[str] = None  # Format: "2025-12-25"
    departure_time: Optional[str] = None  # Format: "14:30" or "2:30 PM"
    mode: Optional[str] = None


@router.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "Transport Service API with Temporal-Aware GNN",
        "status": "running",
        "version": "3.0.0",
        "description": "AI-powered transport recommendations using temporal features (date/time) + Graph Neural Networks",
        "features": {
            "temporal_awareness": "Considers day type (regular/weekend/poya/holiday) and time period (early_morning/morning/day/evening/night)",
            "survey_based_rules": "Uses survey data for bus/train availability, crowding patterns",
            "no_static_calendar": "Uses HolidayDetector API instead of calendar.csv",
            "no_service_conditions": "Uses learned rules instead of service_conditions.csv",
        },
        "endpoints": {
            "temporal_recommendations": "/api/recommend - Get best transport with date/time",
            "day_info": "/api/day-info - Get temporal info and reliability scores",
            "query": "/api/query - Natural language queries",
            "parse": "/api/parse - NLP parsing only",
            "services": "/api/services - Structured query",
            "all_services": "/api/all-services - All services with ratings",
            "health": "/api/health - System health check"
        },
        "example_usage": {
            "temporal_recommendation": {
                "endpoint": "/api/recommend",
                "body": {
                    "origin": "Colombo",
                    "destination": "Anuradhapura",
                    "departure_date": "2025-12-25",
                    "departure_time": "14:30"
                }
            },
            "check_day_info": {
                "endpoint": "/api/day-info?date=2025-12-25&time=14:30",
                "description": "Check if it's a poya/holiday and reliability patterns"
            }
        }
    }


@router.post("/api/query")
def process_natural_language_query(request: QueryRequest):
    """
    Process a natural language query and return transport options.

    Example queries:
    - "I want to go from Kandy to Colombo at 2pm"
    - "Bus from Galle to Colombo tomorrow morning"
    - "Train to Ella leaving after 3pm"
    """
    # ensure_models_downloaded()  # Disabled: using local model

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


@router.post("/api/parse")
def parse_query(request: QueryRequest):
    """
    Parse a natural language query without fetching services.
    Useful for testing the NLP parser.

    Returns extracted entities: origin, destination, time, mode, date.
    """
    # ensure_models_downloaded()  # Disabled: using local model

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


@router.post("/api/services")
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

        # Use GNN recommendations if both origin and destination provided
        if query.origin and query.destination:
            result = service.get_recommendations(
                query.origin, 
                query.destination,
                departure_date=query.departure_date,
                departure_time=query.departure_time
            )
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "available_locations": result.get("available_locations", [])
                }
            
            return {
                "success": True,
                "origin": result["origin"],
                "destination": result["destination"],
                "distance_km": result["distance_km"],
                "recommendations": result["recommendations"],
                "best_option": result["best_option"],
                "count": result["total_services"]
            }
        
        return {
            "success": False,
            "error": "Both origin and destination are required"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/recommend")
def get_recommendations(query: ServiceQuery):
    """
    Get GNN-powered transport recommendations with temporal awareness.
    
    Uses refactored temporal-aware Graph Neural Network to recommend best transport
    based on origin, destination, date, and time.
    
    Predicts the most reliable transport method considering:
    - Day type (regular, weekend, poya day, holiday)
    - Time period (early morning, morning, day, evening, night, late night)
    - Bus/train availability and crowding patterns from survey data
    
    Parameters:
    - origin: Starting location (e.g., "Colombo")
    - destination: Ending location (e.g., "Anuradhapura")
    - departure_date: Optional, format "2025-12-25" (uses API to detect holidays/poya)
    - departure_time: Optional, format "14:30" (24-hour)
    
    Returns recommendations sorted by reliability for the given time/day.
    """
    try:
        service = get_transport_service()
        
        if service is None:
            return {
                "success": False,
                "error": "Transport service not available. Service data could not be loaded."
            }
        
        if not query.origin or not query.destination:
            return {
                "success": False,
                "error": "Both origin and destination are required",
                "example": {
                    "origin": "Colombo",
                    "destination": "Anuradhapura",
                    "departure_date": "2025-12-25",
                    "departure_time": "14:30"
                }
            }
        
        result = service.get_recommendations(
            query.origin, 
            query.destination, 
            departure_date=query.departure_date,
            departure_time=query.departure_time,
            top_k=10
        )
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "available_locations": result.get("available_locations", [])
            }
        
        return {
            "success": True,
            "origin": result["origin"],
            "destination": result["destination"],
            "distance_km": result["distance_km"],
            "departure_date": result.get("departure_date"),
            "departure_time": result.get("departure_time"),
            "temporal_context": result.get("temporal_context"),
            "best_mode_prediction": result.get("best_mode"),
            "total_options": result["total_services"],
            "recommendations": result["recommendations"]
        }
        
    except Exception as e:
        logger.error(f"Error in recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/all-services")
def get_all_services_with_ratings():
    """
    Get all available services with GNN predicted ratings.
    
    Returns all transport services ranked by quality rating.
    """
    try:
        service = get_transport_service()
        
        if service is None:
            return {
                "success": False,
                "error": "GNN service not available"
            }
        
        result = service.get_all_services()
        
        if "error" in result:
            return {
                "success": False,
                "error": result["error"]
            }
        
        return {
            "success": True,
            "total_services": result["total_services"],
            "services": result["services"]
        }
        
    except Exception as e:
        logger.error(f"Error getting all services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/health")
def health_check():
    """Detailed health check with model and data status."""
    parser = get_nlp_parser()
    service = get_transport_service()

    model_exists = os.path.exists('model/transport_gnn_routing.pth')

    return {
        "status": "healthy",
        "nlp_parser": "loaded" if parser else "not loaded",
        "gnn_model": "loaded" if (service and service.model) else ("not found" if not model_exists else "using survey-based rules"),
        "data": "loaded" if (service and service.nodes_df is not None) else "not loaded",
        "endpoints": {
            "natural_language": "/api/query",
            "nlp_parse_only": "/api/parse",
            "structured_query": "/api/services",
            "temporal_recommendations": "/api/recommend",
            "day_info": "/api/day-info",
            "all_services": "/api/all-services",
            "health": "/api/health"
        },
        "note": "Uses temporal-aware GNN with survey-based rules for holidays/poya/crowding"
    }


@router.get("/api/day-info")
def get_day_info(date: Optional[str] = None, time: Optional[str] = None):
    """
    Get temporal information for a given date/time.
    
    Uses HolidayDetector API to determine:
    - Day type (regular, weekend, poya day, holiday)
    - Time period (early morning, morning, day, evening, night, late night)
    - Crowding likelihood (based on survey data)
    - Reliability patterns by mode
    
    Parameters:
    - date: Optional, format "2025-12-25" (defaults to today)
    - time: Optional, format "14:30" (defaults to current time)
    
    Returns temporal features and reliability metrics.
    """
    try:
        service = get_transport_service()
        
        if service is None:
            return {"error": "Service not available"}
        
        # Get temporal features
        temporal_features = service._get_temporal_features(date, time)
        
        # Add reliability scores for each mode
        modes_reliability = {}
        for mode in ["bus", "train", "ridehailing"]:
            modes_reliability[mode] = {
                "reliability": float(
                    get_reliability_score(
                        mode,
                        temporal_features["time_period"],
                        temporal_features["day_type"]
                    )
                ),
                "crowding": float(
                    get_crowding_score(
                        mode,
                        temporal_features["time_period"],
                        temporal_features["day_type"]
                    )
                )
            }
        
        return {
            "success": True,
            "date": date or "today",
            "time": time or "current",
            "temporal_features": temporal_features,
            "mode_reliability": modes_reliability,
            "note": "Based on survey data for Sri Lankan transport patterns"
        }
        
    except Exception as e:
        logger.error(f"Error getting day info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

