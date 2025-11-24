"""
Transport Service Integration
Combines NLP Parser with GNN Model for end-to-end query processing
"""

import torch
from typing import Dict, List
import pandas as pd
from utils.nlp_parser import TransportQueryParser


class TransportService:
    """
    Main service class that integrates NLP parsing and GNN predictions.
    """
    
    def __init__(self, model_path: str, data_path: str = None):
        """
        Initialize the transport service.
        
        Args:
            model_path: Path to trained GNN model (.pth file)
            data_path: Path to transport data CSVs
        """
        self.parser = TransportQueryParser(use_bert=True)
        self.model = None
        self.graph_data = None
        self.node2idx = None
        self.services_data = None
        
        # Load model
        self._load_model(model_path)
        
        # Load data if provided
        if data_path:
            self._load_data(data_path)
    
    def _load_model(self, model_path: str):
        """Load the trained GNN model."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract model configuration
            node_features = checkpoint['node_features']
            edge_features = checkpoint['edge_features']
            hidden_dim = checkpoint['hidden_dim']
            
            # Reconstruct model architecture
            from model.gnn_model import ServiceAvailabilityGNN  # You'll create this
            self.model = ServiceAvailabilityGNN(
                node_in=node_features,
                hidden=hidden_dim,
                edge_in=edge_features
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load node mapping
            self.node2idx = checkpoint['node2idx']
            
            print(f"✅ Model loaded successfully")
            print(f"   Test MAE: {checkpoint['test_mae']:.4f}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self.model = None
    
    def _load_data(self, data_path: str):
        """Load transport network data."""
        try:
            # Load services and routes
            services_df = pd.read_csv(f"{data_path}/services.csv")
            routes_df = pd.read_csv(f"{data_path}/routes.csv")
            
            # Merge for full service information
            self.services_data = services_df.merge(routes_df, on='route_id', how='left')
            
            print(f"✅ Data loaded: {len(self.services_data)} services")
        except Exception as e:
            print(f"⚠️ Could not load data: {e}")
            self.services_data = None
    
    def process_query(self, query: str) -> Dict:
        """
        Process a natural language query and return transport options.
        
        Args:
            query: Natural language query (e.g., "Bus from Kandy to Colombo at 2pm")
            
        Returns:
            Dictionary with:
            - parsed_query: Extracted parameters
            - services: List of available services with predictions
            - error: Error message if any
        """
        # Parse the query
        parsed = self.parser.parse(query)
        
        # Validate
        is_valid, error = self.parser.validate_query(parsed)
        if not is_valid:
            return {
                'success': False,
                'error': error,
                'parsed_query': parsed
            }
        
        # Filter services based on parsed query
        services = self._filter_services(parsed)
        
        # Get predictions if model is loaded
        if self.model and services:
            services = self._add_predictions(services, parsed)
        
        return {
            'success': True,
            'parsed_query': parsed,
            'services': services,
            'count': len(services)
        }
    
    def _filter_services(self, parsed: Dict) -> List[Dict]:
        """Filter services based on parsed query parameters."""
        if self.services_data is None:
            return []
        
        # Start with all services
        filtered = self.services_data.copy()
        
        # Filter by origin
        if parsed['origin']:
            filtered = filtered[filtered['origin'] == parsed['origin']]
        
        # Filter by destination
        if parsed['destination']:
            filtered = filtered[filtered['destination'] == parsed['destination']]
        
        # Filter by mode
        if parsed['mode']:
            filtered = filtered[filtered['mode'] == parsed['mode']]
        
        # Filter by departure time
        if parsed['departure_time'] is not None:
            # Convert departure_time to hours
            filtered = filtered[
                filtered['departure_time'].apply(self._time_to_hours) >= parsed['departure_time']
            ]
        
        # Convert to list of dicts
        services = filtered.to_dict('records')
        
        # Sort by departure time
        services.sort(key=lambda x: self._time_to_hours(x['departure_time']))
        
        return services[:10]  # Return top 10 services
    
    def _add_predictions(self, services: List[Dict], parsed: Dict) -> List[Dict]:
        """Add GNN predictions to services."""
        # This will use the loaded GNN model to predict availability
        # Implementation depends on your GNN model structure
        
        for service in services:
            # Placeholder: Add predicted availability
            service['predicted_availability'] = 0.85  # Mock value
            service['confidence'] = 0.92  # Mock value
        
        return services
    
    @staticmethod
    def _time_to_hours(time_str: str) -> float:
        """Convert HH:MM format to decimal hours."""
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours + minutes / 60.0
        except:
            return 0.0
    
    def get_service_details(self, service_id: str) -> Dict:
        """Get detailed information about a specific service."""
        if self.services_data is None:
            return {'error': 'Data not loaded'}
        
        service = self.services_data[
            self.services_data['service_id'] == service_id
        ]
        
        if service.empty:
            return {'error': f'Service {service_id} not found'}
        
        return service.iloc[0].to_dict()


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = TransportService(
        model_path='model/transport_gnn_model.pth',
        data_path='data'
    )
    
    # Test queries
    test_queries = [
        "I need a bus from Kandy to Colombo at 2pm",
        "Train from Colombo to Ella tomorrow morning",
        "How do I get to Galle from Colombo after 3pm?",
    ]
    
    print("\n🧪 Testing Transport Service:\n")
    for query in test_queries:
        print(f"Query: {query}")
        result = service.process_query(query)
        
        if result['success']:
            print(f"✅ Found {result['count']} services")
            for i, svc in enumerate(result['services'][:3], 1):
                print(f"   {i}. {svc.get('service_name', 'N/A')} - "
                      f"{svc.get('departure_time', 'N/A')} "
                      f"({svc.get('mode', 'N/A')})")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 80)
