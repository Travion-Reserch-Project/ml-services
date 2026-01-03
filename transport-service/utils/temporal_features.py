"""
Refactored edge features for temporal-aware GNN.
Consolidates timetable and service_conditions data into a single feature set.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np


def create_temporal_edge_features(
    data_path: str = "data",
    output_path: str = "data/edges_temporal.csv"
):
    """
    Create new temporal-aware edge features by consolidating:
    - edges.csv (base route info)
    - timetables.csv (availability)
    - service_conditions.csv (reliability/crowding)
    
    Output includes aggregated features per service:
    - availability_score (0-1): % of time periods available
    - avg_reliability (0-1): from survey data
    - crowding_pattern: typical crowding by time
    - time_period_scores: reliability/crowding per time period
    
    This replaces service_conditions.csv with a learned/rule-based approach.
    """
    
    # Load data
    edges_df = pd.read_csv(f"{data_path}/edges.csv")
    timetables_df = pd.read_csv(f"{data_path}/timetables.csv")
    service_conditions_df = pd.read_csv(f"{data_path}/service_conditions.csv")
    
    # Create temporal features per service
    edge_features = []
    
    for _, edge in edges_df.iterrows():
        origin_id = edge['origin_id']
        dest_id = edge['destination_id']
        service_id = edge['service_id']
        mode = edge['mode']
        
        # Get timetables for this service
        service_timetables = timetables_df[timetables_df['service_id'] == service_id]
        
        # Get conditions for this service
        service_conditions = service_conditions_df[
            service_conditions_df['service_id'] == service_id
        ]
        
        # Aggregate features
        feature_dict = {
            'edge_id': edge['edge_id'],
            'origin_id': origin_id,
            'destination_id': dest_id,
            'service_id': service_id,
            'mode': mode,
            'distance_km': edge['distance_km'],
            'avg_duration_min': edge['avg_duration_min'],
            'is_active': edge['is_active'],
        }
        
        # Availability score: how many day types have this service available
        day_types = service_timetables['day_type'].unique()
        feature_dict['availability_score'] = len(day_types) / 3.0  # 3 day types: regular, poya, weekend
        
        # Average reliability across all day types and time periods
        if not service_conditions.empty:
            feature_dict['avg_reliability'] = service_conditions['reliability'].mean()
            
            # Crowding features by time period
            for time_period in service_conditions['time_period'].unique():
                period_data = service_conditions[
                    service_conditions['time_period'] == time_period
                ]
                crowding_score = period_data['is_crowded'].mean()
                reliability_score = period_data['reliability'].mean()
                
                feature_dict[f'{time_period}_crowding'] = crowding_score
                feature_dict[f'{time_period}_reliability'] = reliability_score
        else:
            # Default values if no conditions data
            feature_dict['avg_reliability'] = 0.75
        
        edge_features.append(feature_dict)
    
    # Create DataFrame
    temporal_edges_df = pd.DataFrame(edge_features)
    
    # Save
    temporal_edges_df.to_csv(output_path, index=False)
    print(f"✅ Created temporal edge features: {output_path}")
    print(f"   Columns: {list(temporal_edges_df.columns)}")
    
    return temporal_edges_df


def create_survey_based_reliability_rules():
    """
    Define reliability/crowding rules based on survey data.
    
    These rules can be used to generate training data without service_conditions.csv
    
    Returns:
        Dict of rules mapping (service_mode, time_period, day_type) → (reliability, crowding)
    """
    
    rules = {
        # TRAIN services
        ('train', 'early_morning', 'regular'): {'reliability': 0.85, 'crowding': 0},
        ('train', 'early_morning', 'poya'): {'reliability': 0.85, 'crowding': 1},
        ('train', 'early_morning', 'weekend'): {'reliability': 0.85, 'crowding': 0},
        
        ('train', 'morning', 'regular'): {'reliability': 0.70, 'crowding': 1},
        ('train', 'morning', 'poya'): {'reliability': 0.70, 'crowding': 1},
        ('train', 'morning', 'weekend'): {'reliability': 0.70, 'crowding': 1},
        
        ('train', 'day', 'regular'): {'reliability': 0.90, 'crowding': 0},
        ('train', 'day', 'poya'): {'reliability': 0.90, 'crowding': 1},
        ('train', 'day', 'weekend'): {'reliability': 0.90, 'crowding': 0},
        
        ('train', 'evening', 'regular'): {'reliability': 0.70, 'crowding': 1},
        ('train', 'evening', 'poya'): {'reliability': 0.70, 'crowding': 1},
        ('train', 'evening', 'weekend'): {'reliability': 0.70, 'crowding': 1},
        
        ('train', 'night', 'regular'): {'reliability': 0.65, 'crowding': 0},
        ('train', 'night', 'poya'): {'reliability': 0.65, 'crowding': 1},
        ('train', 'night', 'weekend'): {'reliability': 0.65, 'crowding': 0},
        
        # BUS services
        ('bus', 'early_morning', 'regular'): {'reliability': 0.75, 'crowding': 0},
        ('bus', 'early_morning', 'poya'): {'reliability': 0.75, 'crowding': 0},
        ('bus', 'early_morning', 'weekend'): {'reliability': 0.75, 'crowding': 0},
        
        ('bus', 'morning', 'regular'): {'reliability': 0.65, 'crowding': 1},
        ('bus', 'morning', 'poya'): {'reliability': 0.65, 'crowding': 1},
        ('bus', 'morning', 'weekend'): {'reliability': 0.65, 'crowding': 0},
        
        ('bus', 'day', 'regular'): {'reliability': 0.80, 'crowding': 0},
        ('bus', 'day', 'poya'): {'reliability': 0.80, 'crowding': 1},
        ('bus', 'day', 'weekend'): {'reliability': 0.80, 'crowding': 0},
        
        ('bus', 'evening', 'regular'): {'reliability': 0.65, 'crowding': 1},
        ('bus', 'evening', 'poya'): {'reliability': 0.65, 'crowding': 1},
        ('bus', 'evening', 'weekend'): {'reliability': 0.65, 'crowding': 0},
        
        ('bus', 'night', 'regular'): {'reliability': 0.60, 'crowding': 0},
        ('bus', 'night', 'poya'): {'reliability': 0.60, 'crowding': 0},
        ('bus', 'night', 'weekend'): {'reliability': 0.60, 'crowding': 0},
        
        # RIDEHAILING services (always higher cost, good availability)
        ('ridehailing', 'early_morning', 'regular'): {'reliability': 0.95, 'crowding': 0},
        ('ridehailing', 'early_morning', 'poya'): {'reliability': 0.95, 'crowding': 0},
        ('ridehailing', 'early_morning', 'weekend'): {'reliability': 0.95, 'crowding': 0},
        
        ('ridehailing', 'morning', 'regular'): {'reliability': 0.90, 'crowding': 0},
        ('ridehailing', 'morning', 'poya'): {'reliability': 0.90, 'crowding': 0},
        ('ridehailing', 'morning', 'weekend'): {'reliability': 0.90, 'crowding': 0},
        
        ('ridehailing', 'day', 'regular'): {'reliability': 0.95, 'crowding': 0},
        ('ridehailing', 'day', 'poya'): {'reliability': 0.95, 'crowding': 0},
        ('ridehailing', 'day', 'weekend'): {'reliability': 0.95, 'crowding': 0},
        
        ('ridehailing', 'evening', 'regular'): {'reliability': 0.90, 'crowding': 0},
        ('ridehailing', 'evening', 'poya'): {'reliability': 0.90, 'crowding': 0},
        ('ridehailing', 'evening', 'weekend'): {'reliability': 0.90, 'crowding': 0},
        
        ('ridehailing', 'night', 'regular'): {'reliability': 0.85, 'crowding': 0},
        ('ridehailing', 'night', 'poya'): {'reliability': 0.85, 'crowding': 0},
        ('ridehailing', 'night', 'weekend'): {'reliability': 0.85, 'crowding': 0},
    }
    
    return rules


def get_reliability_score(service_mode: str, time_period: str, day_type: str) -> float:
    """
    Get reliability score for a service at a specific time/day.
    
    Args:
        service_mode: 'train', 'bus', 'ridehailing'
        time_period: 'early_morning', 'morning', 'day', 'evening', 'night'
        day_type: 'regular', 'weekend', 'poya', 'holiday'
        
    Returns:
        Reliability score 0-1 (higher = more reliable)
    """
    # Map holiday to poya for lookup
    if day_type == 'holiday':
        day_type = 'poya'
    
    rules = create_survey_based_reliability_rules()
    key = (service_mode, time_period, day_type)
    
    if key in rules:
        return rules[key]['reliability']
    
    # Default fallback
    return 0.75


def get_crowding_score(service_mode: str, time_period: str, day_type: str) -> float:
    """
    Get crowding likelihood (0=empty, 1=crowded) for a service.
    
    Args:
        service_mode: 'train', 'bus', 'ridehailing'
        time_period: 'early_morning', 'morning', 'day', 'evening', 'night'
        day_type: 'regular', 'weekend', 'poya', 'holiday'
        
    Returns:
        Crowding score 0-1 (0=empty, 1=crowded)
    """
    # Map holiday to poya for lookup
    if day_type == 'holiday':
        day_type = 'poya'
    
    rules = create_survey_based_reliability_rules()
    key = (service_mode, time_period, day_type)
    
    if key in rules:
        return rules[key]['crowding']
    
    # Default fallback
    return 0.5


if __name__ == "__main__":
    # Generate temporal edge features
    create_temporal_edge_features()
    
    # Show rules
    print("\n📋 Survey-based Reliability Rules:")
    rules = create_survey_based_reliability_rules()
    for key, value in sorted(rules.items()):
        print(f"  {key} → {value}")
