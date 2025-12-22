"""
NLP Query Parser using BERT for Tourist Transport Queries
Uses: google-bert/bert-base-uncased from Hugging Face
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import torch
from transformers import BertTokenizer, BertForTokenClassification
import spacy

class TransportQueryParser:
    def __init__(self, use_bert: bool = True):
        """
        Initialize the parser with BERT or fallback to spaCy.
        
        Args:
            use_bert: If True, use BERT model. If False, use spaCy (faster).
        """
        self.use_bert = use_bert
        
        # Common Sri Lankan cities (used for hints, but NOT limiting)
        # Parser will extract ANY location name, not just these
        self.known_cities = [
            "Colombo", "Kandy", "Galle", "Jaffna", "Negombo",
            "Trincomalee", "Batticaloa", "Anuradhapura", "Polonnaruwa",
            "Badulla", "Matara", "Ella", "Nuwara Eliya", "Sigiriya", "Dambulla",
            "Ratnapura", "Ampara", "Kurunegala", "Puttalam", "Vavuniya",
            "Mannar", "Kilinochchi", "Mullaitivu", "Hambantota", "Monaragala",
            "Kegalle", "Kalutara", "Gampaha", "Nuwara", "Eliya", "Bentota",
            "Hikkaduwa", "Unawatuna", "Mirissa", "Tangalle", "Arugam Bay",
            "Pasikuda", "Nilaveli", "Kalpitiya", "Habarana", "Mihintale"
        ]
        
        # Common transport mode keywords (hints, but will extract others too)
        self.mode_keywords = {
            'bus': ['bus', 'buses', 'coach'],
            'train': ['train', 'railway', 'rail', 'intercity'],
            'tuk-tuk': ['tuk-tuk', 'tuktuk', 'tuk tuk', 'three-wheeler', 'trishaw'],
            'taxi': ['taxi', 'cab', 'car'],
            'van': ['van', 'minivan', 'minibus'],
            'ferry': ['ferry', 'boat', 'ship']
        }
        
        if use_bert:
            self._init_bert()
        else:
            self._init_spacy()
    
    def _init_bert(self):
        """Initialize BERT model for token classification."""
        try:
            print("🤖 Loading BERT model (google-bert/bert-base-uncased)...")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # For now, use pre-trained BERT. Later can fine-tune for NER
            self.model = None  # Will implement custom NER head if needed
            print("✅ BERT model loaded")
            
            # Always initialize spaCy as well for entity extraction
            # BERT will be used for embeddings/classification later
            print("📥 Also loading spaCy for entity extraction...")
            self._init_spacy()
            
        except Exception as e:
            print(f"⚠️ Could not load BERT: {e}")
            print("   Falling back to spaCy only...")
            self.use_bert = False
            self._init_spacy()
    
    def _init_spacy(self):
        """Initialize spaCy for NER."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("⚠️ spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom entity ruler for known Sri Lankan locations
        # This helps with recognition but doesn't limit to only these
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = [{"label": "GPE", "pattern": city} for city in self.known_cities]
            ruler.add_patterns(patterns)
    
    def parse(self, query: str) -> Dict:
        """
        Parse a natural language query into structured parameters.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with extracted parameters:
            {
                'origin': str or None,
                'destination': str or None,
                'departure_time': float or None (hours, e.g., 14.5 for 2:30pm),
                'mode': str or None ('bus', 'train', 'tuk-tuk'),
                'date': str or None ('today', 'tomorrow', or date string),
                'raw_query': str
            }
        """
        query_lower = query.lower()
        
        result = {
            'origin': None,
            'destination': None,
            'departure_time': None,
            'mode': None,
            'date': 'today',
            'raw_query': query
        }
        
        # Extract locations
        locations = self._extract_locations(query)
        if len(locations) >= 2:
            result['origin'] = locations[0]
            result['destination'] = locations[1]
        elif len(locations) == 1:
            # If only one location, assume it's destination (origin is current location)
            result['destination'] = locations[0]
        
        # Extract time
        result['departure_time'] = self._extract_time(query_lower)
        
        # Extract transport mode
        result['mode'] = self._extract_mode(query_lower)
        
        # Extract date
        result['date'] = self._extract_date(query_lower)
        
        return result
    
    def _extract_locations(self, query: str) -> List[str]:
        """
        Extract ANY location names from query using NER.
        Not limited to predefined cities - extracts any proper nouns that could be places.
        """
        locations = []
        
        # Clean query by removing filler words at boundaries
        filler_words = ['please', 'thanks', 'thank you', 'kindly', 'urgent', 'asap']
        cleaned_query = query
        for filler in filler_words:
            cleaned_query = re.sub(rf'\b{filler}\b', '', cleaned_query, flags=re.IGNORECASE)
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()  # Remove extra spaces
        
        # PATTERN 1: "to DESTINATION from ORIGIN" (destination mentioned first)
        to_from_pattern = r'to\s+(\w+(?:\s+\w+)?)\s+from\s+(\w+(?:\s+\w+)?)'
        to_from_match = re.search(to_from_pattern, cleaned_query, re.IGNORECASE)
        if to_from_match:
            destination = to_from_match.group(1).strip().capitalize()
            origin = to_from_match.group(2).strip().capitalize()
            # Filter out common verbs/words
            skip_words = ['go', 'get', 'going', 'getting', 'travel', 'trip', 'way', 'how', 'want', 'need', 'like', 'prefer', 'nedd', 'ned']
            if origin.lower() not in skip_words and destination.lower() not in skip_words:
                return [origin, destination]
        
        # PATTERN 2: "from ORIGIN to DESTINATION" (origin mentioned first)
        from_to_pattern = r'from\s+(\w+(?:\s+\w+)?)\s+to\s+(\w+(?:\s+\w+)?)'
        from_to_match = re.search(from_to_pattern, cleaned_query, re.IGNORECASE)
        if from_to_match:
            origin = from_to_match.group(1).strip().capitalize()
            destination = from_to_match.group(2).strip().capitalize()
            skip_words = ['go', 'get', 'going', 'getting', 'travel', 'trip', 'way', 'how', 'want', 'need', 'like', 'prefer']
            if origin.lower() not in skip_words and destination.lower() not in skip_words:
                return [origin, destination]
        
        # PATTERN 3: Simple "ORIGIN to DESTINATION" pattern (case-insensitive)
        simple_to_pattern = r'\b(\w+(?:\s+\w+)?)\s+to\s+(\w+(?:\s+\w+)?)\b'
        simple_match = re.search(simple_to_pattern, cleaned_query, re.IGNORECASE)
        if simple_match:
            origin = simple_match.group(1).strip().capitalize()
            destination = simple_match.group(2).strip().capitalize()
            # Filter out common words
            skip_words = ['go', 'get', 'going', 'getting', 'travel', 'trip', 'way', 'how', 'want', 'need', 'like', 'prefer', 'nedd']
            if origin.lower() not in skip_words and destination.lower() not in skip_words:
                return [origin, destination]
        
        # First check for common multi-word location patterns (X National Park, X Beach, X Fort, etc.)
        # This must come before "to/from" patterns to capture full names
        multiword_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(National Park|Beach|Fort|Station|Airport|Plains|Bay)\b',
        ]
        
        for pattern in multiword_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                # match is a tuple: (place_name, suffix)
                full_location = f"{match[0]} {match[1]}"
                locations.append(full_location)
        
        if locations:
            # If we found multi-word locations, also check for other simple locations
            # Try "to X from Y" or "from Y to X" patterns
            to_from_pattern = r'(?:to|going to|heading to)\s+(?:[A-Z][\w\s]+?\s+(?:National Park|Beach|Fort|Plains))\s+(?:from|leaving from)\s+([A-Z][\w\s]+?)(?:\s+by|\s+at|\s+tomorrow|\s+today|$)'
            to_from_match = re.search(to_from_pattern, query, re.IGNORECASE)
            
            if to_from_match:
                origin = to_from_match.group(1).strip()
                if origin:
                    locations.append(origin)
            
            from_to_pattern = r'(?:from|leaving)\s+([A-Z][\w\s]+?)\s+(?:to|heading to|going to)\s+(?:[A-Z][\w\s]+?\s+(?:National Park|Beach|Fort|Plains))'
            from_to_match = re.search(from_to_pattern, query, re.IGNORECASE)
            
            if from_to_match:
                origin = from_to_match.group(1).strip()
                if origin:
                    locations.insert(0, origin)  # Origin should come first
            
            return locations
        
        # Try "to X from Y" pattern for better accuracy
        to_from_pattern = r'(?:to|going to|heading to)\s+([A-Z][\w\s]+?)\s+(?:from|leaving from)\s+([A-Z][\w\s]+?)(?:\s+by|\s+at|\s+tomorrow|\s+today|$)'
        to_from_match = re.search(to_from_pattern, query, re.IGNORECASE)
        
        if to_from_match:
            destination = to_from_match.group(1).strip()
            origin = to_from_match.group(2).strip()
            
            # Clean up - remove transport mode words and "ride"
            transport_words = ['bus', 'train', 'taxi', 'van', 'car', 'tuk-tuk', 'motorcycle', 'scooter', 'truck', 'vehicle', 'goods', 'private', 'ride']
            for word in transport_words:
                origin = re.sub(rf'\b{word}\b', '', origin, flags=re.IGNORECASE).strip()
                destination = re.sub(rf'\b{word}\b', '', destination, flags=re.IGNORECASE).strip()
            
            if origin:
                locations.append(origin)
            if destination:
                locations.append(destination)
            
            return locations
        
        # Then try "from X to Y" pattern
        from_to_pattern = r'(?:from|leaving)\s+([A-Z][\w\s]+?)\s+(?:to|heading to|going to)\s+([A-Z][\w\s]+?)(?:\s+by|\s+at|\s+tomorrow|\s+today|$)'
        from_to_match = re.search(from_to_pattern, query, re.IGNORECASE)
        
        if from_to_match:
            origin = from_to_match.group(1).strip()
            destination = from_to_match.group(2).strip()
            
            # Clean up - remove transport mode words and "ride"
            transport_words = ['bus', 'train', 'taxi', 'van', 'car', 'tuk-tuk', 'motorcycle', 'scooter', 'truck', 'vehicle', 'goods', 'private', 'ride']
            for word in transport_words:
                origin = re.sub(rf'\b{word}\b', '', origin, flags=re.IGNORECASE).strip()
                destination = re.sub(rf'\b{word}\b', '', destination, flags=re.IGNORECASE).strip()
            
            if origin:
                locations.append(origin)
            if destination:
                locations.append(destination)
            
            return locations
        
        # Use spaCy NER to extract all location entities
        doc = self.nlp(query)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:  # Geographic, Location, Facility
                # Filter out transport modes that spaCy might tag as locations
                if ent.text.lower() not in ['bus', 'train', 'taxi', 'van', 'car', 'motorcycle', 'scooter', 'truck', 'ride']:
                    location = ent.text.strip()
                    # Capitalize properly (Sri Lankan convention)
                    location = ' '.join(word.capitalize() for word in location.split())
                    locations.append(location)
        
        # Also check for capitalized words that might be places (proper nouns)
        # This catches places that spaCy might miss
        if not locations:
            # Look for capitalized words (likely place names)
            words = query.split()
            for i, word in enumerate(words):
                # Check if word starts with capital letter and isn't a common word
                if word and word[0].isupper():
                    # Skip common English words that are capitalized and transport modes
                    skip_words = ['I', 'My', 'The', 'A', 'An', 'To', 'From', 'At', 'On', 'In', 'Bus', 'Train', 'Taxi', 'Van', 'Car', 'Goods', 'Motorcycle', 'Ride']
                    if word not in skip_words:
                        # Check if it might be a place name
                        cleaned = word.strip('.,!?;:')
                        if len(cleaned) > 2:  # At least 3 characters
                            locations.append(cleaned)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_locations = []
        for loc in locations:
            loc_lower = loc.lower()
            if loc_lower not in seen:
                seen.add(loc_lower)
                unique_locations.append(loc)
        
        return unique_locations
    
    def _extract_time(self, query: str) -> Optional[float]:
        """
        Extract departure time from query.
        Returns time in hours (e.g., 14.5 for 2:30pm).
        """
        # Pattern: "2pm", "2:30pm", "14:00", "2 pm"
        time_pattern = r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?\b'
        matches = re.findall(time_pattern, query)
        
        if matches:
            hour, minute, period = matches[0]
            hour = int(hour)
            minute = int(minute) if minute else 0
            
            # Convert to 24-hour format
            if period and period.lower() == 'pm' and hour < 12:
                hour += 12
            elif period and period.lower() == 'am' and hour == 12:
                hour = 0
            
            return hour + minute / 60.0
        
        # Pattern: "morning", "afternoon", "evening", "night"
        time_keywords = {
            'morning': 8.0,
            'afternoon': 14.0,
            'evening': 18.0,
            'night': 20.0
        }
        
        for keyword, time_value in time_keywords.items():
            if keyword in query:
                return time_value
        
        # Pattern: "after 2", "before 5"
        after_pattern = r'after\s+(\d{1,2})'
        before_pattern = r'before\s+(\d{1,2})'
        
        after_match = re.search(after_pattern, query)
        if after_match:
            return float(after_match.group(1))
        
        before_match = re.search(before_pattern, query)
        if before_match:
            return float(before_match.group(1))
        
        return None
    
    def _extract_mode(self, query: str) -> Optional[str]:
        """
        Extract ANY transport mode from query.
        Supports common modes and any vehicle type mentioned.
        """
        query_lower = query.lower()
        
        # First check for "X ride" pattern (e.g., "motorcycle ride", "bike ride")
        # This prevents the vehicle name from being extracted as a location
        ride_pattern = r'\b(motorcycle|motorbike|bike|bus|train|car|tuk-tuk|tuktuk|scooter|van|truck)\s+ride\b'
        ride_match = re.search(ride_pattern, query_lower)
        if ride_match:
            vehicle = ride_match.group(1)
            if vehicle in ['motorcycle', 'motorbike', 'bike']:
                return 'motorcycle'
            elif vehicle in ['tuk-tuk', 'tuktuk']:
                return 'tuk-tuk'
            else:
                return vehicle
        
        # Check for "private car" specifically
        if re.search(r'\bprivate\s+car\b', query_lower):
            return 'car'
        
        # Check known mode keywords
        for mode, keywords in self.mode_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return mode
        
        # Look for any vehicle/transport-related words
        transport_patterns = [
            r'\b(vehicle|transport|ride|trip)\b',
            r'\b(motorcycle|motorbike|bike|scooter)\b',
            r'\b(truck|lorry|goods vehicle)\b',
            r'\b(helicopter|plane|flight|air)\b',
            r'\b(uber|grab|pickme|ride-share|rideshare)\b'
        ]
        
        for pattern in transport_patterns:
            match = re.search(pattern, query_lower)
            if match:
                vehicle_type = match.group(1)
                # Return the matched vehicle type
                return vehicle_type
        
        return None
    
    def _extract_date(self, query: str) -> str:
        """Extract date from query."""
        if 'tomorrow' in query:
            return 'tomorrow'
        elif 'today' in query:
            return 'today'
        
        # Pattern: "next Monday", "this Friday"
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day in query:
                return f'next_{day}'
        
        # Pattern: "Dec 25", "December 25th", "25/12"
        date_pattern = r'\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b'
        date_match = re.search(date_pattern, query)
        if date_match:
            day, month, year = date_match.groups()
            year = year if year else str(datetime.now().year)
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        return 'today'
    
    def validate_query(self, parsed: Dict) -> tuple[bool, str]:
        """
        Validate if parsed query has minimum required information.
        Now accepts ANY location names, not just predefined cities.
        
        Returns:
            (is_valid, error_message)
        """
        if not parsed['origin'] and not parsed['destination']:
            return False, "Please specify at least a destination location."
        
        if parsed['origin'] and parsed['destination'] and parsed['origin'].lower() == parsed['destination'].lower():
            return False, "Origin and destination cannot be the same."
        
        # No longer restricting to known cities - accept any location
        # The backend will handle validation against available routes
        
        return True, ""


# Example usage
if __name__ == "__main__":
    parser = TransportQueryParser(use_bert=True)
    
    # Test queries
    test_queries = [
        "I want to go from Kandy to Colombo at 2pm",
        "Bus from Galle to Colombo tomorrow morning",
        "Train tickets from Colombo to Ella leaving after 3pm",
        "How do I get to Kandy from Colombo?",
        "Show me buses to Galle this evening",
    ]
    
    print("\n🧪 Testing Query Parser:\n")
    for query in test_queries:
        print(f"Query: {query}")
        result = parser.parse(query)
        print(f"Parsed: {result}")
        is_valid, error = parser.validate_query(result)
        print(f"Valid: {is_valid} {f'({error})' if error else ''}")
        print("-" * 80)
