"""
Text Processor

Handles text preprocessing for multilingual support (Sinhala, English, Tamil).
Includes language detection, normalization, and query preprocessing.
"""

import logging
import re
from typing import Optional, Literal
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Text processing utilities for multilingual RAG
    
    Features:
    - Language detection
    - Text normalization
    - Query preprocessing
    - Sinhala/English/Tamil support
    """
    
    # Unicode ranges for language detection
    SINHALA_RANGE = (0x0D80, 0x0DFF)
    TAMIL_RANGE = (0x0B80, 0x0BFF)
    
    def __init__(self):
        logger.info("✓ Text processor initialized")
    
    def detect_language(self, text: str) -> Literal["en", "si", "ta", "mixed"]:
        """
        Detect language of text
        
        Args:
            text: Input text
            
        Returns:
            Language code: "en" (English), "si" (Sinhala), "ta" (Tamil), "mixed"
        """
        try:
            # Check for Sinhala characters
            sinhala_chars = sum(
                1 for char in text 
                if self.SINHALA_RANGE[0] <= ord(char) <= self.SINHALA_RANGE[1]
            )
            
            # Check for Tamil characters
            tamil_chars = sum(
                1 for char in text 
                if self.TAMIL_RANGE[0] <= ord(char) <= self.TAMIL_RANGE[1]
            )
            
            total_chars = len(text.replace(" ", ""))
            
            if total_chars == 0:
                return "en"
            
            # If >60% Sinhala characters
            if sinhala_chars / total_chars > 0.6:
                return "si"
            
            # If >60% Tamil characters
            if tamil_chars / total_chars > 0.6:
                return "ta"
            
            # If mixed (both Indic and Latin)
            if sinhala_chars > 0 or tamil_chars > 0:
                return "mixed"
            
            # Use langdetect for Latin script
            try:
                lang = detect(text)
                if lang in ["en", "si", "ta"]:
                    return lang
                return "en"  # Default to English
            except LangDetectException:
                return "en"
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to 'en'")
            return "en"
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Normalize common punctuation
        text = text.replace('​', '')  # Remove zero-width spaces
        text = text.replace('\u200b', '')  # Remove zero-width spaces
        
        return text
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query for search
        
        Args:
            query: Raw query
            
        Returns:
            Preprocessed query
        """
        # Normalize
        query = self.normalize_text(query)
        
        # Handle common abbreviations
        query = self._expand_abbreviations(query)
        
        return query
    
    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand common abbreviations in queries
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded abbreviations
        """
        # English abbreviations
        abbreviations = {
            r'\bCMB\b': 'Colombo',
            r'\bKDY\b': 'Kandy',
            r'\bGLE\b': 'Galle',
            r'\bJFN\b': 'Jaffna',
            r'\bMTR\b': 'Matara',
            r'\bAC\b': 'air conditioned',
            r'\bNon-AC\b': 'normal',
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(abbr, full, text, flags=re.IGNORECASE)
        
        return text
    
    def extract_locations(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract origin and destination from query
        
        Args:
            text: Query text
            
        Returns:
            Tuple of (origin, destination) or (None, None)
        """
        # Common patterns for route queries
        patterns = [
            r'from\s+(.+?)\s+to\s+(.+?)(?:\s+bus|\s+fare|\?|$)',
            r'(.+?)\s+to\s+(.+?)(?:\s+bus|\s+fare|\?|$)',
            r'(.+?)\s+ගාස්තුව\s+(.+?)(?:\?|$)',  # Sinhala
            r'(.+?)\s+සිට\s+(.+?)(?:\s+|$)',  # Sinhala
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                origin = match.group(1).strip()
                destination = match.group(2).strip()
                
                # Clean up
                origin = self._clean_location_name(origin)
                destination = self._clean_location_name(destination)
                
                if origin and destination:
                    return origin, destination
        
        return None, None
    
    def _clean_location_name(self, location: str) -> str:
        """Clean location name"""
        # Remove common words
        location = re.sub(r'\b(bus|fare|price|cost)\b', '', location, flags=re.IGNORECASE)
        location = location.strip()
        return location
    
    def format_fare_response(
        self,
        origin: str,
        destination: str,
        fares: dict,
        language: str = "en"
    ) -> str:
        """
        Format fare information as natural text
        
        Args:
            origin: Origin location
            destination: Destination location
            fares: Dictionary of fare types and amounts
            language: Target language
            
        Returns:
            Formatted fare text
        """
        if language == "si":
            # Sinhala format
            text = f"{origin} සිට {destination} දක්වා බස් ගාස්තු:\n"
            
            fare_labels = {
                "normal": "සාමාන්‍ය සේවාව",
                "semi_luxury": "අර්ධ සුඛෝපභෝගී",
                "luxury": "සුඛෝපභෝගී",
                "super_luxury": "ඉහළම සුඛෝපභෝගී"
            }
            
            for fare_type, amount in fares.items():
                if amount:
                    label = fare_labels.get(fare_type, fare_type)
                    text += f"  • {label}: රු. {amount:.2f}\n"
        
        else:
            # English format
            text = f"Bus fares from {origin} to {destination}:\n"
            
            fare_labels = {
                "normal": "Normal Service",
                "semi_luxury": "Semi-Luxury",
                "luxury": "Luxury",
                "super_luxury": "Super Luxury"
            }
            
            for fare_type, amount in fares.items():
                if amount:
                    label = fare_labels.get(fare_type, fare_type.replace('_', ' ').title())
                    text += f"  • {label}: LKR {amount:.2f}\n"
        
        return text.strip()
    
    def is_fare_query(self, query: str) -> bool:
        """
        Check if query is asking about fares
        
        Args:
            query: User query
            
        Returns:
            True if fare-related query
        """
        fare_keywords = [
            'fare', 'price', 'cost', 'charge', 'fee',  # English
            'ගාස්තුව', 'මිල', 'ගාස්තු',  # Sinhala
            'கட்டணம்', 'விலை'  # Tamil
        ]
        
        query_lower = query.lower()
        
        return any(keyword in query_lower for keyword in fare_keywords)
    
    def is_route_query(self, query: str) -> bool:
        """
        Check if query is asking about routes
        
        Args:
            query: User query
            
        Returns:
            True if route-related query
        """
        route_keywords = [
            'route', 'bus number', 'which bus', 'how to get',  # English
            'මාර්ගය', 'බස් අංකය', 'යන්නේ කොහොමද',  # Sinhala
        ]
        
        query_lower = query.lower()
        
        return any(keyword in query_lower for keyword in route_keywords)
    
    def truncate_text(self, text: str, max_length: int = 500) -> str:
        """
        Truncate text to maximum length
        
        Args:
            text: Input text
            max_length: Maximum characters
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + "..."


# ================================================
# Singleton Instance
# ================================================

_text_processor = None


def get_text_processor() -> TextProcessor:
    """Get or create singleton instance"""
    global _text_processor
    
    if _text_processor is None:
        _text_processor = TextProcessor()
    
    return _text_processor


# ================================================
# Testing
# ================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = TextProcessor()
    
    # Test language detection
    test_texts = [
        "What is the bus fare from Colombo to Galle?",
        "කොළඹ සිට ගාල්ලේ බස් ගාස්තුව කීයද?",
        "Colombo to Galle කොහොමද යන්නේ?",
    ]
    
    print("\n=== Language Detection ===")
    for text in test_texts:
        lang = processor.detect_language(text)
        print(f"{text[:40]}... → {lang}")
    
    # Test location extraction
    print("\n=== Location Extraction ===")
    queries = [
        "What is the fare from Colombo to Kandy?",
        "Colombo to Galle bus fare?",
        "කොළඹ සිට මාතර ගාස්තුව?",
    ]
    
    for query in queries:
        origin, destination = processor.extract_locations(query)
        print(f"{query}")
        print(f"  → Origin: {origin}, Destination: {destination}\n")
    
    # Test fare formatting
    print("\n=== Fare Formatting ===")
    fares = {
        "normal": 250.0,
        "semi_luxury": 350.0,
        "luxury": 450.0
    }
    
    formatted_en = processor.format_fare_response("Colombo", "Galle", fares, "en")
    formatted_si = processor.format_fare_response("ක ොළඹ", "ගාල්ල", fares, "si")
    
    print("English:")
    print(formatted_en)
    print("\nSinhala:")
    print(formatted_si)
