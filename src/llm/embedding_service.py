import requests
import json
import time
import warnings
import urllib3
from typing import List, Dict, Any, Union
from ..config import config, logger

class EmbeddingService:
    """Service for generating and working with text embeddings"""
    
    def __init__(self, api_key: str, endpoint: str = "https://va-east-apim.devtest.spd.vaec.va.gov/openai/deployments/text-embedding-3-large/embeddings", 
                api_version: str = "2025-01-01-preview", verify_ssl: bool = False):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.verify_ssl = verify_ssl
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        # Add query params if not in endpoint
        if "?" not in endpoint:
            self.endpoint = f"{endpoint}?api-version={api_version}"
        
        # Handle SSL verification
        if not verify_ssl:
            # Disable SSL warnings when verification is turned off
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled for embedding service")
        
        logger.debug(f"Initialized EmbeddingService with endpoint: {self.endpoint}")
    
    def get_embedding(self, text: str, retry_count: int = 3, retry_delay: float = 1.0) -> List[float]:
        """Get embedding vector for a text string"""
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
        
        payload = {
            "input": text,
            "dimensions": 1536,  # Can be adjusted based on use case
            "encoding_format": "float"
        }
        
        # Implement retry logic
        for attempt in range(retry_count):
            try:
                # Pass verify=False to bypass SSL verification if configured
                response = requests.post(
                    self.endpoint, 
                    headers=self.headers, 
                    json=payload,
                    verify=self.verify_ssl
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result["data"][0]["embedding"]
                logger.debug(f"Successfully retrieved embedding of dimension {len(embedding)}")
                return embedding
                
            except Exception as e:
                logger.error(f"Embedding API error (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("All embedding API attempts failed")
                    return []
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Get embeddings for multiple texts in efficient batches"""
        embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Small delay between batches if not the last batch
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        return embeddings
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not a or not b:
            return 0.0
        
        try:
            import numpy as np
            
            a_array = np.array(a)
            b_array = np.array(b)
            
            dot_product = np.dot(a_array, b_array)
            norm_a = np.linalg.norm(a_array)
            norm_b = np.linalg.norm(b_array)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return float(dot_product / (norm_a * norm_b))  # Convert from numpy type to float
        except ImportError:
            logger.warning("NumPy not available for vector operations, using fallback implementation")
            # Fallback implementation without numpy
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
