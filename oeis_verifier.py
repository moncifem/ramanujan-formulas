import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class OEISResult:
    """Result from OEIS search"""
    sequence_id: str
    name: str
    description: str
    formula: Optional[str] = None
    url: str = ""
    confidence: float = 0.0

class OEISVerifier:
    """Verifies mathematical discoveries against OEIS database"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.cache = {}  # Simple cache for repeated queries
        
    def search_oeis_by_value(self, value: float, tolerance: float = 1e-10) -> List[OEISResult]:
        """Search OEIS for sequences containing a specific value"""
        try:
            # Convert value to decimal string with appropriate precision
            value_str = f"{value:.15f}".rstrip('0').rstrip('.')
            
            # Check cache first
            cache_key = f"value_{value_str}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Search OEIS
            search_url = f"https://oeis.org/search?q={value_str}&language=english&go=Search"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Parse search results
            for result_div in soup.find_all('div', class_='sequence'):
                try:
                    # Extract sequence ID
                    id_link = result_div.find('a', href=re.compile(r'/A\d+'))
                    if not id_link:
                        continue
                    
                    sequence_id = id_link.text.strip()
                    
                    # Extract name and description
                    name_elem = result_div.find('td', class_='rh')
                    name = name_elem.text.strip() if name_elem else "Unknown"
                    
                    # Extract description
                    desc_elem = result_div.find('td', class_='lh')
                    description = desc_elem.text.strip() if desc_elem else ""
                    
                    # Calculate confidence based on value match
                    confidence = self._calculate_confidence(description, value_str)
                    
                    result = OEISResult(
                        sequence_id=sequence_id,
                        name=name,
                        description=description,
                        url=f"https://oeis.org/{sequence_id}",
                        confidence=confidence
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error parsing OEIS result: {e}")
                    continue
            
            # Cache results
            self.cache[cache_key] = results
            
            # Sort by confidence
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            return results[:10]  # Return top 10 results
            
        except Exception as e:
            logger.error(f"OEIS search error: {e}")
            return []
    
    def search_oeis_by_expression(self, expression: str) -> List[OEISResult]:
        """Search OEIS for formulas similar to the given expression"""
        try:
            # Clean expression for search
            clean_expr = self._clean_expression_for_search(expression)
            
            # Check cache
            cache_key = f"expr_{clean_expr}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Search for key terms from the expression
            search_terms = self._extract_search_terms(expression)
            
            results = []
            for term in search_terms[:3]:  # Limit to avoid too many requests
                term_results = self._search_oeis_term(term)
                results.extend(term_results)
                time.sleep(0.5)  # Rate limiting
            
            # Remove duplicates and sort
            unique_results = {}
            for result in results:
                if result.sequence_id not in unique_results:
                    unique_results[result.sequence_id] = result
            
            final_results = list(unique_results.values())
            final_results.sort(key=lambda x: x.confidence, reverse=True)
            
            # Cache results
            self.cache[cache_key] = final_results
            
            return final_results[:5]
            
        except Exception as e:
            logger.error(f"OEIS expression search error: {e}")
            return []
    
    def _clean_expression_for_search(self, expression: str) -> str:
        """Clean expression for OEIS search"""
        # Replace symbols with words
        cleaned = expression.replace('π', 'pi')
        cleaned = cleaned.replace('φ', 'phi')
        cleaned = cleaned.replace('γ', 'gamma')
        cleaned = cleaned.replace('sqrt', 'square root')
        cleaned = cleaned.replace('log', 'logarithm')
        cleaned = cleaned.replace('exp', 'exponential')
        
        return cleaned
    
    def _extract_search_terms(self, expression: str) -> List[str]:
        """Extract key search terms from mathematical expression"""
        terms = []
        
        # Mathematical constants
        if 'π' in expression or 'pi' in expression:
            terms.append('pi')
        if 'e^' in expression or 'exp' in expression:
            terms.append('exponential')
        if 'φ' in expression or 'phi' in expression:
            terms.append('golden ratio')
        if 'sqrt' in expression:
            terms.append('square root')
        if 'log' in expression:
            terms.append('logarithm')
        
        # Special numbers
        if '163' in expression:
            terms.append('163')
        if '262537412640768744' in expression:
            terms.append('Ramanujan constant')
        
        # Continued fractions
        if expression.count('/') > 1 and '+' in expression:
            terms.append('continued fraction')
        
        # Nested radicals
        if expression.count('sqrt') > 1:
            terms.append('nested radical')
        
        return terms
    
    def _search_oeis_term(self, term: str) -> List[OEISResult]:
        """Search OEIS for a specific term"""
        try:
            search_url = f"https://oeis.org/search?q={term}&language=english&go=Search"
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            for result_div in soup.find_all('div', class_='sequence')[:3]:  # Limit results
                try:
                    id_link = result_div.find('a', href=re.compile(r'/A\d+'))
                    if not id_link:
                        continue
                    
                    sequence_id = id_link.text.strip()
                    
                    name_elem = result_div.find('td', class_='rh')
                    name = name_elem.text.strip() if name_elem else "Unknown"
                    
                    desc_elem = result_div.find('td', class_='lh')
                    description = desc_elem.text.strip() if desc_elem else ""
                    
                    confidence = 0.3  # Base confidence for term matches
                    if term.lower() in description.lower():
                        confidence += 0.4
                    if term.lower() in name.lower():
                        confidence += 0.3
                    
                    result = OEISResult(
                        sequence_id=sequence_id,
                        name=name,
                        description=description,
                        url=f"https://oeis.org/{sequence_id}",
                        confidence=confidence
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error parsing OEIS term result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"OEIS term search error: {e}")
            return []
    
    def _calculate_confidence(self, text: str, value_str: str) -> float:
        """Calculate confidence score for OEIS match"""
        confidence = 0.0
        
        # Check if value appears in text
        if value_str in text:
            confidence += 0.8
        
        # Check for partial matches
        value_parts = value_str.split('.')
        if len(value_parts) == 2:
            integer_part, decimal_part = value_parts
            if integer_part in text:
                confidence += 0.3
            if decimal_part[:6] in text:  # First 6 decimal digits
                confidence += 0.4
        
        # Bonus for mathematical keywords
        math_keywords = ['pi', 'euler', 'golden', 'fibonacci', 'ramanujan', 'constant']
        for keyword in math_keywords:
            if keyword.lower() in text.lower():
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def verify_discovery(self, expression: str, value: float, error: float) -> Dict[str, any]:
        """Comprehensive verification of a mathematical discovery"""
        verification_result = {
            "expression": expression,
            "value": value,
            "error": error,
            "is_novel": True,
            "oeis_matches": [],
            "confidence_score": 0.0,
            "verification_timestamp": time.time()
        }
        
        try:
            # Search by value
            value_matches = self.search_oeis_by_value(value)
            
            # Search by expression
            expr_matches = self.search_oeis_by_expression(expression)
            
            # Combine and deduplicate matches
            all_matches = {}
            for match in value_matches + expr_matches:
                if match.sequence_id not in all_matches:
                    all_matches[match.sequence_id] = match
                else:
                    # Keep the one with higher confidence
                    if match.confidence > all_matches[match.sequence_id].confidence:
                        all_matches[match.sequence_id] = match
            
            matches = list(all_matches.values())
            matches.sort(key=lambda x: x.confidence, reverse=True)
            
            verification_result["oeis_matches"] = [
                {
                    "sequence_id": match.sequence_id,
                    "name": match.name,
                    "description": match.description,
                    "url": match.url,
                    "confidence": match.confidence
                }
                for match in matches[:5]
            ]
            
            # Determine if discovery is novel
            if matches and matches[0].confidence > 0.7:
                verification_result["is_novel"] = False
                verification_result["confidence_score"] = matches[0].confidence
            else:
                verification_result["is_novel"] = True
                verification_result["confidence_score"] = 1.0 - (matches[0].confidence if matches else 0.0)
            
        except Exception as e:
            logger.error(f"Discovery verification error: {e}")
            verification_result["error_message"] = str(e)
        
        return verification_result
    
    def batch_verify_discoveries(self, discoveries: List[Dict]) -> List[Dict]:
        """Verify multiple discoveries with rate limiting"""
        verified_discoveries = []
        
        for i, discovery in enumerate(discoveries):
            try:
                verification = self.verify_discovery(
                    discovery["expression"],
                    discovery["value"],
                    discovery["error"]
                )
                
                # Merge verification with discovery
                verified_discovery = {**discovery, **verification}
                verified_discoveries.append(verified_discovery)
                
                # Rate limiting
                if i < len(discoveries) - 1:
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error verifying discovery {discovery.get('expression', 'unknown')}: {e}")
                verified_discoveries.append(discovery)
        
        return verified_discoveries
