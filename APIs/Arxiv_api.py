import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict
import json

class ArxivAPI:
    BASE_URL = 'http://export.arxiv.org/api/query?'
    
    @staticmethod
    def search(query: str, max_results: int = 10) -> List[Dict]:
        """
        Search arXiv and return papers matching the query
        """
        # Encode query parameters
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        # Construct URL
        url = ArxivAPI.BASE_URL + urllib.parse.urlencode(params)
        
        try:
            # Fetch data
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
            
            # Parse XML
            root = ET.fromstring(xml_data)
            
            # Namespace for parsing
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Extract paper information
            papers = []
            for entry in root.findall('atom:entry', ns):
                paper = {
                    'title': entry.find('atom:title', ns).text.strip(),
                    'authors': [
                        author.find('atom:name', ns).text 
                        for author in entry.findall('atom:author', ns)
                    ],
                    'summary': entry.find('atom:summary', ns).text.strip(),
                    'published': entry.find('atom:published', ns).text.split('T')[0],
                    'pdf_url': entry.find('atom:link[@title="pdf"]', ns).get('href'),
                    'arxiv_url': entry.find('atom:id', ns).text,
                    'source': 'arXiv'
                }
                papers.append(paper)
            
            return papers
        
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
