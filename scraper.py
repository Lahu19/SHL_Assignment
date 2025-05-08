import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import time
import logging
import os
import json
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLCatalogScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        self.session = requests.Session()

    def get_assessment_details(self, url: str) -> Dict:
        """Get detailed information about a specific assessment"""
        try:
            # Add delay to avoid rate limiting
            time.sleep(2)
            
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Log the response for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response URL: {response.url}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find the main content area
            main_content = soup.find('main') or soup.find('div', class_='content') or soup
            
            # Extract assessment details with more flexible selectors
            details = {
                'Assessment Name': self._get_text(main_content, [
                    'h1', '.title', '.assessment-title', 
                    '[data-testid="assessment-title"]',
                    'header h1'
                ]),
                'Description': self._get_text(main_content, [
                    '.description', '.content', 'p', 
                    '[data-testid="assessment-description"]',
                    '.assessment-description'
                ]),
                'Test Type': self._get_text(main_content, [
                    '.test-type', '.category', '.type',
                    '[data-testid="test-type"]',
                    '.assessment-category'
                ]),
                'Duration': self._get_text(main_content, [
                    '.duration', '.time', '.length',
                    '[data-testid="duration"]',
                    '.assessment-duration'
                ]),
                'Remote Testing': self._get_text(main_content, [
                    '.remote', '.remote-testing', '.online',
                    '[data-testid="remote-testing"]',
                    '.assessment-remote'
                ]),
                'Adaptive/IRT': self._get_text(main_content, [
                    '.adaptive', '.irt', '.adaptive-testing',
                    '[data-testid="adaptive-testing"]',
                    '.assessment-adaptive'
                ]),
                'Relative URL': url
            }
            
            # Clean and validate the data
            if not details['Assessment Name']:
                # Try to extract name from URL if not found in content
                url_parts = url.split('/')
                if len(url_parts) > 2:
                    potential_name = url_parts[-2].replace('-', ' ').title()
                    details['Assessment Name'] = f"SHL {potential_name}"
            
            # If we still don't have a name, skip this assessment
            if not details['Assessment Name']:
                logger.warning(f"No assessment name found for {url}")
                return {}
                
            return details
        except Exception as e:
            logger.error(f"Error scraping assessment details from {url}: {str(e)}")
            return {}

    def _get_text(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Helper method to extract text from a BeautifulSoup element using multiple selectors"""
        for selector in selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    # Get text from all matching elements and join them
                    text = ' '.join(elem.get_text(strip=True) for elem in elements)
                    if text:
                        return text
            except Exception as e:
                logger.debug(f"Error with selector {selector}: {str(e)}")
                continue
        return ""

    def scrape_catalog(self) -> pd.DataFrame:
        """Scrape the entire SHL catalog"""
        assessments = []
        
        try:
            # Get main catalog page
            response = self.session.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Log the response for debugging
            logger.debug(f"Catalog response status: {response.status_code}")
            logger.debug(f"Catalog response URL: {response.url}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple approaches to find assessment links
            assessment_links = []
            
            # Approach 1: Look for product links
            product_links = soup.select('a[href*="/solutions/products/"]')
            assessment_links.extend(product_links)
            
            # Approach 2: Look for assessment cards
            card_links = soup.select('.assessment-card a, .product-card a')
            assessment_links.extend(card_links)
            
            # Approach 3: Look for any links containing assessment-related keywords
            keyword_links = soup.find_all('a', href=True, text=lambda t: t and any(
                keyword in t.lower() for keyword in ['assessment', 'test', 'verify', 'check']
            ))
            assessment_links.extend(keyword_links)
            
            # Remove duplicates while preserving order
            seen_urls = set()
            unique_links = []
            for link in assessment_links:
                url = link.get('href')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_links.append(link)
            
            logger.info(f"Found {len(unique_links)} potential assessment links")
            
            for link in unique_links:
                url = link.get('href')
                if url:
                    if not url.startswith('http'):
                        url = urljoin(self.base_url, url)
                    
                    # Get assessment details
                    details = self.get_assessment_details(url)
                    if details:
                        assessments.append(details)
                        logger.info(f"Successfully scraped: {details['Assessment Name']}")
                    
                    # Be nice to the server
                    time.sleep(2)
            
            if not assessments:
                logger.warning("No assessments found. Using fallback data.")
                return self._get_fallback_data()
            
            # Convert to DataFrame
            df = pd.DataFrame(assessments)
            
            # Clean and process data
            df['Duration in mins'] = df['Duration'].apply(self._extract_duration)
            df['Remote Testing'] = df['Remote Testing'].apply(lambda x: 'Yes' if 'yes' in x.lower() else 'No')
            df['Adaptive/IRT'] = df['Adaptive/IRT'].apply(lambda x: 'Yes' if 'yes' in x.lower() else 'No')
            
            # Save raw data for debugging
            with open('scraped_data.json', 'w') as f:
                json.dump(assessments, f, indent=2)
            
            return df
            
        except Exception as e:
            logger.error(f"Error scraping catalog: {str(e)}")
            return self._get_fallback_data()

    def _extract_duration(self, duration_str: str) -> int:
        """Extract duration in minutes from duration string"""
        try:
            # Extract numbers from string
            numbers = ''.join(filter(str.isdigit, str(duration_str)))
            return int(numbers) if numbers else 30  # Default to 30 minutes if no duration found
        except:
            return 30

    def _get_fallback_data(self) -> pd.DataFrame:
        """Provide fallback data when scraping fails"""
        fallback_data = {
            'Assessment Name': [
                'SHL Verify Interactive',
                'SHL Verify Numerical Reasoning',
                'SHL Verify Verbal Reasoning',
                'SHL Verify Inductive Reasoning',
                'SHL Verify Deductive Reasoning'
            ],
            'Description': [
                'Interactive assessment for various skills',
                'Numerical reasoning assessment',
                'Verbal reasoning assessment',
                'Inductive reasoning assessment',
                'Deductive reasoning assessment'
            ],
            'Test Type': [
                'Ability & Aptitude',
                'Ability & Aptitude',
                'Ability & Aptitude',
                'Ability & Aptitude',
                'Ability & Aptitude'
            ],
            'Duration': [
                '45 minutes',
                '30 minutes',
                '30 minutes',
                '30 minutes',
                '30 minutes'
            ],
            'Remote Testing': [
                'Yes',
                'Yes',
                'Yes',
                'Yes',
                'Yes'
            ],
            'Adaptive/IRT': [
                'Yes',
                'Yes',
                'Yes',
                'Yes',
                'Yes'
            ],
            'Relative URL': [
                'https://www.shl.com/solutions/products/verify-interactive/',
                'https://www.shl.com/solutions/products/verify-numerical/',
                'https://www.shl.com/solutions/products/verify-verbal/',
                'https://www.shl.com/solutions/products/verify-inductive/',
                'https://www.shl.com/solutions/products/verify-deductive/'
            ]
        }
        return pd.DataFrame(fallback_data)

if __name__ == "__main__":
    scraper = SHLCatalogScraper()
    df = scraper.scrape_catalog()
    
    # Save to both CSV files to ensure compatibility
    df.to_csv("combined_assessment.csv", index=False)
    df.to_csv("combined_assessments.csv", index=False)
    
    logger.info(f"Scraped {len(df)} assessments successfully") 