import requests
import json
from bs4 import BeautifulSoup
import time
from datetime import datetime
import random
import pandas as pd

# CORS proxies that can be used
CORS_PROXIES = [
    'https://api.allorigins.win/raw?url=',
    'https://api.codetabs.com/v1/proxy?quest=',
    'https://corsproxy.io/?'
]

def super_fetch(url, method='GET', headers=None, timeout=15):
    """Fetch a URL using multiple proxies for CORS issues"""
    headers = headers or {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for proxy_url in CORS_PROXIES:
        try:
            full_url = f"{proxy_url}{url}"
            response = requests.get(full_url, headers=headers, timeout=timeout)
            
            if response.status_code >= 200 and response.status_code < 300:
                return {
                    'ok': True,
                    'status': response.status_code,
                    'statusText': response.reason,
                    'text': response.text,
                    'json': response.json() if 'application/json' in response.headers.get('Content-Type', '') else None
                }
        except Exception as e:
            print(f"Proxy {proxy_url} failed: {str(e)}")
            # If this is the last proxy, raise the error
            if proxy_url == CORS_PROXIES[-1]:
                raise e
    
    raise Exception('All proxies failed')

def perform_duckduckgo_search(query):
    """Perform a search using DuckDuckGo"""
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    
    try:
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if not response.ok:
            return {
                'success': False,
                'error': f"Failed to fetch search results: {response.status_code}",
                'errorDetails': response.reason
            }
        
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        results = []
        result_elements = soup.select('.result__body')
        
        for result in result_elements:
            title_element = result.select_one('.result__title')
            snippet_element = result.select_one('.result__snippet')
            link_element = title_element.select_one('a') if title_element else None
            
            if link_element and title_element:
                title = title_element.get_text(strip=True)
                url = link_element.get('href', '')
                snippet = snippet_element.get_text(strip=True) if snippet_element else ''
                
                # Extract thumbnail if available
                thumbnail = ''
                image_element = result.select_one('.result__image')
                if image_element:
                    thumbnail = image_element.get('src', '')
                
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet,
                    'thumbnail': thumbnail,
                    'source': 'DuckDuckGo',
                    'dateRetrieved': datetime.now().isoformat()
                })
        
        return {
            'success': True,
            'query': query,
            'results': results,
            'count': len(results),
            'resultSummary': f"{len(results)} results found for '{query}'."
        }
    except Exception as e:
        print(f"Error fetching search results: {str(e)}")
        return {
            'success': False,
            'error': "Failed to fetch search results. Please try again.",
            'errorDetails': str(e)
        }

def fetch_website_content(url):
    """Fetch content from a website"""
    try:
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }, timeout=15)
        
        if not response.ok:
            return f"Failed to fetch {url}: {response.status_code} {response.reason}"
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts, styles and other non-content elements
        for script in soup(["script", "style", "meta", "noscript", "iframe"]):
            script.extract()
            
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Truncate if too long
        if len(text) > 5000:
            text = text[:5000] + "...[truncated]"
            
        return text
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"

def extract_structured_data(text, columns):
    """Extract structured data from text content based on column names"""
    try:
        # Use Gemini API to extract structured data
        import google.generativeai as genai
        
        # Configure the Gemini API
        api_key = "AIzaSyB_dNcoAstUzFWW3c_N5vHtl08YfSYSxOA"
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        
        # Create the prompt for extraction
        prompt = f"""
        Extract structured data from the following text to create a dataset with these columns: {', '.join(columns)}.
        
        Text content:
        {text[:3000]}  # Limiting to 3000 chars to avoid token limits
        
        Output the data as a CSV table with header row. 
        Generate 5-10 rows of high-quality, relevant data based on the content.
        Do not include any explanations, just return the CSV data.
        """
        
        # Generate the structured data
        response = model.generate_content(prompt)
        
        # Parse the CSV data
        csv_data = response.text.strip()
        
        # Clean up the response if needed (remove markdown code blocks)
        if csv_data.startswith("```") and csv_data.endswith("```"):
            csv_data = csv_data[3:-3].strip()
        if csv_data.startswith("```csv") and csv_data.endswith("```"):
            csv_data = csv_data[6:-3].strip()
            
        # Parse the CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data))
        
        # Ensure all requested columns are present
        for col in columns:
            if col not in df.columns:
                df[col] = ""
                
        # Keep only the requested columns
        df = df[[col for col in columns if col in df.columns]]
        
        return df
    except Exception as e:
        print(f"Error extracting structured data: {str(e)}")
        # Return an empty DataFrame with the requested columns
        return pd.DataFrame(columns=columns)

def create_dataset_from_web_search(query, columns, num_rows):
    """Generate a dataset from web search results"""
    # Perform the search
    search_results = perform_duckduckgo_search(query)
    
    if not search_results['success'] or not search_results['results']:
        return None, "Failed to get search results or no results found."
    
    # Collect content from top results
    all_content = ""
    for i, result in enumerate(search_results['results'][:3]):  # Use top 3 results
        content = fetch_website_content(result['url'])
        all_content += f"\n\nSOURCE {i+1}: {result['title']}\n{content}\n"
        # Add a small delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.5))
    
    # Extract structured data
    df = extract_structured_data(all_content, columns)
    
    # If we don't have enough rows, generate more using Gemini
    if len(df) < num_rows:
        # Configure the Gemini API
        import google.generativeai as genai
        api_key = "AIzaSyB_dNcoAstUzFWW3c_N5vHtl08YfSYSxOA"
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        
        # Create a sample from existing data
        sample = df.head(min(5, len(df))).to_csv(index=False) if not df.empty else ""
        
        # Generate more data
        prompt = f"""
        Generate additional rows for this dataset with columns: {', '.join(columns)}.
        The dataset is about: {query}
        
        Here's a sample of existing data:
        {sample}
        
        Generate {num_rows - len(df)} more rows in CSV format without headers.
        The data should be consistent with the existing sample and related to the topic.
        """
        
        try:
            response = model.generate_content(prompt)
            csv_data = response.text.strip()
            
            # Clean up the response
            if csv_data.startswith("```") and csv_data.endswith("```"):
                csv_data = csv_data[3:-3].strip()
            if csv_data.startswith("```csv") and csv_data.endswith("```"):
                csv_data = csv_data[6:-3].strip()
                
            # Parse the additional data
            from io import StringIO
            additional_df = pd.read_csv(StringIO(csv_data), names=columns)
            
            # Combine with existing data
            df = pd.concat([df, additional_df], ignore_index=True)
            
            # Limit to requested number of rows
            if len(df) > num_rows:
                df = df.iloc[:num_rows]
        except Exception as e:
            print(f"Error generating additional data: {str(e)}")
    
    return df, search_results
