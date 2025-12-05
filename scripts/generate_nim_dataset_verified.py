import os
import json
import time
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def generate_qa_pairs(api_key, base_url, content, model="meta/llama-3.3-70b-instruct"):
    """
    Generate QA pairs with source excerpts using LLM.
    """
    if not api_key:
        return []

    # Truncate content if too long
    if len(content) > 15000: 
        content = content[:15000] + "..."

    prompt = f"""
    You are an expert in creating evaluation datasets for RAG systems.
    Read the following technical documentation content and generate 3 to 5 high-quality question and answer pairs.
    
    CRITICAL REQUIREMENTS:
    1. The questions must be specific to the content provided.
    2. The "answer" (ground truth) must be based ONLY on the provided text.
    3. For each pair, you MUST extract the exact "source_excerpt" from the text that supports the answer. This allows human verification.
    
    Format the output as a valid JSON array of objects. Each object must have these exact keys:
    - "question": The question string
    - "ground_truth": The correct answer string
    - "source_excerpt": The direct quote from the text that proves the answer is correct
    
    Do not include markdown formatting (like ```json). Just the raw JSON string.
    
    Content:
    {content}
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Construct endpoint
    endpoint = base_url.rstrip('/') + "/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2048, # Increased for excerpts
        "stream": False
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        
        # Log response status and content for debugging
        if response.status_code != 200:
            print(f"\n[DEBUG] API Error Status: {response.status_code}")
            print(f"[DEBUG] API Error Content: {response.text[:500]}")
            
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Clean up markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"\n[DEBUG] JSON Decode Error: {e}")
            print(f"[DEBUG] Raw Content Received: {content[:500]}")
            return []
            
    except Exception as e:
        print(f"Error generating QA: {e}")
        return []

def get_all_links(start_url):
    """
    Simple crawler to get links from the same documentation section.
    """
    print(f"Fetching links from {start_url}...")
    try:
        response = requests.get(start_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {start_url}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    links = set()
    
    links.add(start_url)
    # Simple heuristic: get all links in the same directory/section
    prefix = start_url.rsplit('/', 1)[0] 
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(start_url, href)
        
        if '#' in full_url:
            full_url = full_url.split('#')[0]
            
        if full_url.startswith(prefix) and full_url.endswith('.html'):
            links.add(full_url)
            
    return sorted(list(links))

def extract_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch content {url}: {e}")
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove irrelevant elements
    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
        script.extract()

    text = soup.get_text(separator=' ', strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate NIM Doc Dataset with Excerpts")
    parser.add_argument("--output", default="../my_data/nim_docs_eval_with_source.json", help="Output JSON file")
    parser.add_argument("--limit", type=int, default=3, help="Max pages to process")
    parser.add_argument("--model", default="meta/llama-3.1-70b-instruct", help="LLM model to use")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Get env vars
    api_key = os.environ.get("NVIDIA_API_KEY")
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    if not api_key:
        print("Error: NVIDIA_API_KEY not set. Please set it to run LLM generation.")
        return
    
    start_url = "https://docs.nvidia.com/nim/large-language-models/latest/introduction.html"
    links = get_all_links(start_url)
    
    print(f"Found {len(links)} links. Processing first {args.limit}...")
    
    all_data = []
    count = 0
    
    for url in links:
        if count >= args.limit:
            break
            
        print(f"Processing [{count+1}/{args.limit}]: {url}")
        content = extract_content(url)
        
        if not content:
            continue
            
        qa_pairs = generate_qa_pairs(api_key, base_url, content, args.model)
        if qa_pairs:
            for pair in qa_pairs:
                pair["id"] = str(len(all_data) + 1)
                pair["source_url"] = url  # Add URL for reference
                all_data.append(pair)
            print(f"  Generated {len(qa_pairs)} verified pairs.")
        else:
            print("  No pairs generated.")
            
        count += 1
        time.sleep(1) 

    with open(args.output, 'w') as f:
        json.dump(all_data, f, indent=4)
        
    print(f"Done. Saved {len(all_data)} verified items to {args.output}")

if __name__ == "__main__":
    main()

