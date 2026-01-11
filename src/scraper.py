import asyncio
import os
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# --- CONFIGURATION ---
BASE_URL = "https://www.maa.ac.in/"
MAX_PAGES = 100
OUTPUT_DIR = "crawled_data"

# File extensions to IGNORE
IGNORED_EXTENSIONS = (
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
    '.css', '.js', '.json', '.xml', '.ico',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.mp3', '.mp4', '.avi', '.mov', '.mkv',
    '.xls', '.xlsx', '.csv', '.doc', '.docx', '.ppt', '.pptx',
    '.exe', '.bin', '.iso', '.dmg'
)

def normalize_domain(url):
    """Removes 'www.' to compare domains accurately."""
    netloc = urlparse(url).netloc
    return netloc.replace('www.', '')

def get_safe_filename(url):
    """
    Converts a URL into a Windows-safe filename.
    1. Removes protocol (https://)
    2. Replaces illegal chars (? = & / :) with underscores (_)
    3. Truncates to 100 chars to avoid path limit errors
    """
    # Remove http/https
    clean_name = url.replace("https://", "").replace("http://", "")
    
    # Replace ALL illegal characters with underscore
    # Forbidden on Windows: < > : " / \ | ? *
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', clean_name)
    
    # Limit length (Windows has a 255 char path limit)
    return clean_name[:100]

async def crawl_recursive():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    visited = set()
    queue = [BASE_URL]
    target_domain = normalize_domain(BASE_URL)
    
    print(f"🚀 Starting Deep Crawl for: {BASE_URL}")
    print(f"🎯 Target Domain: {target_domain}")

    browser_conf = BrowserConfig(
        headless=True,
        accept_downloads=True, # Prevents crash on download links
        verbose=False
    )
    
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        while queue and len(visited) < MAX_PAGES:
            current_url = queue.pop(0)
            
            # Normalize URL (strip fragment #)
            current_url = current_url.split('#')[0]

            if current_url in visited:
                continue

            # Skip binary files
            if current_url.lower().split('?')[0].endswith(IGNORED_EXTENSIONS):
                continue
            
            visited.add(current_url)
            print(f"🕷️ Crawling ({len(visited)}/{MAX_PAGES}): {current_url}")

            try:
                # 1. SCRAPE
                result = await crawler.arun(url=current_url, config=run_conf)
                
                if not result.success:
                    print(f"   ⚠️ Failed to load: {current_url}")
                    continue

                # 2. SAVE CONTENT (With Fixed Filename Logic)
                safe_name = get_safe_filename(current_url)
                filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.md")
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Source: {current_url}\n\n")
                    f.write(result.markdown)

                # 3. EXTRACT LINKS (Hybrid: Crawl4AI + BeautifulSoup)
                found_links = set()

                # Method A: Crawl4AI detected links
                if result.links and "internal" in result.links:
                    for link in result.links['internal']:
                        found_links.add(link['href'])

                # Method B: Manual BS4 Parse (catches what Method A misses)
                soup = BeautifulSoup(result.html, 'html.parser')
                for a_tag in soup.find_all('a', href=True):
                    raw_link = a_tag['href']
                    absolute_link = urljoin(current_url, raw_link)
                    found_links.add(absolute_link)

                # 4. FILTER & QUEUE
                new_links_count = 0
                for link in found_links:
                    clean_link = link.split('#')[0]
                    link_domain = normalize_domain(clean_link)

                    if (link_domain == target_domain and 
                        clean_link not in visited and 
                        clean_link not in queue and
                        not clean_link.lower().split('?')[0].endswith(IGNORED_EXTENSIONS)):
                        
                        queue.append(clean_link)
                        new_links_count += 1
                
                print(f"   ↳ Added {new_links_count} new pages.")

            except Exception as e:
                print(f"   ❌ Error: {e}")

            await asyncio.sleep(0.2)

    print(f"✅ DONE. Scraped {len(visited)} pages.")

if __name__ == "__main__":
    asyncio.run(crawl_recursive())