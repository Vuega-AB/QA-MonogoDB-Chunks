import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Function to extract all links from a page
def get_links_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

# Function to extract PDF links from a page
def get_pdfs_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.pdf')]
    return pdf_links

def get_next_page_url(soup, current_url):
    next_page = None
    
    # Try to find common pagination containers (nav, div, ul)
    pagination = soup.find("nav", class_="pagination") or soup.find("ul", class_="pagination") or soup.find("div", class_="pagination")

    if pagination:
        # Find the currently active page
        active_page = pagination.find("li", class_="active")  # Example: <li class="active"><a>2</a></li>
        if active_page and active_page.find_next_sibling("li"):
            next_link = active_page.find_next_sibling("li").find("a", href=True)
            if next_link:
                next_page = urljoin(current_url, next_link["href"])
    
    # Fallback: Look for any 'Next' or arrow button
    if not next_page:
        next_buttons = soup.find_all("a", href=True)
        for btn in next_buttons:
            btn_text = btn.get_text(strip=True).lower()
            if btn_text in ["next", ">", "»", "nästa"]:  # Add localized words if needed
                next_page = urljoin(current_url, btn["href"])
                break  # Stop at the first match

    return next_page

# Scraping function with dynamic pagination handling
def scrape_pdfs(base_url, output_file='pdf_links.txt', max_pages=50):
    pdfs = set()
    current_page_url = base_url
    page_count = 0

    with open(output_file, 'w') as file:
        while current_page_url:
            page_count += 1
            print(f"Processing Page {page_count}: {current_page_url}")

            # Request the page and parse it
            response = requests.get(current_page_url)
            if response.status_code != 200:
                print(f"Stopping: Page {page_count} returned {response.status_code}")
                break  # Stop if page doesn't exist

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all links from the current page
            page_links = get_links_from_page(current_page_url)

            # Extract PDFs from each listed link on the page
            for link in page_links:
                absolute_url = urljoin(current_page_url, link)
                pdf_urls = get_pdfs_from_page(absolute_url)

                # Write new PDFs to file
                for pdf_url in pdf_urls:
                    if pdf_url not in pdfs:
                        pdfs.add(pdf_url)
                        file.write(pdf_url + '\n')
                        print(f"Found PDF: {pdf_url}")

            # Find the next page dynamically
            current_page_url = get_next_page_url(soup, current_page_url)

            # Stop if no more pages or max pages reached
            if not current_page_url or page_count >= max_pages:
                print("No more pages found or reached max pages. Stopping.")
                break

# Example usage
main_page_url = 'https://www.edpb.europa.eu/news/news_en'  # Replace with actual website
scrape_pdfs(main_page_url)
