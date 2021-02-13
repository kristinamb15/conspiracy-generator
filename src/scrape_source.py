# Scrape Wikipedia for source material to build dataset from
from utilities import *

base_url = 'https://en.wikipedia.org'
consp_url = '/wiki/Category:Conspiracy_theories_in_the_United_States'

def get_urls(base_url, page_url):
    """Collects urls from wikipedia category page.

    Args:
        base_url: base url of pages.
        page_url: specific page url.
    Returns:
        List of urls.

    """
    urls = []
    res = requests.get(base_url + page_url)
    soup = bs4.BeautifulSoup(res.text, 'lxml')

    # Find hrefs on page
    soup = soup.find_all(class_='mw-category-group')

    for item in soup:
        for a in item.find_all('a', href=True):
            urls.append(a['href'])

    return urls

def get_page_text(base_url, page_url):
    """Collects main text from wikipedia page.

    Args:
        base_url: base url of pages.
        page_url: specific page url.
    Returns:
        Page text.

    """
    text = []
    res = requests.get(base_url + page_url)
    soup = bs4.BeautifulSoup(res.text, 'lxml')
    soup = soup.find_all('p')
    for p in soup:
        text.append(p.text.strip())

    return text

if __name__ == '__main__':
    # Collect urls from wikipedia category page
    urls = get_urls(base_url, consp_url)

    # Find all urls that do not represent category pages (further scraping of category pages is required)
    all_urls = [url for url in urls if not url.startswith('/wiki/Category:')]
    # For urls that represent category pages, collect urls from those pages
    for url in urls:
        if url.startswith('/wiki/Category:'):
            all_urls.extend(get_urls(base_url, url))
    
    # Collect text for all urls
    url_text = []
    for url in all_urls:
        text = get_page_text(base_url, url)
        new_dict = {'page': url, 'text': ' '.join(text)}
        url_text.append(new_dict)

    # Create dataframe and save to file
    conspiracy_df = pd.DataFrame(url_text)
    conspiracy_df.to_csv(path.join(raw_data, 'conspiracy_df.csv'), index=False)
