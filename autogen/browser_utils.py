import traceback
import json
import os
import requests
import re
import markdownify
import io
import uuid
import mimetypes
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Optional, Union, Tuple

# Optional PDF support
IS_PDF_CAPABLE = False
try:
    import pdfminer
    import pdfminer.high_level

    IS_PDF_CAPABLE = True
except ModuleNotFoundError:
    pass

# Other optional dependencies
try:
    import pathvalidate
except ModuleNotFoundError:
    pass


class SimpleTextBrowser:
    """(In preview) An extremely simple text-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
        self,
        start_page: Optional[str] = None,
        viewport_size: Optional[int] = 1024 * 8,
        downloads_folder: Optional[Union[str, None]] = None,
        bing_api_key: Optional[Union[str, None]] = None,
        request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        self.history: List[str] = list()
        self.page_title: Optional[str] = None
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.bing_api_key = bing_api_key
        self.request_kwargs = request_kwargs

        self._page_content = ""

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1]

    def set_address(self, uri_or_path: str) -> None:
        self.history.append(uri_or_path)

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("bing:"):
            self._bing_search(uri_or_path[len("bing:") :].strip())
        else:
            if not uri_or_path.startswith("http:") and not uri_or_path.startswith("https:"):
                uri_or_path = urljoin(self.address, uri_or_path)
                self.history[-1] = uri_or_path  # Update the address with the fully-qualified path
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def visit_page(self, path_or_uri: str) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri)
        return self.viewport

    def _split_pages(self) -> None:
        # Split only regular pages
        if not self.address.startswith("http:") and not self.address.startswith("https:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _bing_api_call(self, query: str) -> Dict[str, Dict[str, List[Dict[str, Union[str, Dict[str, str]]]]]]:
        # Make sure the key was set
        if self.bing_api_key is None:
            raise ValueError("Missing Bing API key.")

        # Prepare the request parameters
        request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}

        if "headers" not in request_kwargs:
            request_kwargs["headers"] = {}
        request_kwargs["headers"]["Ocp-Apim-Subscription-Key"] = self.bing_api_key

        if "params" not in request_kwargs:
            request_kwargs["params"] = {}
        request_kwargs["params"]["q"] = query
        request_kwargs["params"]["textDecorations"] = False
        request_kwargs["params"]["textFormat"] = "raw"

        request_kwargs["stream"] = False

        # Make the request
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", **request_kwargs)
        response.raise_for_status()
        results = response.json()

        return results  # type: ignore[no-any-return]

    def _bing_search(self, query: str) -> None:
        results = self._bing_api_call(query)

        web_snippets: List[str] = list()
        idx = 0
        for page in results["webPages"]["value"]:
            idx += 1
            web_snippets.append(f"{idx}. [{page['name']}]({page['url']})\n{page['snippet']}")
            if "deepLinks" in page:
                for dl in page["deepLinks"]:
                    idx += 1
                    web_snippets.append(
                        f"{idx}. [{dl['name']}]({dl['url']})\n{dl['snippet'] if 'snippet' in dl else ''}"  # type: ignore[index]
                    )

        news_snippets = list()
        if "news" in results:
            for page in results["news"]["value"]:
                idx += 1
                news_snippets.append(f"{idx}. [{page['name']}]({page['url']})\n{page['description']}")

        self.page_title = f"{query} - Search"

        content = (
            f"A Bing search for '{query}' found {len(web_snippets) + len(news_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )
        if len(news_snippets) > 0:
            content += "\n\n## News Results:\n" + "\n\n".join(news_snippets)
        self._set_page_content(content)

    def _fetch_page(self, url: str) -> None:
        try:
            # Prepare the request parameters
            request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
            request_kwargs["stream"] = True

            # Send a HTTP request to the URL
            response = requests.get(url, **request_kwargs)
            response.raise_for_status()

            # If the HTTP request returns a status code 200, proceed
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                for ct in ["text/html", "text/plain", "application/pdf"]:
                    if ct in content_type.lower():
                        content_type = ct
                        break

                if content_type == "text/html":
                    # Get the content of the response
                    html = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        html += chunk

                    soup = BeautifulSoup(html, "html.parser")

                    # Remove javascript and style blocks
                    for script in soup(["script", "style"]):
                        script.extract()

                    # Convert to markdown -- Wikipedia gets special attention to get a clean version of the page
                    if url.startswith("https://en.wikipedia.org/"):
                        body_elm = soup.find("div", {"id": "mw-content-text"})
                        title_elm = soup.find("span", {"class": "mw-page-title-main"})

                        if body_elm:
                            # What's the title
                            main_title = soup.title.string
                            if title_elm and len(title_elm) > 0:
                                main_title = title_elm.string
                            webpage_text = (
                                "# " + main_title + "\n\n" + markdownify.MarkdownConverter().convert_soup(body_elm)
                            )
                        else:
                            webpage_text = markdownify.MarkdownConverter().convert_soup(soup)
                    else:
                        webpage_text = markdownify.MarkdownConverter().convert_soup(soup)

                    # Convert newlines
                    webpage_text = re.sub(r"\r\n", "\n", webpage_text)

                    # Remove excessive blank lines
                    self.page_title = soup.title.string
                    self._set_page_content(re.sub(r"\n{2,}", "\n\n", webpage_text).strip())
                elif content_type == "text/plain":
                    # Get the content of the response
                    plain_text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        plain_text += chunk

                    self.page_title = None
                    self._set_page_content(plain_text)
                elif IS_PDF_CAPABLE and content_type == "application/pdf":
                    pdf_data = io.BytesIO(response.raw.read())
                    self.page_title = None
                    self._set_page_content(pdfminer.high_level.extract_text(pdf_data))
                elif self.downloads_folder is not None:
                    # Try producing a safe filename
                    fname = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension

                    # Open a file for writing
                    download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Return a page describing what just happened
                    self.page_title = "Download complete."
                    self._set_page_content(f"Downloaded '{url}' to '{download_path}'.")
                else:
                    self.page_title = f"Error - Unsupported Content-Type '{content_type}'"
                    self._set_page_content(self.page_title)
            else:
                self.page_title = "Error"
                self._set_page_content("Failed to retrieve " + url)
        except requests.exceptions.RequestException as e:
            self.page_title = "Error"
            self._set_page_content(str(e))


get_scheme    = lambda url: urlparse(url).scheme if isinstance(url,str)  else url.scheme
get_domain    = lambda url: urlparse(url).netloc if isinstance(url,str)  else url.netloc
get_path      = lambda url: urlparse(url).path   if isinstance(url, str) else url.path
get_last_path = lambda url: os.path.basename(urlparse(url).path) if isinstance(url, str) else os.path.basename(url.path)

def get_file_path_from_url(url): # URL to Directory function
    """
    get_file_path_from_url function: This function takes a URL as input and returns the corresponding local file path as a string.

    Parameters:
    url (str | ParseResult): The URL of the file for which the local path is to be obtained.

    Returns:
    str: The local file path on the system as a string.
    """

    # Remove any trailing forward slash
    url = url[:-1] if url[-1] == '/' else url

    # Parse the URL
    parsed_url    = urlparse(url) if isinstance(url, str) else url
    canonical_url = parsed_url.netloc.replace("www.","")

    if 'github' in url and len(parsed_url.path.split('/')) >= 2:
        relative_path = os.path.join(canonical_url, parsed_url.path)
    elif len(parsed_url.path.split('/')) >= 1:
        relative_path = os.path.join(canonical_url, get_last_path(parsed_url))

    return relative_path

def get_protocol(url): # Get the protocol for a URL since occasionally links are malformed
    """
    Determines and returns the protocol (http, https, or ftp) of a given URL.

    Parameters:
    - url (str/Url): The input URL to be parsed and analyzed for its protocol.

    Returns:
    - str: A string representing the protocol of the provided URL (http, https, or ftp).
    """

    parsed_url = urlparse(url) if isinstance(url, str) else url
    if parsed_url.scheme == 'https':
        return "https://"
    elif parsed_url.scheme == 'http':
        return "http://"
    elif parsed_url.scheme == 'ftp':
        return "ftp://"
    else:
        return None  # Handle cases where the URL doesn't have a scheme or is invalid

def analyze_href(href): # Returns a helper dictionary 
  """
  Analyzes an href link and returns a dictionary indicating the presence of a domain and schema.

  Args:
      href (str): The href link to analyze.

  Returns:
      dict: A dictionary with keys 'has_domain' and 'has_schema', both set to True or False.
  """
  parsed_url = urlparse(href)
  return {
      'has_domain': bool(parsed_url.netloc),
      'has_schema': bool(parsed_url.scheme)
  }

def fix_missing_protocol(img_url, domain): # Correct a url if it's missing the protocol
    """
    Fixes a URL by adding the missing protocol (http or https) based on the provided domain.

    Parameters:
    - img_url (str): The input image URL to be fixed.
    - domain (str): The domain of the image URL which is used to determine the protocol.

    Returns:
    - str: A corrected URL string with the missing protocol added.
    """

    protocol = get_scheme(domain)
    img_attr = analyze_href(img_url)
    if img_url.startswith('//'):  # If the URL starts with "//"
        img_url = protocol + img_url[2:]  # Add "https://" before it
    elif not img_attr['has_domain']: # domain not in img_url:
        img_url = domain+img_url
    return img_url

def SeleniumBrowser(**kwargs): # Function that loads the web driver
    """
    This function launches a headless Selenium browser based on the specified 'browser'. The available options are 'edge', 'firefox', and 'chrome'.
    
    Parameters:
        browser (str): A string specifying which browser to launch. Defaults to 'firefox'. 
        download_dir (str): A path to where downloaded files are stored.  Defaults to None

    Returns:
        webdriver: An instance of the Selenium WebDriver based on the specified browser.  User can open a new page by `webdriver.get('https://www.microsoft.com')`.
        
    Raises:
        ImportError: If selenium package is not installed, it raises an ImportError with a message suggesting to install it using pip.
    """    
    
    # Load the argumnets from kwargs
    browser      = kwargs.get('browser', 'edge')
    download_dir = kwargs.get('download_dir', None)

    try:
        from selenium import webdriver
    except ImportError as e:
        logging.fatal("Failed to import selenium. Try running 'pip install selenium'.  You may need to run 'sudo easy_install selenium' on Linux or MacOS")
        raise e

    def get_headless_options(download_dir):
        options = Options()
        options.headless = True
        options.add_argument('--headless')
        options.add_argument("--window-size=1920,5200")
        options.add_argument('--downloadsEnabled')
        if download_dir:
            options.set_preference("download.default_directory",download_dir)
        return options

    if browser.lower()=='firefox':
        from selenium.webdriver.firefox.options import Options
        driver = webdriver.Firefox(options=get_headless_options(download_dir))
    elif browser.lower()=='chrome':
        from selenium.webdriver.chrome.options import Options
        driver = webdriver.Chrome(options=get_headless_options(download_dir))
    elif browser.lower()=='edge':
        from selenium.webdriver.edge.options import Options
        driver = webdriver.Edge(options=get_headless_options(download_dir))
    driver.capabilities['se:downloadsEnablead'] = True
    
    return driver 

def extract_pdf_text(local_pdf_path):
    """
    Extracts the text content from a local PDF file and returns it as a string.

    Parameters:
    - local_pdf_path (str): The path to the local PDF file from which the text will be extracted.

    Returns:
    - str: A string containing the text content of the provided PDF file.
    """

    try:
        import pdfminer
        text = pdfminer.high_level.extract_text(local_pdf_path)
    except:
        try:
            from pdfminer import high_level
            text = high_level.extract_text(local_pdf_path)        
        except: 
            text = ''

    return text

class ContentCollector():
    
    def __init__(self, storage_path='./content', page_loading_time=5, *args, **kwargs):

        from collections import deque
        self.additional_links    = deque()

        self.link_depth      = 0
        self.local_dir       = storage_path
        self.page_load_time  = page_loading_time
        self.request_kwargs  = kwargs['request_kwargs']
        
        self.llm_config      = kwargs['llm_config']

        import tiktoken
        from guidance import models
        self.small_lm = models.OpenAIChat(model="small", base_url="http://localhost:5001/v1", api_key="not-needed", tokenizer=tiktoken.get_encoding('cl100k_base'), temperature=0.01)


    # Main entry point
    def ingest_link(self, recipient, messages, sender, config):
        content_type, content = '', ''
        success = False
        all_links = []
        for message in messages:
            if message.get("role") == "user":
                links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message.get("content"))
                for link in links:
                    all_links.append(link)

        for link in all_links:
            content_type, content = self.fetch_content(link)
        
        # Inform self that it has completed the root level of link(s)
        self.link_depth = 1

        while len(self.additional_links) > 0:
            content_type, content = self.fetch_content( self.additional_links.pop() )
        

        self.link_depth = 0
        return content_type, content

    def fetch_content(self, link):

        # if not self.validate_link(link):
        #     return "Error", "Invalid link. Please submit a valid URL."
        
        # Parse the link        
        parsed_url = urlparse(link)

        # A special case for arxiv links
        if 'arxiv' in link:
            return 'pdf', self.fetch_arxiv_content(parsed_url)
        
        elif parsed_url.path.endswith('.pdf'):
            return 'pdf', self.fetch_pdf_content(link)
        
        else:
            return 'html', self.fetch_html_content(link)

    def fetch_html_content(self, link):
            
        browser = SeleniumBrowser(browser='firefox') #, download_dir=sd['browser_screenshot_path'])
        browser.get(link)
        browser.maximize_window()
        browser.implicitly_wait(self.page_load_time)


        # Handle web page content
        sd = {} # submission_data
        sd['url'] = link
        sd['local_path'] = os.path.join( self.local_dir, get_file_path_from_url(link) )
        os.makedirs(sd['local_path'], exist_ok=True)

        sd['browser_screenshot_path'] = os.path.join( sd['local_path'], "screenshot.png" )
        
        # Save a screenshot of the browser window
        browser.save_full_page_screenshot(sd['browser_screenshot_path'])
        
        sd['title'] = browser.title
        sd['html']  = browser.page_source

        # Write the HTML to disk for archival purposes
        with open(os.path.join(sd['local_path'],'index.html'), 'w', encoding='utf-8') as f:
            f.write(str(browser.page_source))

        # Store the BS object
        sd['soup'] = BeautifulSoup(sd['html'], 'html.parser')
        
        sd['content'] = self.identify_content( sd['soup'] )
        
        # Save the content to a text file on disk
        with open(os.path.join(sd['local_path'], "content.txt"), "w") as f:
            for data in sd['content']: # Iterate over each record
                f.write(data + "\n") # Write the content to the file

        # Save the original URL for convenience elsewhere (when parsing images)
        sd['soup'].url = link

        # Parse and store the Metadata
        sd['meta'] = self.identify_metadata(sd['soup']) # [ data.attrs for data in sd['soup'].find_all("meta") ]
        
        # Open a file to write the metadata to
        with open(os.path.join(sd['local_path'], "metadata.txt"), "w") as f:
            for data in sd['meta']: # Iterate over each record
                f.write(json.dumps(data) + "\n") # Write the link to the file

        # Parse and store the links
        sd['links'] = [{'text': link.get_text().strip(), 'href': link['href']} for link in sd['soup'].find_all('a') if link.has_attr('href') and '/' in link['href']]
        
        # Open a file to write the link URLs to
        with open(os.path.join(sd['local_path'], "links.txt"), "w") as f:
            for link in sd['links']: # Iterate over each link
                f.write(json.dumps(link) + "\n") # Write the link to the file

                # Recursive link checking, up to 1 level deep past the root
                if self.link_depth < 1:

                    # Check if we find any useful relevant links that we should catalog
                    if ('project' in link['text'] or 'paper' in link['text'] or 'code' in link['text']) and 'marktekpost' in link['href'].lower():
                        self.additional_links.append(link['href'])
                    elif 'arxiv' in link['href'] or 'github' in link['href']:
                        self.additional_links.append(link['href'])
                
        # Parse and store the images
        self.collect_images(sd['soup'], sd['local_path'])
                
        # Close down the browser
        browser.quit()

        return 'success'
    
    def fetch_pdf_content(self, link):

        local_pdf_path = os.path.join( self.local_dir,
            os.path.join( get_file_path_from_url(link), link.split('/')[-1] )
        )
        os.makedirs(local_pdf_path, exist_ok=True)


        response = requests.get(link, params={'headers': self.request_kwargs})

        if response.status_code == 200:
            with open(local_pdf_path, 'wb') as f:
                f.write(response.content)
        
            # Extract text from the PDF file
            text = extract_pdf_text(local_pdf_path)

            return text
        else:
            return None

    def fetch_arxiv_content(self, link):
        # Import the arxiv library
        import arxiv

        # Identify the paper identification
        arxiv_id = link.path.split('/')[-1]

        # Define the local directory
        local_base_path = os.path.join( self.local_dir, get_file_path_from_url(link) )
        os.makedirs(local_base_path, exist_ok=True)

        local_pdf_path = os.path.join( local_base_path, f"{arxiv_id}.pdf" )

        # Download the paper if we don't already have it
        if not os.path.exists(local_pdf_path):
            # Define the record belonging to the paper        
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
            
            # Download the archive to the local downloads folder.
            paper.download_pdf(dirpath=local_base_path, filename=f"{arxiv_id}.pdf")

            # Download the archive to the local downloads folder.
            paper.download_source(dirpath=local_base_path, filename=f"{arxiv_id}.tar.gz")

        text = extract_pdf_text(local_pdf_path)
        return text
       
    def identify_content(self, soup, verbose=False):

        # Get the page title for use with the queries
        page_title = soup.find('head').find('title').string

        # Find and extract relevant content from soup based on the title
        relevant_content = []

        for element in soup.find_all(True):
            if element.name in ["h1", "h2", "h3", "p"]:
                text = element.text.strip()
                if len(text) > 0:
                    relevant = self.classify_content(page_title, text)
                    if relevant:
                        relevant_content.append(text)                
                        if verbose: 
                            print(element)

        return relevant_content
             
    def classify_content( self, title:str, content:str) -> str: 
        from guidance import assistant, system, user, select

        with system():
            self.small_lm += "You are to classify web data as content or other (such as an adversitement) based on the page title.  Respond True if it is content, False if not."

        with user():
            self.small_lm += f"Title: `{title}`, Data: ```{content}```."

        with assistant():
            self.small_lm += select(["True", "False"], name="choice") 
            
        return eval(self.small_lm['choice'])
    
    def identify_metadata(self, soup, verbose=False):

        page_title = soup.find('head').find('title').string
        relevant_content = []
        for data in soup.find_all("meta"):
            relevant = False
            if 'content' in data.attrs and 'http' in data.attrs['content']:
                relevant = True
            elif 'property' in data.attrs:
                relevant = self.classify_metadata(data.attrs)
            elif 'name' in data.attrs:
                relevant = self.classify_metadata(data.attrs)
                
            if relevant:
                relevant_content.append(data.attrs)                
                if verbose: print(data.attrs['content'])

        return relevant_content

    def classify_metadata(self, content:str) -> str: 
        
        # Todo: Move this to the init
        from guidance import assistant, system, user, select

        with system():
            self.small_lm += "Help the user identify if the metadata contains potentially useful information such as: author, title, description, a date, etc. Respond True for useful, False for not."
            
        with user():
            self.small_lm += f"We are parsing html metadata to extract useful data.  Should we hold onto this item? {content}."

        with assistant():
            self.small_lm += select(["True", "False"], name="choice") 
            
        return eval(self.small_lm['choice'])
    
    def collect_images(self, soup, local_path, verbose=False):
        import os
        def get_basename(filename):
            return os.path.splitext(os.path.basename(filename))[0]

        protocol = get_protocol(soup.url)
        
        for img in soup.find_all('img'):

            relevant = False
            img_alt = img.attrs['alt'] if 'alt' in img.attrs else ""
            img_src = img.attrs['src'].lower()

            if 'png;base64' in img_src:
                from io import BytesIO
                from PIL import Image
                import base64

                # Step 1: Strip the prefix to get the Base64 data
                encoded_data = img.attrs['src'].split(",")[1]

                # Step 2: Decode the Base64 string
                # decoded_png = base64.b64decode(encoded_data)
                image_data = base64.b64decode(encoded_data)

                # Step 3: Create a BytesIO buffer from the decoded data
                image_buffer = BytesIO(image_data)

                # Step 4: Open the image using PIL
                image = Image.open(image_buffer)

                # Save the image to a file
                image.save(f"{img_src.replace('data:image/png;base64','')[:28]}.png")

            elif 'logo' in img_src:
                continue

            elif 'png' in img_src or 'jpg' in img_src or 'jpeg' in img_src or 'webp' in img_src or 'avif' in img_src or 'heif' in img_src or 'heic' in img_src or 'svg' in img_src:

                file_name = img_src.split("/")[-1] # there are other ways to do this
                local_image_description_path = os.path.join(local_path, get_basename(file_name) + ".txt")
                local_image_path = os.path.join(local_path, file_name)
                if len(img_alt) > 0 and not os.path.exists(local_image_description_path):
                    with open(local_image_description_path, 'w') as f:
                        f.write(img_alt) 
                if not os.path.exists(local_image_path):
                        
                    image_url = fix_missing_protocol(img.attrs['src'], soup.url)
                    try:
                        response = requests.get(image_url, params={'headers': self.request_kwargs})
                    except Exception:
                        print(image_url, protocol, img.attrs['src'])
                        traceback.print_exc()

                    if response.status_code == 200:
                        with open(local_image_path, 'wb') as f:
                            f.write(response.content)
