from pathlib import Path
import os,requests,markdown2
from bs4 import BeautifulSoup
from docutils import io, core
from PyPDF2 import PdfReader


### For Pdfs - str
####################
def PdfToStr():
    """ Takes User name and goes to that folder in DB and extracts Text from all the PDFs """
    
    extracted_text=""

    try:
        for pdf_file in os.listdir("saved docs"):
            doc_reader=PdfReader(os.path.join("saved docs",doc))
            for page in doc_reader.pages:
                extracted_text+= " " + page.extract_text()
        
    except Exception as e:
        extracted_text+=""
        print(f"Error: {e}") 

    return extracted_text

## PdfToStr()



### For txt - str
####################
def TxtToStr():
    extracted_text=""

    for doc in os.listdir("saved docs"):
        if doc.endswith(".txt"):
        
            try:
                with open(os.path.join("saved docs",doc), 'r', encoding='utf-8') as file:
                    all_text=file.read()
                    cleaned_text=" ".join(line.strip() for line in all_text.splitlines() if line.strip())  ## join line and remove empty line
                    extracted_text+=cleaned_text
                    
            except Exception as e:
                extracted_text+=""
                print(f"Error: {e}")
                    
    return extracted_text

## TxtToStr()


### For Urls - str
####################
def ScrapeUrlToStr(user_urls):
    extracted_text=""
    url_list=[url for url in user_urls.split(",")]

    for url in url_list:
        try:
            response = requests.get(url)
            response.raise_for_status() 

            soup = BeautifulSoup(response.content, 'html.parser')                                     ## Parse HTML content 
            all_text = soup.get_text(separator=' ')                                                   ##  text extraction from html tags
            cleaned_text = " ".join(line.strip() for line in all_text.splitlines() if line.strip())   ## join line and remove empty line
            extracted_text+=cleaned_text
        
        except Exception as e:
            print(f"Error: {e}")
            extracted_text+=""

    return extracted_text

# extracted_text=ScrapeUrlToStr('https://www.ibgroup.co.in/parivartan-next-gen,https://en.wikipedia.org/wiki/Main_Page')
# extracted_text



### For .rst - str
####################      
def RstToStr():
    extracted_text=""

    for doc in os.listdir("saved docs"):
        if doc.endswith(".rst"):
            try: 
                with open(os.path.join("saved docs",doc), 'r', encoding='utf-8') as rst_file:
                    rst_content = rst_file.read()

                    # Convert reStructuredText to HTML
                    settings={'output_encoding': 'unicode'}
                    parts=core.publish_parts(source=rst_content, writer_name='html', settings_overrides=settings)
                    html_content=parts['whole']

                    # Remove HTML tags to get plain text
                    soup=BeautifulSoup(html_content, 'html.parser')
                    all_text=soup.get_text(separator=' ')
                    cleaned_text=" ".join(line.strip() for line in all_text.splitlines() if line.strip())  ## join line and remove empty line
                    extracted_text+=cleaned_text
            
            except Exception as e:
                extracted_text+=""
                print(f"Error: {e}")
                    
    return extracted_text

# extracted_text=RstToStr()
# extracted_text



### For .md - str
####################
def MdToStr():
    extracted_text=""

    for doc in os.listdir("saved docs"):
        if doc.endswith(".md"):

            try:
                with open(os.path.join("saved docs",doc),'r',encoding='utf-8') as md_file:
                    md_content=md_file.read()
                    html_content=markdown2.markdown(md_content)   ## Convert Markdown to HTML

                    # Remove HTML tags to get plain text
                    soup=BeautifulSoup(html_content, 'html.parser')
                    txt_content = soup.get_text()
                    all_text=soup.get_text(separator=' ')
                    cleaned_text=" ".join(line.strip() for line in all_text.splitlines() if line.strip())  ## join line and remove empty line
                    extracted_text+=cleaned_text
            
            except Exception as e:
                extracted_text+=cleaned_text
                print(f"Error: {e}")
            
    return extracted_text
## Add code to remove icons

# extracted_text=MdToStr()
# extracted_text



## Driver function
###################
