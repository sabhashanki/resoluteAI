import streamlit as st
from pathlib import Path
import pickle,os,utils
import streamlit_authenticator as stauth
#from admin import admin_page
from PIL import Image
from st_pages import Page, show_pages, add_page_title, Section, add_indentation

from extractor import scrape_and_create_embeddings_1,scrape_and_create_embeddings_2


# get user information
user_info = {}
cred_path = Path(__file__).parent / "./hashed_passwords.pkl"
with cred_path.open("rb") as file:
    user_info = pickle.load(file)
    
credentials = {
    "usernames":{
        user_info["usernames"][0] : {
            "name" : user_info["names"][0],
            "password" : user_info["passwords"][0]
            }         
        }
}


cookie_name = "sample_app"
authenticator = stauth.Authenticate(credentials, cookie_name, "abcd", cookie_expiry_days=30)

st.session_state['authenticator'] = authenticator
st.session_state['cookie_name'] = cookie_name

name, authentication_status, username = authenticator.login("Login", "main")

def logout():
    authenticator.cookie_manager.delete(cookie_name)
    st.session_state['logout'] = True
    st.session_state['name'] = None
    st.session_state['username'] = None
    st.session_state['authentication_status'] = None

    
# set sidebar collapsed before login
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'collapsed'


           
hide_bar = '''
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
</style>
'''


if authentication_status == False:
    st.error("Username/password is incorrect")
    st.markdown(hide_bar, unsafe_allow_html=True)
elif authentication_status == None:
    st.warning("Please enter your username and password")
    st.markdown(hide_bar, unsafe_allow_html=True)

if authentication_status:
    # Adds company logo at the top of sidebar
    utils.add_company_logo()
    
    # set logout session state to false 
    st.session_state['logout'] = False
    
    from st_pages import Page, Section, add_page_title, show_pages
    show_pages([Page("Admin.py", "Admin", "⚙️"),Page("chat.py", in_section=False, icon="📝"),])
    add_indentation()
    
    #admin_page()
    #st.markdown("<h1 style='text-align: center; color: black;'>GenAI led ChatBot</h1>", unsafe_allow_html=True)
    ## Columns !
    with st.container():
        col1,col2,col3=st.columns([7,2,7])
        
        with col1:
            with st.container():
                #st.title("Enter documents for Parivartan")
                url_input_1 = st.text_input('Enter Parivartan url') 
                pdfs_folder_1 = st.file_uploader("Upload PDF files for Parivartan",type="pdf",accept_multiple_files=True)

                if pdfs_folder_1 is not None:
                    for pdf in pdfs_folder_1:
                        #st.write(pdf.name)

                        # Save the PDF file to the specified folder
                        pdf_path_1=os.path.join("parivartan_stored_pdfs",pdf.name)
                        with open(pdf_path_1,"wb") as f:
                            f.write(pdf.getbuffer())
                st.write("")
                

            with col3:
                with st.container():
                    #st.title("Enter documents for Parivartan_gen_nxt")
                    url_input_2 = st.text_input('Enter Parivartan gen nxt url') 
                    pdfs_folder_2 = st.file_uploader("Upload PDF files for Parivartan_gen_nxt",type="pdf",accept_multiple_files=True)

                    if pdfs_folder_2 is not None:
                        for pdf in pdfs_folder_2:
                            #st.write(pdf.name)

                            # Save the PDF file to the specified folder
                            pdf_path_2=os.path.join("parivartan_gen_nxt_stored_pdfs",pdf.name)
                            with open(pdf_path_2,"wb") as f:
                                f.write(pdf.getbuffer())
                    st.write("")

    with st.container():
        st.write("")
        st.write("")
        # col11,col12,col13=st.columns([3,3,.5])    
        # with col12:

        process_button=st.button("Process")
            
        if process_button:
            if url_input_1!=None and url_input_2!=None:
                st.write('searching for website',url_input_1)
                scrape_and_create_embeddings_1(url=url_input_1)
                st.write('searching for website',url_input_2)
                scrape_and_create_embeddings_2(url=url_input_2)

    
    
    with st.sidebar:
        authenticator.logout("Logout", "sidebar")

else:
    st.session_state.sidebar_state = 'collapsed'
    st.markdown(hide_bar, unsafe_allow_html=True)
