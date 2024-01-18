import configparser
import streamlit as st
from utils import login, add_company_logo, extract_pdf, process_text, hide_bar
from st_pages import Page, show_pages, add_indentation
from dotenv import load_dotenv

load_dotenv()
add_company_logo()
authentication_status, authenticator = login()

# set sidebar collapsed before login
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'collapsed'

# authentication module
if authentication_status == False:
    st.error("Username/password is incorrect")
    st.markdown(hide_bar, unsafe_allow_html=True)
elif authentication_status == None:
    st.info("Enter username and password")
    st.markdown(hide_bar, unsafe_allow_html=True)

if authentication_status:
    st.session_state['logout'] = False
    show_pages([Page("admin.py", "Admin", "⚙️"),Page("settings.py", "Settings", "⚙️"), Page("chat.py", in_section=False, icon="📝"),])
    add_indentation()

    with st.sidebar:
        authenticator.logout("Logout", "sidebar")
    with st.container():
        pdf_folder = st.file_uploader("Upload your PDF Document",type="pdf",accept_multiple_files=True)

    process_button=st.button("Proceed")
    if process_button:
        raw_text = extract_pdf(pdf_folder)
        process_text(raw_text)

else:
    st.session_state.sidebar_state = 'collapsed'
    st.markdown(hide_bar, unsafe_allow_html=True)


