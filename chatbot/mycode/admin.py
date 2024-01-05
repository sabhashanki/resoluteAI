import streamlit as st
import pickle
import utils
from pathlib import Path
import streamlit_authenticator as stauth
from st_pages import Page, show_pages, add_indentation
from dotenv import load_dotenv


load_dotenv()
# Reading login information
user_info = {}
cred_path = Path(__file__).parent / "./hashed_passwords.pkl"
with cred_path.open('rb') as file:
    user_info = pickle.load(file)

credentials = {
    'usernames' : {
        user_info['usernames'][0] : {
            'name' : user_info['names'][0],
            'password' : user_info['passwords'][0]
        }
    }
}

cookie_name = 'sample_app'
authenticator = stauth.Authenticate(credentials, cookie_name, 'abcd', cookie_expiry_days=60)

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
    show_pages([Page("admin.py", "Admin", "⚙️"),Page("chat.py", in_section=False, icon="📝"),])
    add_indentation()

    with st.sidebar:
        authenticator.logout("Logout", "sidebar")

    with st.container():
        pdf = st.file_uploader("Upload your PDF Document",type="pdf",accept_multiple_files=False)

    process_button=st.button("Process")
    if process_button:
        raw_text = utils.extract_pdf(pdf)
        utils.process_text(raw_text)


else:
    st.session_state.sidebar_state = 'collapsed'
    st.markdown(hide_bar, unsafe_allow_html=True)


