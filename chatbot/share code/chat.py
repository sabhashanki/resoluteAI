import streamlit as st
from pathlib import Path
import pickle,os,utils
import streamlit_authenticator as stauth

#from admin import admin_page
from PIL import Image
from st_pages import Page, show_pages, add_page_title, Section, add_indentation
from extractor import load_and_answer_questions
# api="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"#"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"
# os.environ["OPENAI_API_KEY"]=api

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
    
    selected_plan=None
    stored_embd=None

    st.write("🤖  Hello! I am a chatbot . ")
    st.write("")

    st.write("🤖  What is your name ?")
    user_name=st.text_input("",label_visibility="collapsed",placeholder="Please, enter your Name here ")
    st.write("")

    if user_name:
        st.write(f"🤖  hii {user_name}, please enter your phone number ")
        user_phone=st.text_input("",label_visibility="collapsed",placeholder="Please, enter your phone number here ")
    st.write("")

    if user_name and user_phone:
        st.write(f"🤖  {user_name}, what is your budget ?")
        user_budget=st.selectbox('Select a budget :- ', ("",'25L', 'above 25L'))
        st.write("")
        if user_budget=="25L":
            selected_plan="25L"
            st.write(f"🤖  {user_name}, You have selected '25L plan' , Loading data for 'Parivartan' Plan . ")

        if user_budget=="above 25L":
            selected_plan="above 25L"
            st.write(f"🤖  {user_name}, You have selected 'above 25L plan' , Loading data for 'Parivartan Gen Nxt' Plan . ")
        st.write("") 


    if selected_plan=="25L":
        stored_embd="parivartan_stored_embd"
        st.success("Loaded Parivartan Embedds")
    if selected_plan=="above 25L":
        stored_embd="parivartan_gen_nxt_stored_embd"
        st.success("Loaded Parivartan Gen Nxt Embedds")

    if selected_plan!=None:  
        if len(os.listdir(stored_embd))==0:
            st.warning("No Embeddings found please give url ")
        if "messages" not in st.session_state:
            st.session_state.messages = []


        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Any question ?"): 
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt) 
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt}) 


        if prompt!=None:
            print("@@@@ Question",prompt)
            answer_text = load_and_answer_questions(prompt,stored_embd)
            if answer_text !=None:
                bot_response=answer_text
            else:
                bot_response="Apologies! The information you have requested is not available at this point"
            response = f"Bot: {bot_response}"


            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response) 
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        

    	
    with st.sidebar:
        authenticator.logout("Logout", "sidebar")
        clear_button =st.button("Clear", type="primary")
        if clear_button:
            try:
                #if len(os.listdir("stored_embd"))!=0:
                    # os.remove("stored_embd/index.pkl")
                    # os.remove("stored_embd/index.faiss")
                    # os.remove("saved_text.txt")
                    
                os.remove("parivartan_gen_nxt_stored_embd/index.pkl")
                os.remove("parivartan_gen_nxt_stored_embd/index.faiss")
                os.remove("parivartan_gen_nxt.txt")

                os.remove("parivartan_stored_embd/index.pkl")
                os.remove("parivartan_stored_embd/index.faiss")
                os.remove("parivartan.txt")

                if len(os.listdir("parivartan_gen_nxt_stored_pdfs"))>0:
                    for doc in os.listdir("parivartan_gen_nxt_stored_pdfs"):
                        os.remove(f"parivartan_gen_nxt_stored_pdfs/{doc}")

                if len(os.listdir("parivartan_stored_pdfs"))>0:
                    for doc in os.listdir("parivartan_stored_pdfs"):
                        os.remove(f"parivartan_stored_pdfs/{doc}")
            except Exception as e:
                print(e)
            st.session_state.messages = []
else:
    st.session_state.sidebar_state = 'collapsed'
    st.markdown(hide_bar, unsafe_allow_html=True)

