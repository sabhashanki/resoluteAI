from streamlit_extras.app_logo import add_logo 
import streamlit as st

import streamlit as st
from streamlit_extras.app_logo import add_logo

# add kitten logo
add_logo("images/resoluteai_logo_smol.jpg")

# add sidebar buttons
st.sidebar.button("Button")
st.sidebar.button("Button 2")

# add sidebar filters
st.sidebar.slider("Slider", 0, 100, 50)
st.sidebar.date_input("Date Input")


# # Adds company logo at the top of sidebar
# def add_company_logo():
#     add_logo('images/resoluteai_logo_smol.jpg', height=80)
#     st.markdown(
#             """
#             <style>
#                 [data-testid="stSidebarNav"] {
#                     padding-top: 1rem;
#                     background-position: 10px 10px;
#                 }
#                 [data-testid="stSidebarNav"]::before {
#                     content: "My Company Name";
#                     margin-left: 0px;
#                     margin-top: 0px;
#                     font-size: 1px;
#                     position: relative;
#                     top: 1px;
#                 }
#             </style>
#             """,
#             unsafe_allow_html=True,
#     )
    
#     st.markdown(
#         """
#         <style>
#             .css-1y4p8pa {
#                 padding-top: 0rem;
#                 max-width: 50rem;
#             }
#         </style>
#         """,
#             unsafe_allow_html=True,
#         )
     
    
# def set_sidebar_state():
#     # set sidebar collapsed before login
#     if 'sidebar_state' not in st.session_state:
#         st.session_state.sidebar_state = 'collapsed'

#     # hide collapsed control button
#     hide_bar = """
#             <style>
#             [data-testid="collapsedControl"] {visibility:hidden;}
#             </style>
#             """
#     st.session_state.sidebar_state = 'collapsed'
#     st.markdown(hide_bar, unsafe_allow_html=True)
    

# hide_bar = '''
# <style>
#     [data-testid="stSidebar"] {
#         display: none;
#     }
# </style>
# '''

# add_company_logo()