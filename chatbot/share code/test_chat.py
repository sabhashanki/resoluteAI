import streamlit as st

# def main():

#     st.title("Chatbot App")
#     st.write("🤖 Chatbot: Hello! I'm your friendly chatbot. What's your name?")
    
#     user_name = st.text_input("Your Name:")

#     if user_name:
#         st.text(f"🤖 Chatbot: Nice to meet you, {user_name}!")

# if __name__ == "__main__":
#     main()

def main():
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
        user_budget=st.selectbox('Select a budget :- ', ("",'25 L', 'above 25L'))
        st.write("")
        if user_budget=="25 L":
            st.write(f"🤖  {user_name}, You have selected '25L plan' , Loading data for 'Parivartan' Plan . ")
        if user_budget=="above 25 L":
            st.write(f"🤖  {user_name}, You have selected 'above 25L plan' , Loading data for 'Parivartan Gen Nxt' Plan . ")
        st.write("")

        

if __name__ == "__main__":
    main()

#user_name,user_phone="",""
# def main():
#         st.write("Hello! I am a chatbot.")
    
    
#     # with st.chat_message("assistant"):
#     #     st.write("What is your name?")

#     # user_name=st.chat_input("Please, enter your Name here ")
#     # if user_name:
#     #     with st.chat_message("assistant"):
#     #         st.write(f"hii {user_name} , please enter your phone number.")

#     #     user_phone=st.chat_input("Please, enter your phone number ")
#     #     if user_phone:
#     #         with st.chat_message("assistant"):
#     #             st.write(f"{user_name}, what is your budget ?")

#     with st.chat_message("assistant"):
#         st.write("What is your name?")

#         user_name=st.text_input("",placeholder="Please, enter your Name here ")
    
#         if user_name!=None:
#             with st.chat_message("assistant"):
#                 st.write(f"hii {user_name} , please enter your phone number.")
    
#     # if user_name:
#     #     with st.chat_message("assistant"):
#     #         st.write(f"hii {user_name} , please enter your phone number.")

#         # user_phone=st.chat_input("Please, enter your phone number ")
#         # if user_phone:
#         #     with st.chat_message("assistant"):
#         #         st.write(f"{user_name}, what is your budget ?")

            

#     # if user_name:
#     #     with st.chat_message("assistant"):
#     #         st.write(f"hii {user_name}, please enter your phone number")
#     # #if user_phone:
#     # user_phone=st.chat_input("Please, enter your phone number ")

#     #st.chat_message("user").markdown(f"Hii , {user_name}") 
        
#         #if user_phone:
#             # with st.chat_message("assistant"):
#             #     st.write(f"hii {user_name}, what is your budget")
#             # user_budget=st.selectbox('Select a budget :- ', ('25 L', 'above 25L'))
#             # if user_budget=="25 L":
#             #     with st.chat_message("assistant"):
#             #         st.write(f"{user_name}, You have selected '25 L' plan . ")
#             # if user_budget=="above 25 L":
#             #     with st.chat_message("assistant"):
#             #         st.write(f"{user_name}, You have selected 'above 25 L' plan . ")





        

    #st.write(user_name)
    
    
    # for ask_index in range(len(ask_list)):
    #     #st.write(ask_list[ask_index])
    #     if ask_index==0:
    #        response=st.chat_input("Please, Your Name here ")
    
    # with st.chat_message("assistant"):
    #     st.write("What is your name?")
    #     user_name = st.text_input("")

    # if user_name:
    #     st.text(f"🤖 Chatbot: Nice to meet you, {user_name}!")

# if __name__ == "__main__":
#     main()