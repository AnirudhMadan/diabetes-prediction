import streamlit as st
from database import create_user_table, add_user, authenticate_user

def login_register_ui():
    st.title("🔐 User Authentication")
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login to your account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.success(f"Welcome {username} 👋")
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
            else:
                st.error("Invalid credentials")

    elif choice == "Register":
        st.subheader("Create a new account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Register"):
            success = add_user(new_user, new_pass)
            if success:
                st.success("Account created successfully! Login now.")
            else:
                st.warning("⚠️ Username already taken. Please choose another.")

def is_authenticated():
    return st.session_state.get("authenticated", False)

def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
