import streamlit as st
import secrets
from auth import login_register_ui, is_authenticated, logout
from database import create_user_table
import webbrowser
import os

TOKEN_FILE = "auth_token.txt"

def main():
    st.set_page_config(page_title="Login Portal", page_icon="🔐")
    create_user_table()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not is_authenticated():
        login_register_ui()
    else:
        st.success(f"✅ Welcome {st.session_state['username']} 🎉")

        if "auth_token" not in st.session_state:
            # Generate a random secure token
            st.session_state["auth_token"] = secrets.token_urlsafe(16)

            # Save token to a file so Flask can read it
            with open(TOKEN_FILE, "w") as f:
                f.write(st.session_state["auth_token"])

        flask_url = f"http://127.0.0.1:5000/?auth_token={st.session_state['auth_token']}"

        if st.button("Launch Diabetes Predictor App"):
            webbrowser.open_new_tab(flask_url)

        if st.button("Logout"):
            logout()
            st.session_state.pop("auth_token", None)
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
            st.rerun()

if __name__ == "__main__":
    main()
