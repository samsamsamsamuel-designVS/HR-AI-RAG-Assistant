"""Username/password login helpers for the HR assistant.

Credentials are read from [passwords] in .streamlit/secrets.toml:
each key is a username and its value is that user's password.
"""
import hmac
import streamlit as st


def configured_passwords() -> dict[str, str]:
    """Return valid credentials without displaying or logging their values."""
    try:
        passwords = st.secrets.get("passwords", {})
    except FileNotFoundError:
        return {}
    return {name: password for name, password in passwords.items()
            if isinstance(name, str) and name and isinstance(password, str) and password}


def check_credentials() -> None:
    """Form callback: check the submitted credentials before the page reruns."""
    username = st.session_state.get("login_username", "").strip()
    password = st.session_state.get("login_password", "")
    expected = configured_passwords().get(username)
    # Compare bytes using a constant-time comparison. Unknown users get the
    # same error message as incorrect passwords.
    valid = expected is not None and hmac.compare_digest(password.encode(), expected.encode())
    if valid:
        st.session_state["authenticated_user"] = username
        st.session_state["login_error"] = False
    else:
        st.session_state["login_error"] = True
    # Do not retain the entered password in session state after submission.
    st.session_state.pop("login_password", None)


def require_login() -> None:
    """Stop the app at the login form until credentials have been accepted."""
    if st.session_state.get("authenticated_user"):
        return

    with st.sidebar:
        st.header("Account", divider="orange")
        st.caption("Please log in to continue.")
    st.title("HR AI Employment Assistant 🗪")
    st.markdown("Log in to ask questions, upload new PDF knowledge, and explore your documents.")
    ready = bool(configured_passwords())
    # A form submits both inputs together. The callback runs before the next
    # page rerun, which then checks authenticated_user above.
    with st.form("login_form"):
        st.subheader("Log in")
        st.text_input("Username", key="login_username", disabled=not ready)
        st.text_input("Password", type="password", key="login_password", disabled=not ready)
        st.form_submit_button("Log in", icon=":material/login:", type="primary",
                              key="login", disabled=not ready, on_click=check_credentials)
    if not ready:
        st.info("Login credentials are not configured. Please contact the app administrator.")
    elif st.session_state.get("login_error"):
        st.error("Incorrect username or password.")
    # Navigation, document uploads, and AI requests must only run after login.
    st.stop()


def account_menu() -> None:
    """Display the username and clear the user's entire session on logout."""
    with st.sidebar:
        st.header("Login As", divider="orange")
        st.text(st.session_state.get("authenticated_user", "Signed-in user"))
        if st.button("Log out", icon=":material/logout:", key="logout", width="stretch"):
            # Clear this browser session's login, chat, and connection reference.
            # Saved PDFs and Chroma embeddings remain on disk for future visits.
            st.session_state.clear()
            st.rerun()
"""Username/password login helpers for the HR assistant.

Credentials are read from [passwords] in .streamlit/secrets.toml:
each key is a username and its value is that user's password.
"""
import hmac
import streamlit as st


def configured_passwords() -> dict[str, str]:
    """Return valid credentials without displaying or logging their values."""
    try:
        passwords = st.secrets.get("passwords", {})
    except FileNotFoundError:
        return {}
    return {name: password for name, password in passwords.items()
            if isinstance(name, str) and name and isinstance(password, str) and password}


def check_credentials() -> None:
    """Form callback: check the submitted credentials before the page reruns."""
    username = st.session_state.get("login_username", "").strip()
    password = st.session_state.get("login_password", "")
    expected = configured_passwords().get(username)
    # Compare bytes using a constant-time comparison. Unknown users get the
    # same error message as incorrect passwords.
    valid = expected is not None and hmac.compare_digest(password.encode(), expected.encode())
    if valid:
        st.session_state["authenticated_user"] = username
        st.session_state["login_error"] = False
    else:
        st.session_state["login_error"] = True
    # Do not retain the entered password in session state after submission.
    st.session_state.pop("login_password", None)


def require_login() -> None:
    """Stop the app at the login form until credentials have been accepted."""
    if st.session_state.get("authenticated_user"):
        return

    with st.sidebar:
        st.header("Account", divider="orange")
        st.caption("Please log in to continue.")
    st.title("HR AI Employment Assistant 🗪")
    st.markdown("Log in to ask questions, upload new PDF knowledge, and explore your documents.")
    ready = bool(configured_passwords())
    # A form submits both inputs together. The callback runs before the next
    # page rerun, which then checks authenticated_user above.
    with st.form("login_form"):
        st.subheader("Log in")
        st.text_input("Username", key="login_username", disabled=not ready)
        st.text_input("Password", type="password", key="login_password", disabled=not ready)
        st.form_submit_button("Log in", icon=":material/login:", type="primary",
                              key="login", disabled=not ready, on_click=check_credentials)
    if not ready:
        st.info("Login credentials are not configured. Please contact the app administrator.")
    elif st.session_state.get("login_error"):
        st.error("Incorrect username or password.")
    # Navigation, document uploads, and AI requests must only run after login.
    st.stop()


def account_menu() -> None:
    """Display the username and clear the user's entire session on logout."""
    with st.sidebar:
        st.header("Login As", divider="orange")
        st.text(st.session_state.get("authenticated_user", "Signed-in user"))
        if st.button("Log out", icon=":material/logout:", key="logout", width="stretch"):
            # Clear this browser session's login, chat, and connection reference.
            # Saved PDFs and Chroma embeddings remain on disk for future visits.
            st.session_state.clear()
            st.rerun()
