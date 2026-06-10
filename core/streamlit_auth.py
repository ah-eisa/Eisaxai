"""
streamlit_auth.py — Lightweight Streamlit auth for EisaX
Uses existing user_db + bcrypt — no JWT needed for Streamlit sessions.
"""
import streamlit as st
from core.user_db import (
    init_users_table, get_user_by_email, list_users,
    create_user, record_login, update_user,
)
from core.auth import verify_password, hash_password


_DEFAULT_ADMIN_EMAIL = "admin@eisax.com"
_DEFAULT_ADMIN_PASS  = "eisax2024"

_AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Tajawal:wght@400;500;700&display=swap');
[data-testid="stAppViewContainer"] { background: #0a0f1e !important; }
[data-testid="stAppViewContainer"] > .main { background: transparent; }
[data-testid="stSidebar"] { display: none !important; }
.stApp { font-family: 'Inter', 'Tajawal', sans-serif; }
.auth-logo { text-align:center; font-size:3rem; font-weight:900; letter-spacing:3px; background:linear-gradient(135deg,#38bdf8,#0ea5a4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.auth-tagline { text-align:center; color:#475569; font-size:.75rem; text-transform:uppercase; letter-spacing:.2em; margin-bottom:1.5rem; }
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stTextInput"]) { background:rgba(15,23,42,0.92); backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px); border:1px solid rgba(99,179,237,0.12); border-radius:24px; padding:2.5rem 2rem; box-shadow:0 25px 50px rgba(0,0,0,0.5); }
</style>
"""


def _ensure_seed_admin():
    """Create a default admin user if the users table is empty."""
    init_users_table()
    if not list_users():
        pw_hash = hash_password(_DEFAULT_ADMIN_PASS)
        create_user(
            email=_DEFAULT_ADMIN_EMAIL,
            name="Admin",
            password_hash=pw_hash,
            role="admin",
            must_change_pw=False,
        )


def _do_login(email: str, password: str) -> str | None:
    """Validate credentials. Returns None on success, error string on failure."""
    user = get_user_by_email(email)
    if not user:
        return "البريد الإلكتروني أو كلمة المرور غير صحيحة"
    if not user["is_active"]:
        return "الحساب غير مفعّل — تواصل مع المسؤول"
    if not verify_password(password, user["password_hash"]):
        return "البريد الإلكتروني أو كلمة المرور غير صحيحة"

    record_login(user["id"])
    st.session_state["user"] = {
        "id":            user["id"],
        "email":         user["email"],
        "name":          user["name"],
        "role":          user["role"],
        "must_change_pw": bool(user["must_change_pw"]),
    }
    return None


def _show_login_page():
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="auth-logo">EisaX</div><div class="auth-tagline">Arab Markets Intelligence</div>', unsafe_allow_html=True)
        email    = st.text_input('', placeholder='البريد الإلكتروني / Email', label_visibility='collapsed')
        password = st.text_input('', placeholder='كلمة المرور / Password', type='password', label_visibility='collapsed')
        if st.button('دخول / Sign in', type='primary', use_container_width=True):
            if not email or not password:
                st.error('أدخل البريد الإلكتروني وكلمة المرور')
            else:
                err = _do_login(email.strip(), password)
                if err:
                    st.error(err)
                else:
                    st.rerun()


def _show_change_password_page():
    """Force password change screen — shown when must_change_pw=True."""
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<div class="auth-card"><div class="auth-logo">EisaX</div>'
                '<div class="auth-sub">يجب تغيير كلمة المرور / Password Change Required</div></div>',
                unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.warning("⚠️ يجب عليك تغيير كلمة المرور قبل المتابعة")
        new_pw  = st.text_input("كلمة المرور الجديدة / New Password", type="password")
        conf_pw = st.text_input("تأكيد كلمة المرور / Confirm Password", type="password")

        if st.button("تغيير وحفظ / Change & Continue", type="primary", use_container_width=True):
            if len(new_pw) < 8:
                st.error("كلمة المرور يجب أن تكون 8 أحرف على الأقل")
            elif new_pw != conf_pw:
                st.error("كلمتا المرور غير متطابقتين")
            else:
                uid = st.session_state["user"]["id"]
                update_user(uid, password_hash=hash_password(new_pw), must_change_pw=0)
                st.session_state["user"]["must_change_pw"] = False
                st.success("✅ تم تغيير كلمة المرور")
                st.rerun()

        if st.button("🚪 تسجيل الخروج / Logout", use_container_width=True):
            logout()


def require_auth() -> dict:
    """
    Call once near the top of your Streamlit app (after set_page_config).
    - If not logged in → shows login page + st.stop()
    - If must_change_pw → shows change-password page + st.stop()
    Returns current user dict: {id, email, name, role, must_change_pw}
    """
    _ensure_seed_admin()

    if st.session_state.get("user") is None:
        _show_login_page()
        st.stop()

    if st.session_state["user"].get("must_change_pw"):
        _show_change_password_page()
        st.stop()

    return st.session_state["user"]


def logout():
    """Clear session and rerun to show login page."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def show_user_badge():
    """Sidebar: styled user card + logout button."""
    user = st.session_state.get("user", {})
    if not user:
        return
    initials = "".join(w[0].upper() for w in user.get("name", "U").split()[:2])
    role_label = "Admin" if user.get("role") == "admin" else "User"
    st.sidebar.markdown(
        f'<div class="user-badge-card">'
        f'  <div class="user-badge-avatar">{initials}</div>'
        f'  <div class="user-badge-info">'
        f'    <div class="user-badge-name">{user["name"]} <span style="font-size:.65rem;opacity:.6">({role_label})</span></div>'
        f'    <div class="user-badge-email">{user["email"]}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🚪 خروج / Logout", use_container_width=True):
        logout()
