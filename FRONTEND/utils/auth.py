# -*- coding: utf-8 -*-
"""
Authentication page - Login / Register
"""
import streamlit as st
import requests
import os

_BACKEND = os.getenv("BACKEND_URL", "http://localhost:3000/api")

# ── CSS ────────────────────────────────────────────────────────────────────
AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, body { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #0f172a 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stToolbarActions"], .stDeployButton { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }

.auth-bg {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(139,92,246,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(16,185,129,0.05) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}
.auth-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 40px 44px 44px;
    backdrop-filter: blur(20px);
    box-shadow: 0 25px 60px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    max-width: 480px; margin: 0 auto;
}
.auth-logo {
    text-align: center; margin-bottom: 32px;
}
.auth-logo-icon {
    width: 64px; height: 64px; border-radius: 18px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 2rem; margin-bottom: 14px;
    box-shadow: 0 8px 32px rgba(99,102,241,0.5);
}
.auth-title {
    font-size: 1.8rem; font-weight: 800; text-align: center;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.auth-subtitle {
    text-align: center; font-size: 0.88rem; color: #94a3b8; margin-bottom: 28px;
}
.auth-tab-wrap {
    display: flex; gap: 4px; background: rgba(255,255,255,0.06);
    border-radius: 12px; padding: 4px; margin-bottom: 28px;
}
.auth-tab {
    flex: 1; padding: 10px; border-radius: 9px; text-align: center;
    font-size: 0.88rem; font-weight: 600; cursor: pointer;
    color: #94a3b8; transition: all 0.2s;
}
.auth-tab.active {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff; box-shadow: 0 4px 12px rgba(99,102,241,0.4);
}
.field-label {
    font-size: 0.82rem; font-weight: 600; color: #cbd5e1;
    margin-bottom: 6px; margin-top: 16px;
}
.auth-footer {
    text-align: center; margin-top: 24px;
    font-size: 0.78rem; color: #64748b;
}
.success-box {
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
    border-radius: 12px; padding: 14px 18px; color: #6ee7b7;
    font-size: 0.88rem; margin: 12px 0;
}
.error-box {
    background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px; padding: 14px 18px; color: #fca5a5;
    font-size: 0.88rem; margin: 12px 0;
}
.user-badge {
    display: flex; align-items: center; gap: 12px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px; padding: 14px 18px; margin-bottom: 20px;
}
.user-avatar {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; font-weight: 700; color: white;
    flex-shrink: 0;
}
.user-name { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; }
.user-role { font-size: 0.75rem; color: #818cf8; font-weight: 500; }
.role-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
}
.role-admin { background: rgba(249,115,22,0.2); color: #fb923c; border: 1px solid rgba(249,115,22,0.3); }
.role-user  { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.role-viewer{ background: rgba(100,116,139,0.2); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

/* Streamlit input overrides */
[data-testid="stTextInput"] > div > div > input,
[data-testid="stTextInput"] input {
    background: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important; color: #f1f5f9 !important;
    font-size: 0.9rem !important; padding: 10px 14px !important;
    caret-color: #818cf8 !important;
}
[data-testid="stTextInput"] > div > div > input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #475569 !important; }
/* Label text — covers all Streamlit versions */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label p,
[data-testid="stTextInput"] label span,
[data-testid="stTextInput"] > label {
    color: #cbd5e1 !important; font-size: 0.82rem !important; font-weight: 600 !important;
}
/* Radio tabs (Login / Register) */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span {
    color: #e2e8f0 !important; font-size: 0.9rem !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: #e2e8f0 !important; }
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 12px 20px !important; font-size: 0.95rem !important;
    font-weight: 700 !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.5) !important;
}
</style>
"""


def _post(endpoint: str, body: dict) -> dict:
    try:
        r = requests.post(f"{_BACKEND}/auth/{endpoint}", json=body, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _get_me(token: str) -> dict:
    try:
        r = requests.get(
            f"{_BACKEND}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _change_password(token: str, current_pw: str, new_pw: str) -> dict:
    try:
        r = requests.post(
            f"{_BACKEND}/auth/me/change-password",
            json={"currentPassword": current_pw, "newPassword": new_pw},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _update_profile(token: str, full_name: str) -> dict:
    try:
        r = requests.patch(
            f"{_BACKEND}/auth/me",
            json={"fullName": full_name},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Render auth card ───────────────────────────────────────────────────────
def render_auth_page():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("""
        <div class="auth-card">
          <div class="auth-logo">
            <div class="auth-logo-icon">🧬</div>
            <div class="auth-title">PharmaLink GCN</div>
            <div class="auth-subtitle">Drug-Disease Prediction Platform</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab = st.radio("", ["🔑  Đăng nhập", "📝  Đăng ký"], horizontal=True,
                       label_visibility="collapsed")

        if tab == "🔑  Đăng nhập":
            _render_login()
        else:
            _render_register()


def _render_login():
    identifier = st.text_input("Email hoặc Tên đăng nhập", placeholder="example@email.com")
    password   = st.text_input("Mật khẩu", type="password", placeholder="••••••••")

    if st.button("Đăng nhập", use_container_width=True):
        if not identifier or not password:
            st.error("Vui lòng nhập đầy đủ thông tin")
            return
        with st.spinner("Đang xác thực..."):
            res = _post("login", {"identifier": identifier, "password": password})
        if "error" in res:
            st.error(f"Lỗi kết nối: {res['error']}")
        elif "message" in res and "access_token" not in res:
            st.error(res.get("message", "Đăng nhập thất bại"))
        elif "access_token" in res:
            st.session_state["auth_token"]   = res["access_token"]
            st.session_state["auth_user"]    = res["user"]
            st.session_state["is_logged_in"] = True
            st.switch_page("home.py")
        else:
            st.error("Đăng nhập thất bại. Kiểm tra lại thông tin.")

    st.markdown('<div class="auth-footer">Chưa có tài khoản? Chọn tab <b>Đăng ký</b></div>',
                unsafe_allow_html=True)


def _render_register():
    email     = st.text_input("Email *", placeholder="example@email.com")
    username  = st.text_input("Tên đăng nhập *", placeholder="Chỉ dùng chữ, số, dấu _")
    full_name = st.text_input("Họ tên (tùy chọn)", placeholder="Nguyễn Văn A")
    password  = st.text_input("Mật khẩu *", type="password", placeholder="Tối thiểu 8 ký tự")
    confirm   = st.text_input("Xác nhận mật khẩu *", type="password", placeholder="Nhập lại mật khẩu")

    if st.button("Tạo tài khoản", use_container_width=True):
        if not email or not username or not password:
            st.error("Vui lòng điền các trường bắt buộc (*)")
            return
        if password != confirm:
            st.error("Mật khẩu xác nhận không khớp")
            return
        if len(password) < 8:
            st.error("Mật khẩu cần ít nhất 8 ký tự")
            return

        with st.spinner("Đang tạo tài khoản..."):
            body = {"email": email, "username": username, "password": password}
            if full_name:
                body["fullName"] = full_name
            res = _post("register", body)

        if "error" in res:
            st.error(f"Lỗi kết nối: {res['error']}")
        elif "access_token" in res:
            st.session_state["auth_token"]   = res["access_token"]
            st.session_state["auth_user"]    = res["user"]
            st.session_state["is_logged_in"] = True
            st.switch_page("home.py")
        else:
            msg = res.get("message", "")
            if isinstance(msg, list):
                msg = "; ".join(msg)
            st.error(f"Đăng ký thất bại: {msg}")

    st.markdown('<div class="auth-footer">Đã có tài khoản? Chọn tab <b>Đăng nhập</b></div>',
                unsafe_allow_html=True)


# ── Sidebar user card ──────────────────────────────────────────────────────
def render_user_sidebar():
    user = st.session_state.get("auth_user", {})
    if not user:
        return

    role     = user.get("role", "user")
    name     = user.get("fullName") or user.get("username", "?")
    initials = "".join(w[0].upper() for w in name.split()[:2]) if name else "?"
    role_cls = f"role-{role}"
    role_lbl = {"admin": "Admin", "user": "Thành viên", "viewer": "Xem"}.get(role, role)

    # Role-specific accent colours
    accent = {"admin": "#f97316", "user": "#818cf8", "viewer": "#94a3b8"}.get(role, "#818cf8")

    st.markdown(f"""
    <div class="user-badge">
      <div class="user-avatar">{initials}</div>
      <div>
        <div class="user-name">{name}</div>
        <span class="role-badge {role_cls}">{role_lbl}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Role capabilities quick summary
    perms_map = {
        "admin":  ["✅ Toàn quyền hệ thống", "✅ Quản lý người dùng", "✅ Chạy dự đoán"],
        "user":   ["✅ Chạy dự đoán", "✅ Lưu lịch sử", "✅ Sinh phân tử"],
        "viewer": ["👁️ Chỉ xem kết quả", "❌ Không chạy dự đoán", "❌ Không lưu"],
    }
    perms_list = perms_map.get(role, [])
    perms_html = "".join(f'<div style="font-size:0.72rem;color:#94a3b8;padding:1px 0">{p}</div>' for p in perms_list)
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);
                border-radius:8px;padding:8px 12px;margin:8px 0 4px;">
        {perms_html}
    </div>
    """, unsafe_allow_html=True)

    if role == "admin":
        st.page_link("pages/6_admin_users.py", label="🛡️  Quản lý tài khoản")

    if st.button("⚙️ Hồ sơ & Cài đặt", use_container_width=True):
        st.session_state["show_profile"] = True
        st.rerun()
    if st.button("🚪 Đăng xuất", use_container_width=True):
        for k in ["auth_token", "auth_user", "is_logged_in", "show_profile"]:
            st.session_state.pop(k, None)
        st.rerun()


# ── Profile modal ──────────────────────────────────────────────────────────
def render_profile_modal():
    if not st.session_state.get("show_profile"):
        return

    user  = st.session_state.get("auth_user", {})
    token = st.session_state.get("auth_token", "")

    st.markdown("---")
    st.markdown("### ⚙️ Hồ sơ cá nhân")

    with st.expander("📋 Thông tin tài khoản", expanded=True):
        col1, col2 = st.columns(2)
        col1.text_input("Email", value=user.get("email", ""), disabled=True)
        col2.text_input("Tên đăng nhập", value=user.get("username", ""), disabled=True)

        new_name = st.text_input("Họ tên", value=user.get("fullName", "") or "")
        if st.button("💾 Lưu thông tin", use_container_width=True):
            res = _update_profile(token, new_name)
            if "error" not in res and "id" in res:
                st.session_state["auth_user"] = res
                st.success("Cập nhật thành công!")
            else:
                st.error("Cập nhật thất bại")

    with st.expander("🔒 Đổi mật khẩu"):
        cur_pw  = st.text_input("Mật khẩu hiện tại", type="password", key="cur_pw")
        new_pw  = st.text_input("Mật khẩu mới (≥8 ký tự)", type="password", key="new_pw")
        new_pw2 = st.text_input("Xác nhận mật khẩu mới", type="password", key="new_pw2")
        if st.button("🔑 Đổi mật khẩu", use_container_width=True):
            if new_pw != new_pw2:
                st.error("Mật khẩu xác nhận không khớp")
            elif len(new_pw) < 8:
                st.error("Mật khẩu mới cần ít nhất 8 ký tự")
            else:
                res = _change_password(token, cur_pw, new_pw)
                if "message" in res and "error" not in res:
                    st.success(res["message"])
                else:
                    st.error(res.get("message", "Đổi mật khẩu thất bại"))

    with st.expander("🛡️ Quyền hạn của tài khoản"):
        render_permission_summary()

    if st.button("✖ Đóng", use_container_width=True):
        st.session_state["show_profile"] = False
        st.rerun()


def is_authenticated() -> bool:
    return bool(st.session_state.get("is_logged_in"))


def get_current_user() -> dict:
    return st.session_state.get("auth_user", {})


def get_auth_token() -> str:
    return st.session_state.get("auth_token", "")


# ═══════════════════════════════════════════════════════════════════════════
# PERMISSION / RBAC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# Ma trận quyền theo role
_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
        "view_predictions", "run_prediction", "save_history",
        "view_history", "view_all_history", "generate_molecule",
        "view_model_stages", "view_ablation",
        "update_own_profile", "change_own_password",
        # Admin only
        "manage_users", "change_user_role", "lock_unlock_user",
        "delete_user", "create_user", "reset_user_password",
        "view_system_stats",
    },
    "user": {
        "view_predictions", "run_prediction", "save_history",
        "view_history", "generate_molecule",
        "view_model_stages", "view_ablation",
        "update_own_profile", "change_own_password",
    },
    "viewer": {
        "view_predictions",
        "view_model_stages", "view_ablation",
        "update_own_profile", "change_own_password",
    },
}

# Tên hiển thị của từng quyền
PERMISSION_LABELS: dict[str, str] = {
    "view_predictions":    "Xem kết quả dự đoán",
    "run_prediction":      "Chạy dự đoán mới",
    "save_history":        "Lưu lịch sử dự đoán",
    "view_history":        "Xem lịch sử cá nhân",
    "view_all_history":    "Xem lịch sử tất cả người dùng",
    "generate_molecule":   "Sinh phân tử VGAE",
    "view_model_stages":   "Xem các giai đoạn mô hình",
    "view_ablation":       "Xem kết quả Ablation",
    "update_own_profile":  "Cập nhật hồ sơ cá nhân",
    "change_own_password": "Đổi mật khẩu",
    "manage_users":        "Quản lý tài khoản",
    "change_user_role":    "Đổi vai trò người dùng",
    "lock_unlock_user":    "Khóa / mở tài khoản",
    "delete_user":         "Xóa tài khoản",
    "create_user":         "Tạo tài khoản mới",
    "reset_user_password": "Reset mật khẩu người dùng",
    "view_system_stats":   "Xem thống kê hệ thống",
}


def can_do(permission: str) -> bool:
    """Kiểm tra người dùng hiện tại có quyền `permission` không."""
    user = get_current_user()
    role = user.get("role", "viewer")
    return permission in _PERMISSIONS.get(role, set())


def require_permission(permission: str, msg: str = "") -> None:
    """Dừng trang nếu user không có quyền `permission`."""
    if not can_do(permission):
        label = PERMISSION_LABELS.get(permission, permission)
        reason = msg or f"Bạn không có quyền: **{label}**"
        st.markdown(f"""
        <div style="
            background:rgba(239,68,68,0.08);
            border:1px solid rgba(239,68,68,0.3);
            border-radius:14px; padding:28px 32px; text-align:center; margin:24px 0;">
            <div style="font-size:2.5rem;margin-bottom:12px;">🔒</div>
            <div style="font-size:1.1rem;font-weight:700;color:#f87171;margin-bottom:6px;">Từ chối truy cập</div>
            <div style="color:#94a3b8;font-size:0.9rem;">{reason}</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()


def require_role(*roles: str, msg: str = "") -> None:
    """Dừng trang nếu user không thuộc role nào trong `roles`."""
    user = get_current_user()
    role = user.get("role", "viewer")
    if role not in roles:
        allowed = " hoặc ".join(f"**{r}**" for r in roles)
        reason = msg or f"Trang này yêu cầu vai trò {allowed}"
        st.markdown(f"""
        <div style="
            background:rgba(239,68,68,0.08);
            border:1px solid rgba(239,68,68,0.3);
            border-radius:14px; padding:28px 32px; text-align:center; margin:24px 0;">
            <div style="font-size:2.5rem;margin-bottom:12px;">🔒</div>
            <div style="font-size:1.1rem;font-weight:700;color:#f87171;margin-bottom:6px;">Từ chối truy cập</div>
            <div style="color:#94a3b8;font-size:0.9rem;">{reason}</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()


def render_permission_summary() -> None:
    """Hiển thị bảng quyền cho user hiện tại (dùng trong profile hoặc trang admin)."""
    user = get_current_user()
    role = user.get("role", "viewer")
    perms = _PERMISSIONS.get(role, set())

    role_color = {"admin": "#fb923c", "user": "#818cf8", "viewer": "#94a3b8"}.get(role, "#94a3b8")
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);
                border-radius:12px;padding:16px 20px;margin-bottom:16px;">
        <div style="font-size:0.82rem;font-weight:600;color:#94a3b8;
                    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:10px;">
            Quyền hạn của vai trò
            <span style="color:{role_color};margin-left:8px;">{role.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    items = list(PERMISSION_LABELS.items())
    for i, (perm, label) in enumerate(items):
        has = perm in perms
        icon = "✅" if has else "❌"
        color = "#4ade80" if has else "#64748b"
        cols[i % 2].markdown(
            f'<div style="padding:4px 0;font-size:0.82rem;color:{color};">{icon} {label}</div>',
            unsafe_allow_html=True,
        )

