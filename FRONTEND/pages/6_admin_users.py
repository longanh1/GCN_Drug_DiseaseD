# -*- coding: utf-8 -*-
"""
Page 6 – Quản lý tài khoản & Phân quyền (Admin only) — RBAC Edition
"""
import os, sys, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from utils.auth import (
    render_user_sidebar, render_profile_modal,
    is_authenticated, require_role, AUTH_CSS,
    get_current_user, get_auth_token,
)

if not is_authenticated():
    st.warning("Vui lòng đăng nhập trước")
    st.stop()

require_role("admin")   # blocks non-admin with 403 UI

_BACKEND = os.getenv("BACKEND_URL", "http://localhost:3000/api")

# ── Styles ─────────────────────────────────────────────────────────────
st.markdown(AUTH_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f1f5f9; }
[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stToolbarActions"], .stDeployButton { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }
.admin-header {
    background: linear-gradient(135deg, #1e293b 0%, #312e81 100%);
    border-radius: 18px; padding: 24px 32px;
    border: 1px solid rgba(99,102,241,0.3); margin-bottom: 24px;
}
.admin-title { font-size: 1.5rem; font-weight: 800; color: #e2e8f0; margin: 0 0 4px; }
.admin-sub   { font-size: 0.85rem; color: #94a3b8; }
.stat-card {
    background: #ffffff; border-radius: 14px; padding: 18px 20px;
    border: 1px solid #e8ecf0; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.user-row {
    background: #ffffff; border-radius: 12px; padding: 14px 18px;
    border: 1px solid #e8ecf0; margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.user-avatar-sm {
    width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 700; color: white; font-size: 0.9rem;
}
.bdg {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
}
.b-admin   { background: rgba(249,115,22,.15); color: #f97316; border: 1px solid rgba(249,115,22,.3); }
.b-user    { background: rgba(99,102,241,.12); color: #818cf8; border: 1px solid rgba(99,102,241,.25); }
.b-viewer  { background: rgba(100,116,139,.12);color: #94a3b8; border: 1px solid rgba(100,116,139,.25); }
.b-active  { background: rgba(34,197,94,.12);  color: #4ade80; border: 1px solid rgba(34,197,94,.25); }
.b-locked  { background: rgba(239,68,68,.12);  color: #f87171; border: 1px solid rgba(239,68,68,.25); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 14px;border-bottom:1px solid rgba(99,102,241,0.15);margin-bottom:8px;">
        <div style="width:36px;height:36px;border-radius:10px;
                    background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    display:inline-flex;align-items:center;justify-content:center;
                    font-size:1.1rem;margin-bottom:6px;">🧬</div>
        <div style="font-size:0.88rem;font-weight:700;
                    background:linear-gradient(90deg,#818cf8,#c084fc);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">PharmaLink GCN</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("home.py",                label="🏠  Tổng quan")
    st.page_link("pages/1_prediction.py",  label="🔬  Dự đoán & Phân tích")
    st.page_link("pages/2_history.py",     label="📋  Lịch sử")
    st.page_link("pages/6_admin_users.py", label="🛡️  Quản lý tài khoản")
    st.markdown("---")
    render_user_sidebar()

user  = get_current_user()
token = get_auth_token()

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="admin-header">
    <div class="admin-title">🛡️ Quản lý tài khoản & Phân quyền</div>
    <div class="admin-sub">Tạo mới • Đổi vai trò • Khóa / Mở • Reset mật khẩu • Xóa tài khoản</div>
</div>
""", unsafe_allow_html=True)

# ── API helpers ───────────────────────────────────────────────────────
def _hdrs():
    return {"Authorization": f"Bearer {token}"}

def _api(method: str, path: str, **kw):
    try:
        fn = {"get": requests.get, "post": requests.post,
              "patch": requests.patch, "delete": requests.delete}[method]
        r = fn(f"{_BACKEND}/{path}", headers=_hdrs(), timeout=10, **kw)
        return r.status_code, (r.json() if r.content else {})
    except Exception as e:
        return 0, {"error": str(e)}

@st.cache_data(ttl=20)
def load_stats(tok: str):
    _, d = _api("get", "auth/admin/stats")
    return d

@st.cache_data(ttl=20)
def load_users(tok: str, search: str = ""):
    url = "auth/admin/users" + (f"?search={search}" if search else "")
    _, d = _api("get", url)
    return d if isinstance(d, list) else []

def bust():
    st.cache_data.clear()
    st.rerun()

# ── Stats banner ──────────────────────────────────────────────────────
stats = load_stats(token)
c5 = st.columns(5)
for col, icon, val, lbl, clr in [
    (c5[0], "👥", stats.get("total",   "—"), "Tổng",           "#6366f1"),
    (c5[1], "✅", stats.get("active",  "—"), "Đang hoạt động", "#10b981"),
    (c5[2], "🔒", stats.get("inactive","—"), "Bị khóa",        "#ef4444"),
    (c5[3], "🛡️",stats.get("admins",  "—"), "Admin",          "#f59e0b"),
    (c5[4], "👤", stats.get("users",   "—"), "User",           "#818cf8"),
]:
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:1.3rem;">{icon}</div>
            <div style="font-size:1.7rem;font-weight:800;color:{clr};line-height:1.1;">{val}</div>
            <div style="font-size:0.72rem;color:#6b7280;margin-top:3px;">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────
tab_list, tab_create = st.tabs(["📋  Danh sách tài khoản", "➕  Tạo tài khoản mới"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 – User list
# ══════════════════════════════════════════════════════════════════════
with tab_list:
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        search_q = st.text_input("🔍 Tìm theo tên / email / username",
                                 placeholder="Nhập từ khoá...", label_visibility="collapsed")
    with sc2:
        if st.button("🔄 Làm mới", use_container_width=True):
            bust()

    users = load_users(token, search_q)
    if not users:
        st.info("Không tìm thấy tài khoản nào")
    else:
        for u in users:
            uid      = u.get("id", "")
            name     = u.get("fullName") or u.get("username", "?")
            initials = "".join(w[0].upper() for w in name.split()[:2]) or "?"
            role     = u.get("role", "user")
            active   = u.get("isActive", True)
            is_self  = uid == user.get("id")

            r_cls = f"b-{role}"
            s_cls = "b-active" if active else "b-locked"
            s_lbl = "Hoạt động" if active else "Bị khóa"
            r_lbl = {"admin": "Admin", "user": "User", "viewer": "Viewer"}.get(role, role)

            hdr_col, btn_col = st.columns([7, 3])
            with hdr_col:
                self_tag = "&nbsp;<span style='font-size:0.72rem;color:#94a3b8'>(bạn)</span>" if is_self else ""
                st.markdown(f"""
                <div class="user-row" style="display:flex;align-items:center;gap:12px;">
                    <div class="user-avatar-sm">{initials}</div>
                    <div style="flex:1;min-width:0;">
                        <div style="font-weight:700;color:#1e293b;font-size:0.9rem;">
                            {name}{self_tag}
                            &nbsp;<span class="bdg {r_cls}">{r_lbl}</span>
                            &nbsp;<span class="bdg {s_cls}">{s_lbl}</span>
                        </div>
                        <div style="color:#6b7280;font-size:0.77rem;margin-top:2px;">
                            📧 {u.get("email","")} &nbsp;·&nbsp;
                            @{u.get("username","")} &nbsp;·&nbsp;
                            {str(u.get("createdAt",""))[:10]}
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with btn_col:
                if not is_self:
                    bc1, bc2, bc3 = st.columns(3)
                    # Lock / Unlock
                    with bc1:
                        lk_icon = "🔓" if not active else "🔒"
                        lk_tip  = "Mở khóa" if not active else "Khóa tài khoản"
                        if st.button(lk_icon, key=f"tog_{uid}", help=lk_tip, use_container_width=True):
                            code, res = _api("patch", f"auth/admin/users/{uid}/toggle")
                            if code == 200:
                                st.toast("✅ Đã cập nhật trạng thái")
                                bust()
                            else:
                                st.error(res.get("message", "Lỗi"))
                    # Change Role
                    with bc2:
                        if st.button("🎭", key=f"role_btn_{uid}", help="Đổi vai trò", use_container_width=True):
                            st.session_state[f"edit_role_{uid}"] = True
                    # Delete
                    with bc3:
                        if st.button("🗑️", key=f"del_{uid}", help="Xóa tài khoản", use_container_width=True):
                            st.session_state[f"confirm_del_{uid}"] = True
                else:
                    st.caption("(không thể chỉnh sửa bản thân)")

            # Change Role panel
            if st.session_state.get(f"edit_role_{uid}"):
                with st.container():
                    opts = ["user", "viewer", "admin"]
                    new_role = st.selectbox(
                        f"Đổi vai trò cho @{u.get('username','')}",
                        options=opts,
                        index=opts.index(role) if role in opts else 0,
                        key=f"sel_role_{uid}",
                    )
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        if st.button("✅ Xác nhận", key=f"conf_role_{uid}", use_container_width=True):
                            code, res = _api("patch", f"auth/admin/users/{uid}/role", json={"role": new_role})
                            if code == 200:
                                st.toast(f"✅ Đã đổi role → {new_role.upper()}")
                                del st.session_state[f"edit_role_{uid}"]
                                bust()
                            else:
                                st.error(res.get("message", "Lỗi"))
                    with rc2:
                        if st.button("✖ Hủy", key=f"cancel_role_{uid}", use_container_width=True):
                            del st.session_state[f"edit_role_{uid}"]
                            st.rerun()

            # Confirm delete
            if st.session_state.get(f"confirm_del_{uid}"):
                st.warning(f"⚠️ Xác nhận xóa tài khoản **@{u.get('username','')}**? Thao tác không thể hoàn tác!")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("🗑️ Xóa ngay", key=f"do_del_{uid}",
                                 use_container_width=True, type="primary"):
                        code, res = _api("delete", f"auth/admin/users/{uid}")
                        if code in (200, 204):
                            st.toast(f"🗑️ Đã xóa @{u.get('username','')}")
                            del st.session_state[f"confirm_del_{uid}"]
                            bust()
                        else:
                            st.error(res.get("message", "Lỗi khi xóa"))
                with dc2:
                    if st.button("Hủy", key=f"cancel_del_{uid}", use_container_width=True):
                        del st.session_state[f"confirm_del_{uid}"]
                        st.rerun()

            # Reset password expander
            with st.expander(f"🔑 Reset mật khẩu — @{u.get('username','')}", expanded=False):
                new_pw = st.text_input("Mật khẩu mới (≥ 8 ký tự)", type="password",
                                       key=f"reset_pw_{uid}", placeholder="•••••••••")
                if st.button("Đặt lại mật khẩu", key=f"do_reset_{uid}", use_container_width=True):
                    if len(new_pw) < 8:
                        st.error("Mật khẩu cần ít nhất 8 ký tự")
                    else:
                        code, res = _api("post", f"auth/admin/users/{uid}/reset-password",
                                         json={"newPassword": new_pw})
                        if code == 200:
                            st.success(res.get("message", "Đã reset"))
                        else:
                            st.error(res.get("message", "Lỗi"))

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 – Create new user
# ══════════════════════════════════════════════════════════════════════
with tab_create:
    st.markdown("### ➕ Tạo tài khoản mới")
    st.caption("Admin có thể tạo tài khoản với bất kỳ role nào — bỏ qua quy trình đăng ký thông thường.")

    with st.form("create_user_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        new_email    = c1.text_input("Email *")
        new_username = c2.text_input("Tên đăng nhập *")
        new_fullname = c1.text_input("Họ tên đầy đủ")
        new_password = c2.text_input("Mật khẩu (≥ 8 ký tự) *", type="password")

        c3, c4 = st.columns(2)
        new_role   = c3.selectbox("Vai trò", ["user", "viewer", "admin"])
        new_active = c4.selectbox("Trạng thái", ["Hoạt động", "Bị khóa"]) == "Hoạt động"

        submitted = st.form_submit_button("✅ Tạo tài khoản", use_container_width=True, type="primary")
        if submitted:
            if not new_email or not new_username or not new_password:
                st.error("Vui lòng điền đầy đủ Email, Tên đăng nhập và Mật khẩu")
            elif len(new_password) < 8:
                st.error("Mật khẩu cần ít nhất 8 ký tự")
            else:
                payload = {
                    "email": new_email, "username": new_username,
                    "password": new_password, "role": new_role,
                    "isActive": new_active,
                }
                if new_fullname:
                    payload["fullName"] = new_fullname
                code, res = _api("post", "auth/admin/users", json=payload)
                if code == 201:
                    st.success(f"✅ Đã tạo tài khoản **@{res.get('username','')}** với role **{new_role.upper()}**")
                    bust()
                else:
                    msg = res.get("message", "Tạo tài khoản thất bại")
                    st.error(msg if isinstance(msg, str) else " | ".join(msg))

render_profile_modal()
