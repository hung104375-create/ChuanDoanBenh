import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import hashlib
from datetime import datetime

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG & NẠP STYLE CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hệ Thống Y Tế AI & Chẩn Đoán Lâm Sàng",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# ---------------------------------------------------------
# 2. HỆ THỐNG XÁC THỰC & LƯU LỊCH SỬ (JSON STORAGE)
# ---------------------------------------------------------
USER_FILE = "users.json"
HISTORY_FILE = "history.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_FILE):
        default_users = {
            "admin": {
                "password": hash_password("123456"),
                "fullname": "BS. Nguyễn Văn An",
                "role": "Bác sĩ / Chuyên gia"
            }
        }
        with open(USER_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, ensure_ascii=False, indent=4)
        return default_users
    
    with open(USER_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user(username, password, fullname, role):
    users = load_users()
    if username in users:
        return False, "Tên đăng nhập đã tồn tại!"
    users[username] = {
        "password": hash_password(password),
        "fullname": fullname,
        "role": role
    }
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)
    return True, "Đăng ký tài khoản thành công!"

def verify_login(username, password):
    users = load_users()
    if username in users and users[username]["password"] == hash_password(password):
        return True, users[username]
    return False, None

# Quản lý Lịch sử Chẩn đoán
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history_record(username, record_type, result_summary, details):
    history = load_history()
    new_entry = {
        "id": f"MED-{int(datetime.now().timestamp())}",
        "username": username,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": record_type,
        "summary": result_summary,
        "details": details
    }
    history.insert(0, new_entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# Khởi tạo Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None

# ---------------------------------------------------------
# 3. MÀN HÌNH ĐĂNG NHẬP / ĐĂNG KÝ
# ---------------------------------------------------------
if not st.session_state.logged_in:
    st.title("🩺 Trợ Lý Y Tế AI - Đăng Nhập Hệ Thống")
    st.caption("Nền tảng hỗ trợ chẩn đoán lâm sàng tích hợp AI & Hệ chuyên gia")
    st.markdown("---")

    auth_col1, auth_col2, auth_col3 = st.columns([1, 1.3, 1])

    with auth_col2:
        tab_login, tab_register = st.tabs(["🔑 Đăng Nhập", "📝 Đăng Ký"])

        with tab_login:
            st.subheader("Cổng Đăng Nhập")
            login_username = st.text_input("Tên đăng nhập", key="login_user")
            login_password = st.text_input("Mật khẩu", type="password", key="login_pass")

            if st.button("🚀 ĐĂNG NHẬP", type="primary"):
                if not login_username or not login_password:
                    st.warning("Vui lòng điền đầy đủ thông tin.")
                else:
                    success, user_data = verify_login(login_username, login_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_data
                        st.session_state.user_info["username"] = login_username
                        st.success(f"Xin chào {user_data['fullname']}!")
                        st.rerun()
                    else:
                        st.error("Tên đăng nhập hoặc mật khẩu chưa đúng!")

            st.info("💡 **Tài khoản demo:** Username: `admin` | Password: `123456`")

        with tab_register:
            st.subheader("Tạo Tài Khoản Mới")
            reg_fullname = st.text_input("Họ và tên", placeholder="Ví dụ: Nguyễn Văn A")
            reg_username = st.text_input("Tên đăng nhập (viết liền)", key="reg_user")
            reg_password = st.text_input("Mật khẩu", type="password", key="reg_pass")
            reg_confirm = st.text_input("Xác nhận mật khẩu", type="password")
            reg_role = st.selectbox("Vai trò", ["Bác sĩ / Chuyên gia", "Bệnh nhân / Người dùng"])

            if st.button("✨ ĐĂNG KÝ NGAY"):
                if not reg_fullname or not reg_username or not reg_password:
                    st.warning("Vui lòng nhập đầy đủ thông tin.")
                elif reg_password != reg_confirm:
                    st.error("Mật khẩu xác nhận không trùng khớp!")
                else:
                    ok, msg = save_user(reg_username, reg_password, reg_fullname, reg_role)
                    if ok:
                        st.success(f"🎉 {msg} Vui lòng đăng nhập.")
                    else:
                        st.error(f"❌ {msg}")

    st.stop()

# ---------------------------------------------------------
# 4. TRẠNG THÁI MÔ HÌNH KERAS & DỮ LIỆU
# ---------------------------------------------------------
@st.cache_resource
def load_heart_disease_model():
    try:
        model = load_model('best_model.h5')
        return model
    except Exception:
        return None

heart_model = load_heart_disease_model()

# CƠ SỞ TRI THỨC HỆ CHUYÊN GIA
DISEASE_RULES = {
    "Cúm mùa": {
        "symptoms": ["sốt", "ho", "đau cơ", "mệt mỏi", "đau đầu", "chảy nước mũi"],
        "advice": "Nghỉ ngơi, uống nhiều nước, bổ sung Vitamin C. Đến cơ sở y tế nếu sốt cao quá 3 ngày.",
        "icon": "🤒"
    },
    "Sốt xuất huyết": {
        "symptoms": ["sốt cao", "phát ban", "đau hốc mắt", "đau khớp", "đau đầu", "mệt mỏi"],
        "advice": "Cần đi xét nghiệm máu ngay. Không tự ý dùng Aspirin/Ibuprofen.",
        "icon": "🦟"
    },
    "Viêm họng cấp": {
        "symptoms": ["ho", "đau họng", "sốt nhẹ", "nuốt đau", "khản tiếng"],
        "advice": "Súc miệng bằng nước muối sinh lý, giữ ấm cổ họng và uống nước ấm.",
        "icon": "🗣️"
    },
    "Đau dạ dày": {
        "symptoms": ["đau thượng vị", "ợ chua", "buồn nôn", "đầy hơi", "chán ăn"],
        "advice": "Hạn chế đồ cay nóng, bia rượu. Ăn đúng giờ và tránh stress.",
        "icon": "🤢"
    },
    "Viêm phế quản": {
        "symptoms": ["ho có đờm", "sốt nhẹ", "khó thở", "tức ngực", "mệt mỏi"],
        "advice": "Tránh khói thuốc lá, giữ ấm ngực và đi khám bác sĩ nghe phổi.",
        "icon": "🫁"
    },
    "COVID-19": {
        "symptoms": ["sốt", "ho", "mất vị giác", "mất khứu giác", "mệt mỏi", "khó thở"],
        "advice": "Tự cách ly, đeo khẩu trang và thực hiện test nhanh.",
        "icon": "🦠"
    }
}

ALL_SYMPTOMS = sorted(list({sym for disease in DISEASE_RULES.values() for sym in disease["symptoms"]}))

def diagnose_engine(selected_symptoms):
    results = []
    if not selected_symptoms:
        return results
    for disease_name, info in DISEASE_RULES.items():
        symptoms_list = info["symptoms"]
        matched = [s for s in selected_symptoms if s in symptoms_list]
        if matched:
            confidence = (len(matched) / len(symptoms_list)) * 100
            results.append({
                "name": disease_name,
                "confidence": round(confidence, 1),
                "matched": matched,
                "advice": info["advice"],
                "icon": info["icon"]
            })
    return sorted(results, key=lambda x: x["confidence"], reverse=True)

# ---------------------------------------------------------
# 5. SIDEBAR ĐIỀU HƯỚNG & NÚT ĐĂNG XUẤT
# ---------------------------------------------------------
user = st.session_state.user_info
all_history = load_history()
user_history = [h for h in all_history if h["username"] == user["username"]]

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=80)
    st.title("Hệ Thống Y Tế AI")
    
    st.markdown(f"""
        <div class="user-badge">
            <b>👤 Người dùng:</b> {user['fullname']}<br>
            <b>🏷️ Vai trò:</b> {user['role']}
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Đăng Xuất"):
        st.session_state.logged_in = False
        st.session_state.user_info = None
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Thống Kê Nhanh")
    st.metric("Lượt chẩn đoán cá nhân", f"{len(user_history)} lần")
    
    if heart_model is not None:
        st.success("Mô hình AI Bệnh tim: **Active** ✅")
    else:
        st.error("Mô hình AI: **Missing** ❌")

# ---------------------------------------------------------
# 6. GIAO DIỆN CHÍNH (4 TABS CHUYÊN NGHIỆP)
# ---------------------------------------------------------
st.title("🩺 Hệ Thống Hỗ Trợ Chẩn Đoán Y Tế Tự Động")
st.markdown("Tích hợp **Mạng Nơ-ron Sâu (Deep Learning)** & **Hệ Chuyên Gia Dựa Trên Triệu Chứng**")

tab1, tab2, tab3, tab4 = st.tabs([
    "❤️ Chẩn Đoán Bệnh Tim (AI)", 
    "📋 Chẩn Đoán Triệu Chứng", 
    "📜 Lịch Sử Chẩn Đoán", 
    "📈 Bảng Điều Khiển & Tri Thức"
])

# =========================================================
# TAB 1: CHẨN ĐOÁN BỆNH TIM (AI MODEL)
# =========================================================
with tab1:
    st.subheader("📋 Nhập thông số lâm sàng bệnh nhân")
    
    if heart_model is None:
        st.warning("⚠️ Chưa tìm thấy tệp `best_model.h5`. Vui lòng chạy file `train.py` để khởi tạo mô hình.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Tuổi (age)", min_value=1, max_value=120, value=55)
            sex = st.selectbox("Giới tính (sex)", options=[1, 0], format_func=lambda x: "Nam" if x == 1 else "Nữ")
            cp = st.selectbox("Loại đau ngực (cp)", options=[0, 1, 2, 3], 
                              format_func=lambda x: {0: "0: Không triệu chứng", 1: "1: Đau thắt ngực điển hình", 2: "2: Đau thắt ngực không điển hình", 3: "3: Đau không do tim"}[x])
            trestbps = st.number_input("Huyết áp lúc nghỉ (trestbps - mmHg)", min_value=50, max_value=250, value=130)
            chol = st.number_input("Cholesterol (chol - mg/dl)", min_value=100, max_value=600, value=246)

        with col2:
            fbs = st.selectbox("Đường huyết lúc đói > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Có (>120 mg/dl)" if x == 1 else "Không (<=120 mg/dl)")
            restecg = st.selectbox("Điện tâm đồ lúc nghỉ (restecg)", options=[0, 1, 2],
                                   format_func=lambda x: {0: "0: Bình thường", 1: "1: Bất thường ST-T", 2: "2: Phì đại thất trái"}[x])
            thalach = st.number_input("Nhịp tim tối đa (thalach)", min_value=50, max_value=230, value=150)
            exang = st.selectbox("Đau ngực khi vận động (exang)", options=[0, 1], format_func=lambda x: "Có" if x == 1 else "Không")

        with col3:
            oldpeak = st.number_input("Độ chênh ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Độ dốc đoạn ST (slope)", options=[0, 1, 2],
                                 format_func=lambda x: {0: "0: Dốc lên", 1: "1: Phẳng", 2: "2: Dốc xuống"}[x])
            ca = st.selectbox("Số mạch chính soi màu (ca)", options=[0, 1, 2, 3])
            thal = st.selectbox("Chỉ số Thalassemia (thal)", options=[0, 1, 2, 3],
                                format_func=lambda x: {0: "0: Chưa xác định", 1: "1: Bình thường", 2: "2: Khuyết tật cố định", 3: "3: Khuyết tật đảo ngược"}[x])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 PHÂN TÍCH NGUY CƠ BỆNH TIM", type="primary"):
            input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = heart_model.predict(input_features)
            prob = float(prediction[0][0]) * 100

            # Lưu vào Lịch sử
            status_text = f"Nguy cơ CAO ({prob:.1f}%)" if prob >= 50 else f"Nguy cơ THẤP ({prob:.1f}%)"
            save_history_record(
                username=user["username"],
                record_type="Bệnh tim (AI)",
                result_summary=status_text,
                details=f"Tuổi: {age}, Giới tính: {'Nam' if sex==1 else 'Nữ'}, Huyết áp: {trestbps}, Cholesterol: {chol}"
            )

            st.markdown("---")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                st.subheader("🔍 Đánh Giá Từ Mô Hình Deep Learning:")
                if prob >= 50:
                    st.error(f"⚠️ **CẢNH BÁO:** Phát hiện nguy cơ mắc bệnh tim **CAO** ({prob:.1f}%)")
                    st.warning("Khuyến nghị: Cần đưa bệnh nhân đến cơ sở y tế chuyên khoa tim mạch để kiểm tra chuyên sâu.")
                else:
                    st.success(f"✅ **AN TOÀN:** Nguy cơ mắc bệnh tim **THẤP** ({prob:.1f}%)")
                    st.info("Khuyến nghị: Chỉ số ổn định. Tiếp tục duy trì lối sống lành mạnh và tập luyện thể thao.")

                # Xuất báo cáo dạng Text chuyên nghiệp
                report_text = f"""==================================================
BÁO CÁO KẾT QUẢ CHẨN ĐOÁN LÂM SÀNG BỆNH TIM
Hệ Thống Y Tế AI - AI Healthcare Assistant
==================================================
Mã Hồ Sơ: MED-{int(datetime.now().timestamp())}
Thời Gian Thực Hiện: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Người Đánh Giá: {user['fullname']} ({user['role']})

I. THÔNG TIN BỆNH NHÂN:
- Độ tuổi: {age}
- Giới tính: {'Nam' if sex==1 else 'Nữ'}
- Huyết áp lúc nghỉ: {trestbps} mmHg
- Cholesterol: {chol} mg/dl
- Nhịp tim tối đa: {thalach} bpm

II. KẾT QUẢ DỰ ĐOÁN TỪ MÔ HÌNH AI:
- Tỷ lệ xác suất bệnh tim: {prob:.2f}%
- Đánh giá nguy cơ: {'CAO (Cần can thiệp)' if prob >= 50 else 'THẤP (An toàn)'}

==================================================
*Báo cáo chỉ mang tính chất tham khảo cho môn học AI.*
"""
                st.download_button(
                    label="📄 Tải Báo Cáo Y Tế (.txt)",
                    data=report_text,
                    file_name=f"Bao_Cao_Y_Te_{age}T_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

            with res_col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Đồng Hồ Nguy Cơ (%)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#d90429" if prob >= 50 else "#2b9348"},
                        'steps': [
                            {'range': [0, 50], 'color': "#e8f5e9"},
                            {'range': [50, 100], 'color': "#ffebee"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2: CHẨN ĐOÁN TRIỆU CHỨNG (HỆ CHUYÊN GIA)
# =========================================================
with tab2:
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("📋 Chọn Triệu Chứng Cơ Thể")
        user_selected = st.multiselect(
            "Tích chọn các triệu chứng ghi nhận được:",
            options=ALL_SYMPTOMS,
            placeholder="Tìm triệu chứng..."
        )

        if user_selected:
            st.success(f"Đã chọn **{len(user_selected)}** triệu chứng.")
            if st.button("💾 Lưu Kết Quả Triệu Chứng Vào Lịch Sử"):
                preds = diagnose_engine(user_selected)
                top_disease = preds[0]['name'] if preds else "Không xác định"
                save_history_record(
                    username=user["username"],
                    record_type="Triệu chứng (Rule-based)",
                    result_summary=f"Khả năng cao: {top_disease}",
                    details=f"Triệu chứng: {', '.join(user_selected)}"
                )
                st.toast("Đã lưu kết quả vào lịch sử chẩn đoán!", icon="✅")
        else:
            st.info("Vui lòng chọn triệu chứng để hệ thống truy vấn cơ sở tri thức.")

    with col_right:
        st.subheader("🔍 Kết Quả Suy Luận Hệ Chuyên Gia")
        if user_selected:
            predictions = diagnose_engine(user_selected)
            if not predictions:
                st.error("Không tìm thấy bệnh phù hợp với tập triệu chứng này.")
            else:
                for idx, item in enumerate(predictions):
                    with st.expander(f"{item['icon']} **{item['name']}** — Độ phù hợp: {item['confidence']}%", expanded=(idx == 0)):
                        st.progress(item['confidence'] / 100)
                        st.write(f"**Triệu chứng khớp:** {', '.join(item['matched'])}")
                        st.info(f"💡 **Hướng xử lý:** {item['advice']}")
        else:
            st.info("Kết quả suy luận sẽ hiển thị ngay khi chọn triệu chứng.")

# =========================================================
# TAB 3: LỊCH SỬ CHẨN ĐOÁN (NEW FEATURE)
# =========================================================
with tab3:
    st.subheader("📜 Nhật Ký Lịch Sử Chẩn Đoán Bệnh Nhân")
    st.caption("Danh sách các lượt chẩn đoán đã thực hiện bởi tài khoản hiện tại")

    curr_user_history = [h for h in load_history() if h["username"] == user["username"]]

    if not curr_user_history:
        st.info("Chưa có lịch sử chẩn đoán nào được lưu.")
    else:
        df_hist = pd.DataFrame(curr_user_history)
        df_display = df_hist[['id', 'timestamp', 'type', 'summary', 'details']]
        df_display.columns = ['Mã HS', 'Thời gian', 'Phân loại', 'Kết quả chính', 'Chi tiết']
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# =========================================================
# TAB 4: BẢNG ĐIỀU KHIỂN & TRI THỨC (NEW FEATURE)
# =========================================================
with tab4:
    st.subheader("📈 Thống Kê Hệ Thống & Cơ Sở Tri Thức")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số Luật (Rules)", f"{len(DISEASE_RULES)} bệnh")
    m2.metric("Tổng Triệu Chứng Dữ Liệu", f"{len(ALL_SYMPTOMS)} triệu chứng")
    m3.metric("Tổng lượt Chẩn đoán Toàn hệ thống", f"{len(load_history())} lượt")

    st.markdown("---")
    st.write("### 🧠 Danh Sách Bệnh & Triệu Chứng Trong Cơ Sở Tri Thức")

    # Tạo biểu đồ số lượng triệu chứng theo bệnh bằng Plotly
    disease_counts = {d: len(info["symptoms"]) for d, info in DISEASE_RULES.items()}
    df_chart = pd.DataFrame({"Bệnh": list(disease_counts.keys()), "Số triệu chứng": list(disease_counts.values())})
    
    fig_bar = px.bar(
        df_chart, x="Bệnh", y="Số triệu chứng", 
        color="Số triệu chứng",
        title="Thống kê Số lượng Triệu chứng Đặc trưng theo Bệnh",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_bar, use_container_width=True)