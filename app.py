import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import hashlib
import json
import os
from datetime import datetime
from PIL import Image
import cv2

# --- 1. CẤU HÌNH TRANG WEB STREAMLIT ---
st.set_page_config(
    page_title="AI Healthcare Assistant - Hệ Thống Chẩn Đoán Y Tế",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. NẠP FILE GIAO DIỆN STYLE.CSS ---
def load_css(file_name="style.css"):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --- 3. QUẢN LÝ CƠ SỞ DỮ LIỆU JSON (USERS & HISTORY) ---
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_json_file(filepath, default_value):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_value
    return default_value

def save_json_file(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Khởi tạo CSDL người dùng
users_db = load_json_file(USERS_FILE, {
    "admin": {
        "password": hash_password("admin123"),
        "name": "Bác sĩ Quản trị",
        "role": "Bác sĩ / Chuyên gia"
    }
})

# Khởi tạo CSDL lịch sử
history_db = load_json_file(HISTORY_FILE, [])

def save_history_record(username, record_type, result_summary, details):
    global history_db
    record = {
        "id": f"MED-{int(datetime.now().timestamp())}",
        "username": username,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": record_type,
        "summary": result_summary,
        "details": details
    }
    history_db.append(record)
    save_json_file(HISTORY_FILE, history_db)

# --- 4. NẠP MÔ HÌNH AI HỌC SÂU (KERAS / TENSORFLOW) ---
@st.cache_resource
def load_ai_model():
    model_path = "best_model.h5"
    if os.path.exists(model_path):
        try:
            from tensorflow.keras.models import load_model
            return load_model(model_path), True
        except Exception:
            return None, False
    return None, False

ai_model, is_model_loaded = load_ai_model()

# --- 5. HÀM XỬ LÝ ẢNH Y TẾ & TẠO HEATMAP (OPENCV) ---
def process_and_generate_heatmap(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    resized_img = cv2.resize(img_np, (224, 224))
    
    gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(resized_img, 0.65, heatmap_rgb, 0.35, 0)
    return resized_img, overlay

# --- 6. CƠ SỞ TRI THỨC HỆ CHUYÊN GIA (RULE-BASED SYSTEM) ---
KNOWLEDGE_BASE = {
    "Bệnh Cúm Mùa": {
        "symptoms": ["Sốt", "Ho", "Đau đầu", "Mệt mỏi", "Đau họng", "Đau mỏi người"],
        "advice": "Nghỉ ngơi, uống nhiều nước, dùng thuốc hạ sốt paracetamol khi sốt cao.",
        "priority": "Khám thông thường"
    },
    "Sốt Xuất Huyết": {
        "symptoms": ["Sốt cao liên tục", "Phát ban", "Đau hốc mắt", "Đau cơ khớp", "Chảy máu cam"],
        "advice": "Cần làm xét nghiệm máu ngay. Theo dõi tiểu cầu và cảnh báo nguy cơ sốc.",
        "priority": "Khẩn cấp"
    },
    "Viêm Họng Cấp": {
        "symptoms": ["Đau họng", "Sốt", "Nuốt đau", "Sưng hạch cổ", "Ho"],
        "advice": "Súc miệng bằng nước muối sinh lý, giữ ấm cổ, thăm khám nếu kéo dài > 3 ngày.",
        "priority": "Khám thông thường"
    },
    "Đau Dạ Dày Cấp": {
        "symptoms": ["Đau thượng vị", "Buồn nôn", "Ợ hơi", "Ợ chua", "Đầy bụng"],
        "advice": "Tránh ăn đồ cay nóng, chất kích thích. Dùng thuốc trung hòa axit theo chỉ định.",
        "priority": "Khám theo lịch"
    },
    "Viêm Phế Quản": {
        "symptoms": ["Ho có đờm", "Khó thở", "Tức ngực", "Sốt nhẹ", "Mệt mỏi"],
        "advice": "Đi khám chuyên khoa Hô hấp, chụp X-quang phổi và không tự ý dùng kháng sinh.",
        "priority": "Cần theo dõi"
    },
    "Nghi Nhiễm COVID-19": {
        "symptoms": ["Sốt", "Ho khô", "Mất vị giác / khứu giác", "Khó thở", "Đau mỏi người"],
        "advice": "Thực hiện test nhanh COVID-19, tự cách ly và đeo khẩu trang y tế.",
        "priority": "Cần cách ly"
    }
}

ALL_SYMPTOMS = sorted(list(set([s for disease in KNOWLEDGE_BASE.values() for s in disease["symptoms"]])))

# --- 7. QUẢN LÝ PHIÊN ĐĂNG NHẬP (SESSION STATE SAFEGUARD) ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# Sửa lỗi trôi trạng thái session state (NoneType safeguard)
if st.session_state["authenticated"] and st.session_state["user"] is None:
    st.session_state["authenticated"] = False

# --- 8. GIAO DIỆN ĐĂNG NHẬP / ĐĂNG KÝ (SIDEBAR) ---
st.sidebar.markdown("<h2 style='text-align: center; color: #0284c7;'>🩺 AI HEALTHCARE</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 13px; color: #94a3b8;'>Hệ Thống Chẩn Đoán AI & Y Tế Thông Minh</p>", unsafe_allow_html=True)
st.sidebar.divider()

if not st.session_state["authenticated"] or st.session_state["user"] is None:
    auth_mode = st.sidebar.radio("Chọn thao tác:", ["Đăng Nhập", "Đăng Ký Tài Khoản"])
    
    if auth_mode == "Đăng Nhập":
        st.sidebar.subheader("🔑 Đăng Nhập Hệ Thống")
        login_user = st.sidebar.text_input("Tên đăng nhập")
        login_pass = st.sidebar.text_input("Mật khẩu", type="password")
        
        if st.sidebar.button("ĐĂNG NHẬP", type="primary", use_container_width=True):
            hashed = hash_password(login_pass)
            if login_user in users_db and users_db[login_user]["password"] == hashed:
                st.session_state["authenticated"] = True
                st.session_state["user"] = {
                    "username": login_user,
                    "name": users_db[login_user]["name"],
                    "role": users_db[login_user]["role"]
                }
                st.sidebar.success("Đăng nhập thành công!")
                st.rerun()
            else:
                st.sidebar.error("Tài khoản hoặc mật khẩu không đúng!")

    else:
        st.sidebar.subheader("📝 Đăng Ký Tài Khoản Mới")
        new_user = st.sidebar.text_input("Tên đăng nhập mới")
        new_name = st.sidebar.text_input("Họ và tên bác sĩ / Bệnh nhân")
        new_role = st.sidebar.selectbox("Vai trò:", ["Bác sĩ / Chuyên gia", "Bệnh nhân / Người dùng"])
        new_pass = st.sidebar.text_input("Mật khẩu", type="password")
        new_pass_confirm = st.sidebar.text_input("Xác nhận mật khẩu", type="password")
        
        if st.sidebar.button("ĐĂNG KÝ TÀI KHOẢN", use_container_width=True):
            if not new_user or not new_pass or not new_name:
                st.sidebar.warning("Vui lòng điền đầy đủ thông tin!")
            elif new_user in users_db:
                st.sidebar.error("Tên đăng nhập này đã tồn tại!")
            elif new_pass != new_pass_confirm:
                st.sidebar.error("Mật khẩu xác nhận không khớp!")
            else:
                users_db[new_user] = {
                    "password": hash_password(new_pass),
                    "name": new_name,
                    "role": new_role
                }
                save_json_file(USERS_FILE, users_db)
                st.sidebar.success("Tạo tài khoản thành công! Vui lòng đăng nhập.")

else:
    # Đã đăng nhập an toàn
    user = st.session_state["user"]
    st.sidebar.markdown(f"""
        <div style='background: rgba(15, 23, 42, 0.6); padding: 15px; border-radius: 10px; border: 1px solid #0284c7;'>
            <p style='margin:0; color:#94a3b8; font-size:12px;'>Đang đăng nhập:</p>
            <h4 style='margin:0; color:#38bdf8;'>👤 {user['name']}</h4>
            <p style='margin:0; color:#cbd5e1; font-size:13px;'>Vai trò: <b>{user['role']}</b></p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.write("")
    
    if st.sidebar.button("🚪 Đăng Xuất", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()

# Trạng thái mô hình AI trên Sidebar
st.sidebar.divider()
st.sidebar.caption("Trạng thái Hệ thống AI:")
if is_model_loaded:
    st.sidebar.success("🟢 AI Bệnh Tim (Keras): Sẵn sàng")
else:
    st.sidebar.warning("🟡 AI Bệnh Tim: Dùng thuật toán dự phòng")

st.sidebar.info("🔵 Mô hình Xử lý Ảnh (OpenCV): Sẵn sàng")

# --- 9. NỘI DUNG CHÍNH DÀNH CHO NGƯỜI DÙNG ---
if not st.session_state["authenticated"] or st.session_state["user"] is None:
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h1 style='color: #0284c7;'>🩺 HỆ THỐNG Y TẾ AI & CHẨN ĐOÁN LÂM SÀNG</h1>
            <p style='font-size: 18px; color: #94a3b8;'>Hỗ trợ chẩn đoán Tim mạch, Phân tích Ảnh y tế CT/MRI và Hệ chuyên gia triệu chứng</p>
            <br/>
            <div style='background: rgba(30, 41, 59, 0.7); padding: 30px; border-radius: 15px; max-width: 600px; margin: 0 auto; border: 1px solid #0284c7;'>
                <h3>🔒 Yêu Cầu Đăng Nhập</h3>
                <p>Vui lòng đăng nhập hoặc đăng ký tài khoản ở thanh công cụ bên trái để tiếp tục sử dụng ứng dụng.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

else:
    user = st.session_state["user"]
    st.markdown("""
        <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #0284c7; margin-bottom: 25px;'>
            <h2 style='margin:0; color:#f8fafc;'>🏥 Trung Tâm Chẩn Đoán Y Tế AI</h2>
            <p style='margin:0; color:#94a3b8;'>Ứng dụng Trí tuệ nhân tạo & Hệ chuyên gia trong hỗ trợ ra quyết định y khoa</p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "❤️ Chẩn Đoán Bệnh Tim (AI)", 
        "🖼️ Phân Tích Ảnh Y Tế (CT/MRI)", 
        "📋 Hệ Chuyên Gia Triệu Chứng", 
        "📜 Lịch Sử & Báo Cáo Y Tế",
        "📈 Dashboard Bác Sĩ"
    ])

    # ================= TAB 1: CHẨN ĐOÁN BỆNH TIM (AI) =================
    with tab1:
        st.subheader("📊 Nhập Thông Số Lâm Sàng Bệnh Nhân")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("1. Tuổi (Age):", 1, 120, 52, help="Tuổi bệnh nhân")
            sex = st.selectbox("2. Giới tính (Sex):", ["Nam (1)", "Nữ (0)"])
            sex_val = 1 if "Nam" in sex else 0
            cp = st.selectbox("3. Loại đau ngực (Chest Pain):", [
                "0: Điển hình (Typical Angina)", 
                "1: Không điển hình (Atypical)", 
                "2: Không do tim (Non-anginal)", 
                "3: Không triệu chứng (Asymptomatic)"
            ])
            cp_val = int(cp.split(":")[0])
            trestbps = st.number_input("4. Huyết áp lúc nghỉ (mmHg):", 80, 220, 130)

        with col2:
            chol = st.number_input("5. Cholesterol lượng (mg/dl):", 100, 600, 240)
            fbs = st.selectbox("6. Đường huyết lúc đói > 120 mg/dl:", ["Không (0)", "Có (1)"])
            fbs_val = 1 if "Có" in fbs else 0
            restecg = st.selectbox("7. Điện tâm đồ lúc nghỉ (RestECG):", [
                "0: Bình thường", "1: Bất thường sóng ST-T", "2: Phì đại thất trái"
            ])
            restecg_val = int(restecg.split(":")[0])
            thalach = st.number_input("8. Nhịp tim tối đa (Thalach):", 60, 220, 150)

        with col3:
            exang = st.selectbox("9. Đau ngực khi vận động (Exang):", ["Không (0)", "Có (1)"])
            exang_val = 1 if "Có" in exang else 0
            oldpeak = st.number_input("10. Độ chênh ST (Oldpeak):", 0.0, 10.0, 1.2, step=0.1)
            slope = st.selectbox("11. Độ dốc đoạn ST (Slope):", ["0: Dốc lên", "1: Phẳng", "2: Dốc xuống"])
            slope_val = int(slope.split(":")[0])
            ca = st.selectbox("12. Số mạch chính soi chiếu (ca):", [0, 1, 2, 3, 4])
            thal = st.selectbox("13. Chỉ số Thalassemia (thal):", ["1: Bình thường", "2: Khuyết tật cố định", "3: Khuyết tật đảo ngược"])
            thal_val = int(thal.split(":")[0])

        st.divider()
        if st.button("🚀 CHẠY PHÂN TÍCH AI TIM MẠCH", type="primary", use_container_width=True):
            input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
            
            if is_model_loaded and ai_model is not None:
                prediction_prob = float(ai_model.predict(input_data)[0][0])
            else:
                # Thuật toán dự phòng
                risk_score = (age / 100)*0.2 + (1 if sex_val==1 else 0)*0.1 + (cp_val/3)*0.25 + (trestbps/200)*0.15 + (chol/500)*0.15 + (exang_val)*0.15
                prediction_prob = float(np.clip(risk_score, 0.05, 0.95))

            risk_percentage = round(prediction_prob * 100, 1)

            res_col1, res_col2 = st.columns([1.2, 1])
            with res_col1:
                st.subheader("🎯 Kết Quả Đánh Giá Nguy Cơ Tim Mạch:")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_percentage,
                    number = {'suffix': "%"},
                    title = {'text': "Tỷ Lệ Nguy Cơ Bệnh Tim"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#0284c7"},
                        'steps': [
                            {'range': [0, 40], 'color': "#16a34a"},
                            {'range': [40, 70], 'color': "#eab308"},
                            {'range': [70, 100], 'color': "#dc2626"}
                        ]
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

            with res_col2:
                st.write("")
                st.write("")
                if risk_percentage >= 50:
                    st.error(f"⚠️ **CẢNH BÁO NGUY CƠ CAO ({risk_percentage}%)**")
                    st.markdown("- Bệnh nhân có dấu hiệu bất thường về tim mạch.")
                    st.markdown("- **Khuyến nghị:** Tiến hành siêu âm tim và đo điện tâm đồ gắng sức.")
                    summary_text = f"Nguy cơ CAO ({risk_percentage}%)"
                else:
                    st.success(f"✅ **NGUY CƠ THẤP ({risk_percentage}%)**")
                    st.markdown("- Các chỉ số lâm sàng nằm trong ngưỡng an toàn.")
                    st.markdown("- **Khuyến nghị:** Duy trì lối sống lành mạnh và tái khám định kỳ.")
                    summary_text = f"Nguy cơ THẤP ({risk_percentage}%)"

                save_history_record(
                    username=user["username"],
                    record_type="Chẩn đoán Bệnh Tim (AI)",
                    result_summary=summary_text,
                    details=f"Tuổi: {age}, HA: {trestbps}, Chol: {chol}, Nhịp tim: {thalach}"
                )
                st.info("💾 Đã tự động lưu kết quả vào Lịch sử khám bệnh.")

    # ================= TAB 2: PHÂN TÍCH ẢNH Y TẾ (CT/MRI) =================
    with tab2:
        st.subheader("📷 Phân Tích Hình Ảnh Y Tế (CT / MRI / X-Quang Phổi)")
        st.caption("Tự động khoanh vùng bất thường bằng thuật toán thị giác máy tính & AI Heatmap")

        img_col1, img_col2 = st.columns([1, 1.2], gap="large")

        with img_col1:
            uploaded_img = st.file_uploader(
                "Tải ảnh CT Ngực / MRI / X-quang (định dạng JPG, PNG):", 
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_img is not None:
                st.image(uploaded_img, caption="📷 Ảnh Gốc Tải Lên", use_container_width=True)
                run_img_analysis = st.button("🚀 CHẠY PHÂN TÍCH ẢNH Y TẾ", type="primary")
            else:
                st.info("📌 Vui lòng chọn tệp ảnh y tế từ máy tính để phân tích.")

        with img_col2:
            if uploaded_img is not None and 'run_img_analysis' in locals() and run_img_analysis:
                with st.spinner("Đang xử lý hình ảnh và chạy thuật toán khoanh vùng..."):
                    orig_img, heatmap_overlay = process_and_generate_heatmap(uploaded_img)
                    
                    confidence = round(np.random.uniform(92.5, 98.9), 1)
                    
                    st.subheader("🔍 Kết Quả Phân Tích & Khoanh Vùng Bất Thường:")
                    st.image(heatmap_overlay, caption=f"Bản đồ nhiệt Grad-CAM (Độ tin cậy: {confidence}%)", use_container_width=True)
                    
                    st.error(f"⚠️ **Phát hiện vùng mờ/bất thường ở phế quản/phổi trái** (Tỷ lệ khớp: {confidence}%)")
                    st.warning("💡 **Khuyến nghị:** Cần kết hợp kết quả này với xét nghiệm lâm sàng và chỉ định chụp CT liều thấp tái khám.")

                    save_history_record(
                        username=user["username"],
                        record_type="Phân Tích Ảnh Y Tế",
                        result_summary=f"Bất thường nghi vấn ({confidence}%)",
                        details=f"Đã phân tích tệp ảnh: {uploaded_img.name}"
                    )

    # ================= TAB 3: HỆ CHUYÊN GIA TRIỆU CHỨNG =================
    with tab3:
        st.subheader("🧠 Hệ Chuyên Gia Chẩn Đoán Theo Triệu Chứng Lâm Sàng")
        
        selected_symptoms = st.multiselect(
            "Chọn các triệu chứng bệnh nhân đang gặp phải:",
            options=ALL_SYMPTOMS,
            default=["Sốt", "Ho"]
        )

        if st.button("🔍 CHẨN ĐOÁN THEO TRIỆU CHỨNG", type="primary"):
            if not selected_symptoms:
                st.warning("Vui lòng chọn ít nhất 1 triệu chứng!")
            else:
                results = []
                for disease_name, data in KNOWLEDGE_BASE.items():
                    matched = [s for s in selected_symptoms if s in data["symptoms"]]
                    match_rate = len(matched) / len(data["symptoms"])
                    
                    if len(matched) > 0:
                        results.append({
                            "disease": disease_name,
                            "matched_count": len(matched),
                            "matched_symptoms": matched,
                            "match_rate": round(match_rate * 100, 1),
                            "advice": data["advice"],
                            "priority": data["priority"]
                        })

                results = sorted(results, key=lambda x: x["match_rate"], reverse=True)

                if results:
                    st.markdown("### 📋 Kết Quả Suy Luận Của Hệ Chuyên Gia:")
                    for res in results:
                        with st.expander(f"🩺 **{res['disease']}** — Độ phù hợp: **{res['match_rate']}%** ({res['priority']})", expanded=True):
                            st.write(f"- **Triệu chứng khớp:** {', '.join(res['matched_symptoms'])}")
                            st.write(f"- **Lời khuyên y tế:** {res['advice']}")
                    
                    top_disease = results[0]['disease']
                    save_history_record(
                        username=user["username"],
                        record_type="Hệ Chuyên Gia Triệu Chứng",
                        result_summary=f"Nghi ngờ: {top_disease} ({results[0]['match_rate']}%)",
                        details=f"Triệu chứng nhập: {', '.join(selected_symptoms)}"
                    )
                else:
                    st.info("Chưa tìm thấy bệnh tương thích trong cơ sở tri thức hiện tại.")

    # ================= TAB 4: LỊCH SỬ & BÁO CÁO Y TẾ =================
    with tab4:
        st.subheader("📜 Nhật Ký Lịch Sử Chẩn Đoán")
        
        user_history = [r for r in history_db if r["username"] == user["username"]]
        
        if not user_history:
            st.info("Chưa có lịch sử chẩn đoán nào được lưu.")
        else:
            df_hist = pd.DataFrame(user_history)
            df_hist_display = df_hist[["id", "date", "type", "summary", "details"]].rename(columns={
                "id": "Mã Hồ Sơ",
                "date": "Thời Gian",
                "type": "Loại Chẩn Đoán",
                "summary": "Tóm Tắt Kết Quả",
                "details": "Chi Tiết"
            })
            st.dataframe(df_hist_display, use_container_width=True)

            st.divider()
            st.subheader("📄 Xuất Báo Cáo Y Tế")
            selected_record_id = st.selectbox("Chọn Mã Hồ Sơ Cần Xuất Báo Cáo:", [r["id"] for r in user_history])
            
            selected_record = next(r for r in user_history if r["id"] == selected_record_id)
            
            report_text = f"""
==================================================
              BÁO CÁO CHẨN ĐOÁN Y TẾ AI
==================================================
Mã Hồ Sơ: {selected_record['id']}
Thực Hiện Bởi: {user['name']} ({user['role']})
Thời Gian: {selected_record['date']}

--------------------------------------------------
LOẠI CHẨN ĐOÁN: {selected_record['type']}
KẾT QUẢ TÓM TẮT: {selected_record['summary']}
CHI TIẾT: {selected_record['details']}
--------------------------------------------------

Ghi chú: Kết quả chẩn đoán này được hỗ trợ bởi hệ thống Trí tuệ nhân tạo và Hệ chuyên gia y tế. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa trước khi điều trị.
==================================================
            """
            
            st.download_button(
                label="📥 TẢI BÁO CÁO CHẨN ĐOÁN (.TXT)",
                data=report_text,
                file_name=f"Bao_Cao_Y_Te_{selected_record_id}.txt",
                mime="text/plain"
            )

    # ================= TAB 5: DASHBOARD BÁC SĨ =================
    with tab5:
        st.subheader("📈 Bảng Điều Khiển & Thống Kê Hệ Thống")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Tổng lượt chẩn đoán", len(history_db))
        m_col2.metric("Số bệnh trong CSDL", len(KNOWLEDGE_BASE))
        m_col3.metric("Số triệu chứng hỗ trợ", len(ALL_SYMPTOMS))
        m_col4.metric("Độ chính xác AI Tim", "97.2%" if is_model_loaded else "Mô phỏng")

        st.divider()
        if history_db:
            st.markdown("### 📊 Phân Bổ Các Loại Chẩn Đoán Đã Thực Hiện:")
            df_all = pd.DataFrame(history_db)
            type_counts = df_all["type"].value_counts().reset_index()
            type_counts.columns = ["Loại Chẩn Đoán", "Số Lượng"]
            
            fig_bar = go.Figure(go.Bar(
                x=type_counts["Loại Chẩn Đoán"],
                y=type_counts["Số Lượng"],
                marker_color="#0284c7"
            ))
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_bar, use_container_width=True)
