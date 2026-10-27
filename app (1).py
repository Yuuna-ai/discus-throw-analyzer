import streamlit as st
import numpy as np
import cv2
import tempfile
import mediapipe as mp
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="円盤投げアドバンスド・プロ・アナライザー",
    page_icon="💿",
    layout="wide"
)

st.markdown('''<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    h1, h2, h3, h4 { color: #38bdf8 !important; }
    @media (max-width: 768px) {
        .stMetric { padding: 8px !important; margin: 5px 0 !important; }
        .css-1e5imqq { padding: 10px !important; }
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: column !important;
        }
    }
</style>''', unsafe_allow_html=True)

st.title("💿 円盤投げ アドバンスド・プロ・アナライザー")
st.markdown("📱 **スマートフォン対応UI** | 重心(CoM)、体軸傾斜、スイング速度、遠心力の自動測定 ＆ サークル内足裏設置プロット")

def calculate_segment_com(p1, p2, ratio=0.5):
    return p1 + ratio * (p2 - p1)

def calculate_total_body_com(landmarks, w, h):
    L_SH = np.array([landmarks[11].x * w, landmarks[11].y * h])
    R_SH = np.array([landmarks[12].x * w, landmarks[12].y * h])
    L_EL = np.array([landmarks[13].x * w, landmarks[13].y * h])
    R_EL = np.array([landmarks[14].x * w, landmarks[14].y * h])
    L_WR = np.array([landmarks[15].x * w, landmarks[15].y * h])
    R_WR = np.array([landmarks[16].x * w, landmarks[16].y * h])
    L_HP = np.array([landmarks[23].x * w, landmarks[23].y * h])
    R_HP = np.array([landmarks[24].x * w, landmarks[24].y * h])
    L_KN = np.array([landmarks[25].x * w, landmarks[25].y * h])
    R_KN = np.array([landmarks[26].x * w, landmarks[26].y * h])
    L_AK = np.array([landmarks[27].x * w, landmarks[27].y * h])
    R_AK = np.array([landmarks[28].x * w, landmarks[28].y * h])
    NOSE = np.array([landmarks[0].x * w, landmarks[0].y * h])
    segments = [
        {"com": calculate_segment_com((L_SH + R_SH)/2, (L_HP + R_HP)/2, 0.45), "weight": 0.483},
        {"com": calculate_segment_com(NOSE, (L_SH + R_SH)/2, 0.5), "weight": 0.081},
        {"com": calculate_segment_com(R_SH, R_EL, 0.436), "weight": 0.028},
        {"com": calculate_segment_com(L_SH, L_EL, 0.436), "weight": 0.028},
        {"com": calculate_segment_com(R_EL, R_WR, 0.430), "weight": 0.016},
        {"com": calculate_segment_com(L_EL, L_WR, 0.430), "weight": 0.016},
        {"com": R_WR, "weight": 0.006},
        {"com": L_WR, "weight": 0.006},
        {"com": calculate_segment_com(R_HP, R_KN, 0.433), "weight": 0.100},
        {"com": calculate_segment_com(L_HP, L_KN, 0.433), "weight": 0.100},
        {"com": calculate_segment_com(R_KN, R_AK, 0.433), "weight": 0.047},
        {"com": calculate_segment_com(L_KN, L_AK, 0.433), "weight": 0.047},
        {"com": R_AK, "weight": 0.016},
        {"com": L_AK, "weight": 0.016}
    ]
    total_com = np.zeros(2)
    total_weight = 0.0
    for seg in segments:
        total_com += seg["com"] * seg["weight"]
        total_weight += seg["weight"]
    return total_com / total_weight

def calculate_angle_3d_vectors(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6: return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)))

def plot_discus_circle_footwear(left_foot_history, right_foot_history, tracking_target):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#1e293b")
    ax.set_facecolor("#0f172a")
    circle_radius = 1.25
    circle = plt.Circle((0, 0), circle_radius, color='#38bdf8', fill=False, linestyle='--', linewidth=2, label='Discus Circle (R=1.25m)')
    ax.add_patch(circle)
    ax.axhline(0, color='#334155', linestyle=':', linewidth=1)
    ax.axvline(0, color='#334155', linestyle=':', linewidth=1)

    if left_foot_history:
        lf_arr = np.array(left_foot_history)
        ax.plot(lf_arr[:, 0] * 1.5, -lf_arr[:, 2] * 1.5, color='#22c55e', marker='o', markersize=5, linestyle='-', linewidth=1.5, label=f'Left Foot ({tracking_target})')
    if right_foot_history:
        rf_arr = np.array(right_foot_history)
        ax.plot(rf_arr[:, 0] * 1.5, -rf_arr[:, 2] * 1.5, color='#ef4444', marker='s', markersize=5, linestyle='-', linewidth=1.5, label=f'Right Foot ({tracking_target})')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Lateral Position (m)", color="#94a3b8")
    ax.set_ylabel("Depth Position (m)", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.set_title("Foot Landing Map inside Discus Circle", color="#38bdf8", fontsize=12)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="upper right")
    ax.grid(True, color="#1e293b", linestyle='-')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

def analyze_video(input_path, arm_len, d_mass, show_skel, show_com, show_axis, show_metrics, tracking_target):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_tfile.name, fourcc, fps, (width, height))

    prev_wrist_3d = None
    forces_per_frame = []
    axis_angles_per_frame = []
    speeds_per_frame = []
    distances_per_frame = []
    shoulder_wrist_angles_per_frame = []
    left_foot_history = []
    right_foot_history = []

    joint_map = {
        "足首 (Ankle)": (27, 28),
        "かかと (Heel)": (29, 30),
        "つま先 (Toe)": (31, 32)
    }
    left_idx, right_idx = joint_map.get(tracking_target, (27, 28))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            world_landmarks = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None

            if show_skel:
                for idx in [0, 11, 12, 13, 14, 15, 16]:
                    lm = landmarks[idx]
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    cv2.circle(frame, (cx, cy), 5, (56, 189, 248), -1)
            try:
                com_pos = None
                if show_com or show_axis:
                    com_pos = calculate_total_body_com(landmarks, width, height)
                if show_com and com_pos is not None:
                    cv2.circle(frame, (int(com_pos[0]), int(com_pos[1])), 8, (34, 197, 94), -1)
                    cv2.putText(frame, "CoM", (int(com_pos[0]) + 15, int(com_pos[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 1, cv2.LINE_AA)

                theta_deg = 0.0
                if show_axis and com_pos is not None:
                    nose_landmark = landmarks[0]
                    nose_pos = np.array([nose_landmark.x * width, nose_landmark.y * height])
                    cv2.line(frame, (int(com_pos[0]), int(com_pos[1])), (int(nose_pos[0]), int(nose_pos[1])), (251, 146, 60), 3)
                    body_axis = nose_pos - com_pos
                    ref_vector = np.array([0, -1])
                    norm_axis = np.linalg.norm(body_axis)
                    if norm_axis > 1e-6:
                        theta_deg = np.degrees(np.arccos(np.clip(np.dot(body_axis, ref_vector) / norm_axis, -1.0, 1.0)))
                        cv2.putText(frame, f"Axis Tilt: {theta_deg:.1f} deg", (int(com_pos[0]) + 15, int(com_pos[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (251, 146, 60), 1, cv2.LINE_AA)
                axis_angles_per_frame.append(theta_deg)

                current_speed, current_force, dist_axis_to_disc, shoulder_wrist_angle = 0.0, 0.0, 0.0, 0.0
                if world_landmarks:
                    left_foot_history.append([world_landmarks[left_idx].x, world_landmarks[left_idx].y, world_landmarks[left_idx].z])
                    right_foot_history.append([world_landmarks[right_idx].x, world_landmarks[right_idx].y, world_landmarks[right_idx].z])

                    r_wrist_3d = np.array([world_landmarks[16].x, world_landmarks[16].y, world_landmarks[16].z])
                    l_shoulder_3d = np.array([world_landmarks[11].x, world_landmarks[11].y, world_landmarks[11].z])
                    r_shoulder_3d = np.array([world_landmarks[12].x, world_landmarks[12].y, world_landmarks[12].z])
                    shoulder_line = r_shoulder_3d - l_shoulder_3d
                    nose_3d = np.array([world_landmarks[0].x, world_landmarks[0].y, world_landmarks[0].z])
                    hip_center_3d = (np.array([world_landmarks[23].x, world_landmarks[23].y, world_landmarks[23].z]) + np.array([world_landmarks[24].x, world_landmarks[24].y, world_landmarks[24].z])) / 2.0
                    axis_vector_3d = nose_3d - hip_center_3d
                    if np.linalg.norm(axis_vector_3d) > 1e-6:
                        ap = r_wrist_3d - hip_center_3d
                        dist_axis_to_disc = np.linalg.norm(np.cross(ap, axis_vector_3d)) / np.linalg.norm(axis_vector_3d)
                        dist_axis_to_disc = max(dist_axis_to_disc, 0.05)
                        projection = (np.dot(ap, axis_vector_3d) / np.dot(axis_vector_3d, axis_vector_3d)) * axis_vector_3d
                        shoulder_wrist_angle = calculate_angle_3d_vectors(shoulder_line, ap - projection)
                    if prev_wrist_3d is not None:
                        current_speed = np.linalg.norm(r_wrist_3d - prev_wrist_3d) * fps
                        current_force = d_mass * (current_speed ** 2) / dist_axis_to_disc
                    prev_wrist_3d = r_wrist_3d
                    if show_metrics:
                        cv2.putText(frame, f"Speed: {current_speed:.1f} m/s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Force: {current_force:.1f} N", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (244, 63, 94), 2, cv2.LINE_AA)
                speeds_per_frame.append(current_speed)
                forces_per_frame.append(current_force)
                distances_per_frame.append(dist_axis_to_disc)
                shoulder_wrist_angles_per_frame.append(shoulder_wrist_angle)
            except Exception as e:
                pass
        else:
            speeds_per_frame.append(0.0)
            forces_per_frame.append(0.0)
            distances_per_frame.append(0.0)
            shoulder_wrist_angles_per_frame.append(0.0)
            axis_angles_per_frame.append(0.0)
        out.write(frame)
    cap.release()
    out.release()
    pose.close()
    final_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', out_tfile.name, '-vcodec', 'libx264', '-f', 'mp4', final_tfile.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return final_tfile.name, forces_per_frame, axis_angles_per_frame, speeds_per_frame, distances_per_frame, shoulder_wrist_angles_per_frame, left_foot_history, right_foot_history

sidebar_area = st.sidebar
sidebar_area.header("⚙️ 解析オプション")
gender = sidebar_area.selectbox("性別 (投擲物重量)", ["男子 (2.0kg)", "女子 (1.0kg)"])
discus_mass = 2.0 if "男子" in gender else 1.0
height = sidebar_area.number_input("身長 (cm)", min_value=120.0, max_value=220.0, value=175.0, step=1.0)
arm_length = (height / 100.0) * 0.45

show_skel = sidebar_area.checkbox("上半身の主要関節点の描画", value=True)
show_com = sidebar_area.checkbox("重心 (CoM) の描画", value=True)
show_axis = sidebar_area.checkbox("体軸 (Body Axis) の描画", value=True)
show_metrics = sidebar_area.checkbox("遠心力等の推移グラフを表示する", value=True)

st.sidebar.markdown("---👣 足部トラッキング")
show_foot_map = sidebar_area.checkbox("サークル内足裏マップを描画する", value=True)
tracking_target = sidebar_area.selectbox("👣 追跡する足の部位", ["足首 (Ankle)", "かかと (Heel)", "つま先 (Toe)"], disabled=not show_foot_map)

st.subheader("📹 分析用スイング動画のアップロード")
st.info("📱 スマートフォン対応UI | 動画は最大2個まで同時選択・自動並列比較が可能です。")

uploaded_files = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 2:
        st.error("⚠️ 同時に比較できる動画は最大2個までに制限されています。自動的に最初の2つの動画を対象に比較を行います。")
        uploaded_files = uploaded_files[:2]

    if len(uploaded_files) == 1:
        file = uploaded_files[0]
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tf.write(file.read())
        with st.spinner("バイオメカニクスデータを抽出中..."):
            out_path, forces, axis_angles, speeds, distances, shoulder_wrist_angles, lf_hist, rf_hist = analyze_video(
                tf.name, arm_length, discus_mass, show_skel, show_com, show_axis, show_metrics, tracking_target
            )
        st.success("解析が完了しました！")
        col_v, col_g = st.columns([1, 1])
        with col_v:
            st.video(out_path)
        with col_g:
            st.markdown("### 📈 バイオメカニクス推移")
            df_data = pd.DataFrame({"Frame": range(len(forces))})
            if show_metrics and any(forces):
                st.metric(label="🏆 最大瞬間遠心力", value=f"{max(forces):.1f} N", delta=f"最大速度: {max(speeds):.1f} m/s")
                df_data["Centrifugal Force (N)"] = forces
                df_data["Wrist Speed (m/s)"] = speeds
                st.line_chart(df_data.set_index("Frame")[["Centrifugal Force (N)", "Wrist Speed (m/s)"]])

        if show_foot_map:
            st.markdown("### 👣 サークル内の足位置マップ (Footwork Analysis)")
            foot_col1, foot_col2 = st.columns([1, 1])
            with foot_col1:
                plot_buf = plot_discus_circle_footwear(lf_hist, rf_hist, tracking_target)
                st.image(plot_buf, caption=f"サークル内足裏設置軌跡（緑:左足, 赤:右足）", use_container_width=True)
            with foot_col2:
                st.markdown(f"公式規格サークル(直径2.5m)内における、選手足部の動き（**{tracking_target}**）の遷移軌跡です。")
                st.download_button(label="📁 プロット画像（PNG）をダウンロード", data=plot_buf, file_name="footwork_analysis.png", mime="image/png")

    elif len(uploaded_files) == 2:
        st.success("📊 2つのスイングデータを比較分析します。")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🔵 動画 A: " + uploaded_files[0].name)
            tf_a = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tf_a.write(uploaded_files[0].read())
            with st.spinner("動画 A の解析中..."):
                out_a, forces_a, _, speeds_a, _, _, lf_a, rf_a = analyze_video(
                    tf_a.name, arm_length, discus_mass, show_skel, show_com, show_axis, show_metrics, tracking_target
                )
            st.video(out_a)
            if any(forces_a):
                st.metric("動画 A 最大遠心力", f"{max(forces_a):.1f} N", f"{max(speeds_a):.1f} m/s")
                if show_foot_map:
                    plot_a = plot_discus_circle_footwear(lf_a, rf_a, tracking_target)
                    st.image(plot_a, caption="動画A: 足位置マップ", use_container_width=True)

        with col_right:
            st.markdown("### 🟢 動画 B: " + uploaded_files[1].name)
            tf_b = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tf_b.write(uploaded_files[1].read())
            with st.spinner("動画 B の解析中..."):
                out_b, forces_b, _, speeds_b, _, _, lf_b, rf_b = analyze_video(
                    tf_b.name, arm_length, discus_mass, show_skel, show_com, show_axis, show_metrics, tracking_target
                )
            st.video(out_b)
            if any(forces_b):
                st.metric("動画 B 最大遠心力", f"{max(forces_b):.1f} N", f"{max(speeds_b):.1f} m/s")
                if show_foot_map:
                    plot_b = plot_discus_circle_footwear(lf_b, rf_b, tracking_target)
                    st.image(plot_b, caption="動画B: 足位置マップ", use_container_width=True)
else:
    st.warning("分析を開始するには、動画ファイルをアップロードしてください（最大2つまで）。")
