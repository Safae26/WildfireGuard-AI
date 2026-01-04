import streamlit as st
import tempfile
import cv2
import numpy as np
import time
import tensorflow as tf  

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="WildfireGuard AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS ---
def local_css():
    st.markdown("""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');

        /* --- GLOBAL COLORS & RESET --- */
        :root {
            --primary-neon: #FF2A2A;
            --secondary-neon: #FF7B00;
            --bg-dark: #050505;
            --sidebar-bg: #0a0a0a;
            --text-color: #E0E0E0;
        }

        .stApp {
            background-color: var(--bg-dark);
            background-image: radial-gradient(circle at 50% 50%, #1a0505 0%, #000000 100%);
            color: var(--text-color);
            font-family: 'Rajdhani', sans-serif;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
            backdrop-filter: blur(5px); 
        }

        /* --- SIDEBAR & SLIDERS STYLING --- */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid #333;
        }

        /* --- TYPOGRAPHY --- */
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            background: linear-gradient(90deg, var(--primary-neon), var(--secondary-neon));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(255, 42, 42, 0.5);
        }
        
        p, li {
            color: #cfcfcf !important;
            font-size: 1.1rem;
            line-height: 1.6;
        }

        /* --- CUSTOM BUTTONS --- */
        .stButton > button {
            background: transparent;
            border: 2px solid var(--primary-neon);
            color: var(--primary-neon);
            border-radius: 0px; 
            font-family: 'Orbitron', sans-serif;
            font-weight: bold;
            transition: all 0.4s ease;
            text-transform: uppercase;
            box-shadow: 0 0 10px rgba(255, 42, 42, 0.2);
            width: 100%;
        }
        .stButton > button:hover {
            background: var(--primary-neon);
            color: black;
            box-shadow: 0 0 25px var(--primary-neon);
            transform: translateY(-2px);
        }

        /* --- FOOTER SIGNATURE --- */
        .footer-safae {
            font-family: 'Orbitron', sans-serif;
            color: #444 !important;
            text-align: center;
            margin-top: 50px;
            font-size: 0.8rem !important;
            letter-spacing: 1px;
            opacity: 0.7;
        }
        .footer-safae span { color: var(--primary-neon); }

        /* --- INFO CARDS --- */
        .info-card {
            background: rgba(20, 20, 20, 0.6);
            border-left: 3px solid var(--secondary-neon);
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 0 10px 10px 0;
            transition: transform 0.3s ease;
        }
        
        /* --- OLD RADAR STYLE (KEPT FOR HOME PAGE) --- */
        .radar {
            width: 150px; height: 150px;
            background: radial-gradient(circle, rgba(255,42,42,0.1) 0%, rgba(255,42,42,0) 70%);
            border: 2px solid var(--primary-neon);
            border-radius: 50%;
            position: relative; margin: 0 auto;
            box-shadow: 0 0 30px rgba(255,42,42,0.4);
            animation: pulse 2s infinite;
            display: flex; align-items: center; justify-content: center;
        }
        .radar::after {
            content: ''; width: 100%; height: 100%; border-radius: 50%;
            border-top: 2px solid rgba(255, 255, 255, 0.8);
            position: absolute; animation: spin 3s linear infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 42, 42, 0.7); }
            70% { transform: scale(1.05); box-shadow: 0 0 0 20px rgba(255, 42, 42, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 42, 42, 0); }
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* --- NEW SCANNER STYLES (ADDED) --- */
        .scan-container {
            position: relative;
            border: 2px solid #333;
            height: 400px;
            background: #000;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(0,0,0,0.8);
            border-radius: 5px;
        }
        .scan-line {
            position: absolute;
            width: 100%;
            height: 5px;
            background: var(--primary-neon);
            box-shadow: 0 0 20px var(--primary-neon);
            opacity: 0.6;
            animation: scan 3s linear infinite;
            z-index: 5;
        }
        .grid-overlay {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: 
                linear-gradient(rgba(255, 42, 42, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 42, 42, 0.1) 1px, transparent 1px);
            background-size: 40px 40px;
            z-index: 2;
        }
        @keyframes scan {
            0% { top: 0%; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }

        /* --- MOBILE RESPONSIVENESS --- */
        @media only screen and (max-width: 768px) {
            h1 { font-size: 1.8rem !important; }
            .radar { width: 100px; height: 100px; }
            .radar span { font-size: 2rem !important; }
            .scan-container { height: 250px !important; }
            div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
            .block-container { padding-top: 2rem !important; }
        }
                /* --- NOUVEAU CSS POUR ABOUT PAGE --- */
        .mission-header {
            border-bottom: 2px solid var(--primary-neon);
            padding-bottom: 10px;
            margin-bottom: 20px;
            letter-spacing: 4px;
            color: #fff;
            text-shadow: 0 0 10px var(--primary-neon);
        }

        /* Cadres technologiques (Tech Stack) */
        .tech-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .tech-badge {
            background: rgba(0,0,0,0.6);
            border: 1px solid var(--secondary-neon);
            color: var(--secondary-neon);
            padding: 5px 15px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8rem;
            text-transform: uppercase;
            border-radius: 4px;
            transition: 0.3s;
            box-shadow: 0 0 5px rgba(255, 123, 0, 0.2);
        }
        .tech-badge:hover {
            background: var(--secondary-neon);
            color: black;
            box-shadow: 0 0 15px var(--secondary-neon);
            transform: scale(1.05);
            cursor: default;
        }

        /* Animation du "Noyau" (Core Visual) */
        .core-visual {
            width: 100%;
            height: 250px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            background: radial-gradient(circle, rgba(255,42,42,0.05) 0%, transparent 70%);
            border: 1px dashed #333;
            border-radius: 10px;
        }
        .core-circle {
            position: absolute;
            border-radius: 50%;
            border: 2px solid transparent;
            border-top: 2px solid var(--primary-neon);
            border-bottom: 2px solid var(--primary-neon);
            animation: spin-core 4s linear infinite;
        }
        .c1 { width: 100px; height: 100px; animation-duration: 4s; opacity: 0.8; }
        .c2 { width: 140px; height: 140px; animation-duration: 6s; opacity: 0.5; border-left: 2px solid var(--secondary-neon); border-right: 2px solid var(--secondary-neon); border-top: transparent; border-bottom: transparent;}
        .c3 { width: 60px; height: 60px; background: var(--primary-neon); opacity: 0.2; box-shadow: 0 0 20px var(--primary-neon); animation: pulse-core 2s infinite; border: none;}

        @keyframes spin-core { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes pulse-core { 0% { transform: scale(0.8); opacity: 0.2; } 50% { transform: scale(1.1); opacity: 0.5; } 100% { transform: scale(0.8); opacity: 0.2; } }
    
        /* --- FILE UPLOADER DARK MODE --- */
        
        /* La zone de dépôt principale */
        [data-testid="stFileUploaderDropzone"] {
            background-color: rgba(10, 10, 10, 0.8) !important;
            border: 1px dashed #444 !important;
            border-radius: 0px !important;
        }

        /* Effet au survol de la souris */
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--primary-neon) !important;
            background-color: rgba(255, 42, 42, 0.1) !important; /* Lueur rouge */
        }

        /* Le texte "Drag and drop file here" */
        div[data-testid="stFileUploaderDropzoneInstructions"] div {
            color: #E0E0E0 !important;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1rem;
        }

        /* Le petit texte "Limit 200MB..." */
        div[data-testid="stFileUploaderDropzoneInstructions"] small {
            color: #888 !important;
        }

        /* Le bouton "Browse files" à l'intérieur */
        [data-testid="stFileUploader"] button {
            border: 1px solid var(--primary-neon) !important;
            color: var(--primary-neon) !important;
            background-color: transparent !important;
            font-family: 'Orbitron', sans-serif !important;
            border-radius: 0px !important;
        }
        
        /* Survol du bouton Browse files */
        [data-testid="stFileUploader"] button:hover {
            background-color: var(--primary-neon) !important;
            color: black !important;
            box-shadow: 0 0 10px var(--primary-neon);
        }
    </style>
    """, unsafe_allow_html=True)
local_css()

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# --- HOME PAGE ---
def show_home():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col2:
        st.markdown("""
            <div style="display: flex; justify-content: center; margin-bottom: 20px; margin-top: 20px;">
                <div class="radar">
                    <span style="font-size: 3rem;">🔥</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center;'>WILDFIRE GUARD AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 0.9rem;'>AUTONOMOUS SATELLITE SURVEILLANCE SYSTEM</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("LAUNCH DASHBOARD", use_container_width=True):
                navigate_to("Dashboard")
        with b2:
            if st.button("ℹ️ MISSION BRIEFING", use_container_width=True):
                navigate_to("About")
        
        st.markdown("<p class='footer-safae'>SYSTEM DEVELOPED BY <span>SAFAE</span></p>", unsafe_allow_html=True)

# --- ABOUT PAGE ---
def show_about():
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Header de navigation
    c_nav, c_title = st.columns([1, 5])
    with c_nav:
        if st.button("⬅ BACK", key="back_btn"):
            navigate_to("Home")
    with c_title:
        st.markdown("<h1 style='margin-top: -10px;'>// MISSION BRIEFING</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    col_info, col_sys = st.columns([3, 2], gap="large")

    with col_info:
        st.markdown("<h3 class='mission-header'>01. THREAT ANALYSIS</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card" style="border-left: 3px solid #FF2A2A;">
            <strong style="color:white;">OBJECTIVE:</strong> Early detection of forest fires.<br>
            <strong style="color:white;">CONTEXT:</strong> Wildfires are accelerating due to climate change. Traditional manual surveillance is slow and error-prone.
            <br><br>
            <em style="opacity:0.7;">"Time is the only resource we cannot recover."</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<h3 class='mission-header'>02. OPERATIONAL SPECS</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <strong style="color:white;">SOLUTION:</strong> <b>WildfireGuard AI</b> deploys autonomous Computer Vision agents to analyze satellite feeds in real-time.
            <br><br>
            It identifies smoke signatures and thermal anomalies instantly.
        </div>
        """, unsafe_allow_html=True)

    with col_sys:
        st.markdown("<h3 class='mission-header' style='text-align:center;'>SYSTEM CORE</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="core-visual">
            <div class="core-circle c1"></div>
            <div class="core-circle c2"></div>
            <div class="core-circle c3"></div>
            <div style="position: absolute; font-family: 'Orbitron'; font-size: 0.8rem; color: white; letter-spacing: 2px;">AI ONLINE</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Section Tech Stack (Les badges)
        st.markdown("<h4 style='text-align:center; color:#888; margin-bottom:10px;'>ARCHITECTURAL MODULES</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class="tech-container" style="justify-content: center;">
            <span class="tech-badge">Python 3.10</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">OpenCV</span>
            <span class="tech-badge">Keras</span>
            <span class="tech-badge">Pandas</span>
        </div>
        """, unsafe_allow_html=True)

    # Footer Signature
    st.markdown("---")
    st.markdown("<p class='footer-safae'>SYSTEM ARCHITECT: <span>SAFAE</span> | CLASSIFICATION: TOP SECRET</p>", unsafe_allow_html=True)

def focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss_fn(y_true, y_pred):
        # S'assurer que y_true a la même shape que y_pred
        y_true = tf.expand_dims(y_true, axis=-1) if len(y_true.shape) < len(y_pred.shape) else y_true
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = K.pow(1 - p_t, gamma)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        return K.mean(alpha_factor * modulating_factor * cross_entropy, axis=-1)
    
    return focal_loss_fn

# --- FONCTION DE CHARGEMENT DU MODÈLE (MISE EN CACHE) ---
@st.cache_resource
def load_model():
    try:
        # 1. On instancie la perte (on crée la fonction interne)
        # On utilise les valeurs par défaut ou celles de votre notebook
        loss_function_instance = focal_loss(gamma=2.0, alpha=0.75)

        # 2. On charge le modèle en lui passant cette fonction
        # La clé 'focal_loss_fn' correspond au nom que Keras cherche (celui de l'erreur)
        model = tf.keras.models.load_model(
            'wildfire_detection_model.keras', 
            custom_objects={'focal_loss_fn': loss_function_instance}
        )
        return model
    except Exception as e:
        st.error(f"Erreur critique de chargement : {e}")
        return None

# --- FONCTION DE PRÉTRAITEMENT ---
def preprocess_frame(frame, target_size):
    # On s'assure que target_size est bien (Hauteur, Largeur)
    img = cv2.resize(frame, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- DASHBOARD PAGE ---
import os 
def show_dashboard():
    model = load_model()

    with st.sidebar:
        st.markdown("### ⚙️ SETTINGS")
        st.markdown("---")
        conf_threshold = st.slider("CONFIDENCE THRESHOLD", 0.0, 1.0, 0.5)
        
        # Le toggle pour activer/désactiver la surveillance
        run_detection = st.toggle("ACTIVATE SURVEILLANCE", value=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅ EXIT"):
            navigate_to("Home")
        
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #555; font-size: 0.8rem; font-family: Orbitron;'>MADE BY <span style='color: #FF2A2A;'>SAFAE</span></div>", unsafe_allow_html=True)

    st.markdown("## 📡 TELEMETRY")
    m1, m2 = st.columns(2)
    with m1: metric_area = st.empty()
    with m2: metric_fire = st.empty()

    # Initialisation des valeurs
    metric_area.metric("AREA", "0 km²", "WAITING")
    metric_fire.metric("FIRES", "0", "SAFE")

    st.markdown("---")
    # ---------------------------------------------------

    col_video, col_terminal = st.columns([2, 1], gap="medium")
    
    with col_terminal:
        st.markdown("### 📟 LOGS")
        log_container = st.empty()
        # Initialisation des logs
        log_html_start = """
        <div class="terminal-container" style="background:#000; border:1px solid #333; padding:15px; font-family:'Courier New'; color:#00ff41; height: 400px; overflow-y:auto;">
            <span style='opacity:0.5'>> System initialized...</span><br>
            <span style='opacity:0.5'>> AI Model Loaded: wildfire_detection.keras</span>
        </div>
        """
        log_container.markdown(log_html_start, unsafe_allow_html=True)

    with col_video:
        st.markdown("### 🔭 VISUAL SURVEILLANCE")
        video_file = st.file_uploader("Stream Input", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
        video_placeholder = st.empty()

        if video_file is None:
            # Animation d'attente (Scanner)
            video_placeholder.markdown(f"""
            <div class="scan-container">
                <div class="grid-overlay"></div>
                <div class="scan-line"></div>
                <div style="z-index: 3; text-align: center;">
                    <h3 style="color: #444; margin: 0; font-family: 'Orbitron';">WAITING FOR SIGNAL</h3>
                    <p style="font-family: monospace; color: #FF2A2A;">SYSTEM STANDBY</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Vérification du modèle
            if model is not None:
                input_shape = model.input_shape
                # Gestion robuste de la taille d'entrée
                if len(input_shape) > 2 and input_shape[1] is not None:
                    target_size = (input_shape[1], input_shape[2])
                else:
                    target_size = (224, 224) # Fallback standard
            
            # Gestion Fichier Temporaire
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(video_file.read())
            tfile.close() 
            tfile_path = tfile.name
            
            vf = cv2.VideoCapture(tfile_path)
            
            if not vf.isOpened():
                st.error("Error: Could not open video file.")
            else:
                if run_detection:
                    st.toast("System Online. Analyzing feed...", icon="🔥")
                
                logs_history = ["> Connecting to satellite...", "> Feed received."]
                frame_skip = 2 
                frame_count = 0

                # Boucle principale
                while vf.isOpened():
                    # Si l'utilisateur désactive le toggle, on arrête la boucle
                    if not run_detection:
                        video_placeholder.markdown("### 🛑 SURVEILLANCE PAUSED")
                        break

                    ret, frame = vf.read()
                    if not ret:
                        break # Fin de la vidéo
                    
                    frame_count += 1
                    
                    # On ne fait la prédiction que tous les X frames pour la performance
                    if frame_count % frame_skip == 0:
                        
                        # Copie pour le dessin (bounding box)
                        display_frame = frame.copy()
                        
                        is_fire = False
                        confidence = 0.0

                        try:
                            processed_frame = preprocess_frame(frame, target_size=target_size)
                            prediction = model.predict(processed_frame, verbose=0)
                            
                            # Gestion binaire vs catégorique
                            if prediction.shape[-1] == 1:
                                confidence = float(prediction[0][0])
                            else:
                                confidence = float(prediction[0][0]) 
                            
                            # Seuil de détection
                            is_fire = confidence > conf_threshold

                        except Exception as e:
                            print(f"Inference error: {e}")

                        # --- DESSIN SUR L'IMAGE ---
                        if is_fire:
                            # Rectangle Rouge
                            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 10)
                            label = f"FIRE DETECTED ({confidence*100:.1f}%)"
                            # Fond noir pour le texte
                            cv2.rectangle(display_frame, (20, 10), (450, 60), (0,0,0), -1) 
                            cv2.putText(display_frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                            # Mise à jour des logs
                            if frame_count % (frame_skip * 5) == 0:
                                current_time = time.strftime("%H:%M:%S")
                                logs_history.append(f"> <span style='color:red'>ALERT: CONF {confidence:.2f} AT {current_time}</span>")
                                if len(logs_history) > 8: logs_history.pop(0)
                                
                                log_text = "<br>".join(logs_history)
                                log_container.markdown(f"""
                                <div class="terminal-container" style="background:#000; border:1px solid #333; padding:15px; font-family:'Courier New'; color:#00ff41; height: 400px; overflow-y:auto;">
                                    {log_text}<br>> <span class="blink">_</span>
                                </div>""", unsafe_allow_html=True)
                                
                                metric_fire.metric("FIRES", "CRITICAL", "DETECTED", delta_color="inverse")
                        
                        elif not is_fire and frame_count % 30 == 0:
                             metric_fire.metric("FIRES", "0", "SAFE")

                        # Affichage de l'image
                        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)

            vf.release()
            
            # --- NETTOYAGE ---
            try:
                os.remove(tfile_path)
            except:
                pass

# --- ROUTER ---
if st.session_state.page == "Home":
    show_home()
elif st.session_state.page == "Dashboard":
    show_dashboard()
elif st.session_state.page == "About":
    show_about()