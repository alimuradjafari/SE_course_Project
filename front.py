import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
from image_module import show_image_detection
from video_module import show_video_detection

# -----------------------------
# Initialize the inference client
# -----------------------------
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Ma3NLpwSqV6e5Bf0VPeM"  
)


st.markdown("""
<div id="home-section"></div>
""", unsafe_allow_html=True)
# Page config >>>>
st.set_page_config(
    page_title="Sky Guard - Intelligent Drone Detection",
    page_icon="images/logo.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Reset padding */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Navigation bar */
    .navbar {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        padding: 1rem 3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
    }
    
    .logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    .nav-links {
        display: flex;
        gap: 2.5rem;
        list-style: none;
        margin: 0;
        padding: 0;
    }
    
    .nav-links a {
        color: #e0e0e0;
        text-decoration: none;
        font-size: 1rem;
        transition: color 0.3s;
    }
    
    .nav-links a:hover {
        color: #00d4ff;
    }
    
    /* Hero section */
    .hero {
        min-height: 100vh;
        background: linear-gradient(rgba(15, 20, 25, 0.85), rgba(15, 20, 25, 0.85)),
                    url('https://images.unsplash.com/photo-1473968512647-3e447244af8f?q=80&w=2070') center/cover;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 2rem;
        margin-top: 70px;
    }
    
    .badge {
        background: rgba(0, 212, 255, 0.15);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff;
        padding: 0.6rem 1.5rem;
        border-radius: 50px;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 1rem 0;
        line-height: 1.2;
    }
    
    .hero p {
        font-size: 1.4rem;
        color: #b0b0b0;
        margin-top: 1.5rem;
        max-width: 700px;
    }
    
    @media (max-width: 768px) {
        .navbar {
            padding: 1rem;
        }
        
        .nav-links {
            gap: 1rem;
            font-size: 0.9rem;
        }
        
        .hero h1 {
            font-size: 2rem;
        }
        
        .hero p {
            font-size: 1.1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Navigation bar
st.markdown("""
<div class="navbar">
    <div class="logo">
        <div style='font-size:15px'; class="logo-icon">AHA</div>
        <span>Sky Guard</span>
    </div>
    <div class="nav-links">
        <a href="#home-section">Home</a>
        <a href="#feature-section">Features</a>
        <a href="#team-section">Team</a>
        <a href="#contact-section">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <div class="badge">Powered by AI & YOLOv8</div>
    <h1>The Intelligent Drone Detection System</h1>
    <p>Detect drones instantly using AI technology</p>
</div>
""", unsafe_allow_html=True)
    
 
# IMAGE DETECTION SECTION
 
show_image_detection(client)

 
# VIDEO DETECTION SECTION
 
show_video_detection(client)
st.markdown("<br>", unsafe_allow_html=True)
 
# DEMONSTRATION SECTION
 
 
st.markdown("""
<div id="feature-section"></div>
""", unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    
    st.markdown(
        """
        <div style='font-size:22px; font-weight:600; text-align:center'>
            <h5>Project Detail</h5>
            Built using
            <span style='color:#4C7CFF;'> YOLOV8 model </span>
            for accurate drone detection
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    img1 = Image.open("images/drone1.jpeg").resize((400, 250))
    img2 = Image.open("images/drone2.gif").resize((400, 250))
    img3 = Image.open("images/drone3.webp").resize((400, 250))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img1, caption="Detected Drone", use_container_width=True)
    with col2:
        st.image(img2, caption="Detected Drone", use_container_width=True)
    with col3:
        st.image(img3, caption="Detected Drone", use_container_width=True)

 

st.markdown("""
<div id="team-section"></div>
""", unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    st.markdown(
        """
        <div style='font-size:22px; font-weight:600; text-align:center'>
            <h6>Meet Our Team</h6>
            <p style ='font-size:12px'>Talented students working together to build innovative AI solutions</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Four boxes for team members
    team = [
        
        {"name": "Osama Alam", "role": "Supervisor","about":"   CEO EMRChain Company"},
        {"name": "Ahmad Riasit", "role": "YOLOV8 Model Trainer", "about": "Software Engineering Student"},
        {"name": "Ali Murad Jafari", "role": "Frontend Developer", "about": "Software Engineering Student"},
        {"name": "Hamza Khalid", "role": "Backend Developer", "about": "Software Engineering Student"},
    ]

    col1, col2, col3, col4 = st.columns(4)
    for col, member in zip([col1, col2, col3, col4], team):
        col.markdown(f"""
            <div style="
                border: 2px solid #4C7CFF; 
                border-radius: 10px; 
                padding: 15px; 
                text-align: center;
                margin: auto;
            ">
                <h6 style ='text-align: left'>{member['name']}</h6>
                <p style ='text-align: left;color:#00FFFF;font-size:10px'>{member['role']}</p>
                <p style ='text-align: left;font-size:10px'>{member['about']}</p>
            </div>
        """, unsafe_allow_html=True)

#footer

st.markdown("<br><br>", unsafe_allow_html=True)  # spacing before footer

st.markdown("""
<div id="contact-section"></div>
""", unsafe_allow_html=True)

# Main footer container
st.markdown("<hr style='border:1px solid light-white;'>", unsafe_allow_html=True)  # separator
with st.container():
    # Top row: Logos and info
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # st.image("images/logo.png", width=50)
        st.markdown("""
            <h2 style='font-size:20px; text-align: center;color:#00FFFF '>Sky Guard</h2>  
            <p style= 'font-size:12px; text-align:center;'>Advanced AI-powered drone detection system using YOLOv8 <br>technology for real-time threat identification.</p>
        """, unsafe_allow_html=True)
    
    with col2:
        
        st.markdown("""
            <p style= 'font-size:12px'>Developed in Collaboration With</p> 
            <h2 style='font-size:20px;color: #00FFFF'>EMR Chain</h2>
            <p style= 'font-size:12px'>Leading technology solutions provider specializing in <br>blockchain and AI innovations.</p>
            <h6>Contact Details</h6>
            <p style = 'font-size: 12px'>+92 312 5695868<br>info@emrchains.com<br>NSTP, NUST, H-12, Islamabad</p>
        """, unsafe_allow_html=True)
    

    # Bottom row: copyright and links
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.markdown("<p style='margin-left: 100px;font-size: 8px'>© 2025 Sky Guard. All rights reserved.</p>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<p style='margin-left: 400px;font-size: 8px'>Website developed in collaboration with <strong style='color:#00FFFF'>EMRChain</strong></p>"
        
        
        , unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True) 