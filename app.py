import pandas as pd
import streamlit as st
import sqlite3
import uuid
import random
import time
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# -------- PDF SAFE IMPORT --------
try:
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Faculty Research System", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp { background-color: #0b1f3a; }
h1, h2, h3 { color: white; }
section[data-testid="stSidebar"] { background-color: #08162b; }
.stButton>button {
    background-color: #0b3d91;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("faculty.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faculty_data (
faculty_id TEXT, faculty TEXT, title TEXT, journal TEXT, status TEXT)
""")
conn.commit()

# ---------------- OTP FUNCTIONS ----------------
def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp(email, otp):
    try:
        sender = "your_email@gmail.com"
        password = "your_app_password"

        msg = MIMEText(f"Your OTP is {otp}")
        msg['Subject'] = "Login OTP"
        msg['From'] = sender
        msg['To'] = email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
    except:
        pass

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "otp" not in st.session_state:
    st.session_state.otp = ""
if "otp_time" not in st.session_state:
    st.session_state.otp_time = 0

# ---------------- OTP LOGIN ----------------
if not st.session_state.logged_in:
    st.title("📧 OTP Login")

    email = st.text_input("Enter Email")
    user_otp = st.text_input("Enter OTP")

    # SEND OTP
    if st.button("Send OTP"):
        otp = generate_otp()
        st.session_state.otp = otp
        st.session_state.otp_time = time.time()
        send_otp(email, otp)
        st.toast("OTP Sent ✅")

    # RESEND OTP
    if st.button("Resend OTP"):
        if time.time() - st.session_state.otp_time > 30:
            otp = generate_otp()
            st.session_state.otp = otp
            st.session_state.otp_time = time.time()
            send_otp(email, otp)
            st.toast("OTP Resent 🔁")
        else:
            st.toast("Wait 30 sec ⏳")

    # VERIFY OTP
    if st.button("Verify OTP"):
        if time.time() - st.session_state.otp_time < 120:
            if user_otp == st.session_state.otp:
                st.session_state.logged_in = True
                st.rerun()
        else:
            st.toast("OTP Expired ⌛")

    st.stop()

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_sql("SELECT * FROM faculty_data", conn)
    if not df.empty:
        df.columns = df.columns.str.lower()
        df["text"] = df["title"] + " " + df["journal"]
    return df

data = load_data()

# ---------------- MODEL ----------------
if not data.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["text"])
    model = LogisticRegression(max_iter=200)
    model.fit(X, data["status"])

# ---------------- FUNCTIONS ----------------
def insert_data(n,t,j,s):
    cursor.execute("INSERT INTO faculty_data VALUES (?,?,?,?,?)",
                   ("FID-"+str(uuid.uuid4())[:8], n,t,j,s))
    conn.commit()

def insert_csv(df):
    df.columns = df.columns.str.lower().str.strip()
    faculty_col = next((c for c in df.columns if "faculty" in c), None)
    title_col = next((c for c in df.columns if "title" in c), None)
    journal_col = next((c for c in df.columns if "journal" in c), None)
    status_col = next((c for c in df.columns if "status" in c), None)

    if not all([faculty_col, title_col, journal_col, status_col]):
        return

    for _, row in df.iterrows():
        try:
            insert_data(
                str(row[faculty_col]),
                str(row[title_col]),
                str(row[journal_col]),
                str(row[status_col])
            )
        except:
            pass

def delete_data(fid):
    cursor.execute("DELETE FROM faculty_data WHERE faculty_id=?", (fid,))
    conn.commit()

def predict_status(t,j):
    return model.predict(vectorizer.transform([t+" "+j]))[0]

def find_similar(q):
    sim = cosine_similarity(vectorizer.transform([q]), X)[0]
    return data[sim>0.3]

# ---------------- SIDEBAR ----------------
menu=st.sidebar.radio("📌 Menu",[
    "Prediction","Search","Analytics","Database","Download"
])

st.markdown("<h1 style='text-align:center;'>📚 Faculty Research System</h1>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if menu=="Prediction":
    t=st.text_input("Title")
    j=st.text_input("Journal")

    if st.button("Predict"):
        if not data.empty and t and j:
            st.success(predict_status(t,j))

# ---------------- SEARCH ----------------
elif menu=="Search":
    name=st.text_input("Faculty Name")
    fid=st.text_input("Faculty ID")

    if st.button("Search"):
        df=data
        if name:
            df=df[df["faculty"].str.contains(name,case=False)]
        if fid:
            df=df[df["faculty_id"].str.contains(fid,case=False)]
        st.dataframe(df)

    q=st.text_input("Search Title")

    if st.button("Find Similar"):
        if not data.empty:
            st.dataframe(find_similar(q))

# ---------------- ANALYTICS ----------------
elif menu=="Analytics":
    if not data.empty:
        st.metric("Total Records",len(data))
        st.metric("Unique Faculty",data["faculty"].nunique())
        st.bar_chart(data["status"].value_counts())
        st.line_chart(data["status"].value_counts())
        st.dataframe(data)

# ---------------- DATABASE ----------------
elif menu=="Database":
    n=st.text_input("Name")
    t=st.text_input("Title")
    j=st.text_input("Journal")
    s=st.selectbox("Status",["Published","Accepted","Rejected","Under Review"])

    if st.button("Add"):
        if n and t and j:
            insert_data(n,t,j,s)
            st.rerun()

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df)
        if st.button("Insert CSV"):
            insert_csv(df)
            st.rerun()

    fid=st.text_input("Delete ID")
    if st.button("Delete"):
        delete_data(fid)
        st.rerun()

    st.dataframe(data)

# ---------------- DOWNLOAD ----------------
elif menu=="Download":
    st.download_button("Download CSV", data.to_csv(index=False), "data.csv")

    if PDF_AVAILABLE:
        if st.button("Generate PDF"):
            path="/mnt/data/report.pdf"
            with open(path,"wb") as f:
                f.write(b"PDF")
            st.download_button("Download PDF", open(path,"rb"), "report.pdf")
