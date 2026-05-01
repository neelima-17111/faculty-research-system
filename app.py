import pandas as pd
import streamlit as st
import sqlite3
import uuid
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

# ---------------- NAVY BLUE CSS ----------------
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
.stButton>button:hover { background-color: #1456c3; }
.stTextInput>div>div>input { border-radius: 8px; }
[data-testid="metric-container"] {
    background-color: #122b52;
    padding: 10px;
    border-radius: 10px;
    color: white;
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

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
username TEXT, password TEXT)
""")
conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- AUTH ----------------
def login(u,p):
    return cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p)).fetchone()

def signup(u,p):
    cursor.execute("INSERT INTO users VALUES (?,?)",(u,p))
    conn.commit()

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:
    st.title("📚 Faculty Research System")

    mode = st.radio("Choose", ["Login","Signup"])
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if mode=="Login":
        if st.button("Login"):
            if login(u,p):
                st.session_state.logged_in=True
                st.rerun()
    else:
        if st.button("Signup"):
            if u and p:
                signup(u,p)
                st.session_state.logged_in=True
                st.rerun()

    st.stop()

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_sql("SELECT * FROM faculty_data", conn)
    if not df.empty:
        df.columns = df.columns.str.lower()
        df["text"] = df["title"].astype(str) + " " + df["journal"].astype(str)
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
        return  # silent skip

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
    return cursor.rowcount>0

def predict_status(t,j):
    return model.predict(vectorizer.transform([t+" "+j]))[0]

def find_similar(q):
    sim = cosine_similarity(vectorizer.transform([q]), X)[0]
    return data[sim>0.3]

# ---------------- PDF ----------------
def generate_pdf():
    path="/mnt/data/report.pdf"
    doc=SimpleDocTemplate(path)
    styles=getSampleStyleSheet()
    elements=[]

    try:
        elements.append(Image("logo.png", width=120, height=60))
    except:
        pass

    elements.append(Paragraph("Faculty Research Report", styles["Title"]))
    elements.append(Spacer(1,12))

    table_data=[["ID","Faculty","Title","Journal","Status"]]

    for _,r in data.iterrows():
        table_data.append([r['faculty_id'], r['faculty'], r['title'], r['journal'], r['status']])

    table=Table(table_data)

    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('BACKGROUND',(0,1),(-1,-1),colors.lightgrey)
    ]))

    elements.append(table)
    doc.build(elements)

    return path

# ---------------- SIDEBAR ----------------
menu=st.sidebar.radio("📌 Menu",[
    "Prediction","Search","Analytics","Database","Download"
])

st.markdown("<h1 style='text-align:center;'>📚 Faculty Research System</h1>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if menu=="Prediction":
    st.subheader("🔮 Predict Status")

    t=st.text_input("Title")
    j=st.text_input("Journal")

    if st.button("🔮 Predict"):
        if not data.empty and t and j:
            st.success(predict_status(t,j))

# ---------------- SEARCH ----------------
elif menu=="Search":
    st.subheader("🔍 Search Faculty")

    name=st.text_input("Faculty Name")
    fid=st.text_input("Faculty ID")

    if st.button("Search"):
        df=data
        if name:
            df=df[df["faculty"].str.contains(name,case=False)]
        if fid:
            df=df[df["faculty_id"].str.contains(fid,case=False)]
        st.dataframe(df)

    st.subheader("🔎 Similar Papers")
    q=st.text_input("Search Title")

    if st.button("Find"):
        if not data.empty:
            st.dataframe(find_similar(q))

# ---------------- ANALYTICS ----------------
elif menu=="Analytics":
    st.subheader("📊 Dashboard")

    if not data.empty:
        c1,c2=st.columns(2)
        c1.metric("Total Records",len(data))
        c2.metric("Unique Faculty",data["faculty"].nunique())

        st.bar_chart(data["status"].value_counts())
        st.line_chart(data["status"].value_counts())

        st.dataframe(data)

# ---------------- DATABASE ----------------
elif menu=="Database":
    st.subheader("🗄️ Manage Data")

    n=st.text_input("Name")
    t=st.text_input("Title")
    j=st.text_input("Journal")
    s=st.selectbox("Status",["Published","Accepted","Rejected","Under Review"])

    if st.button("➕ Add Record"):
        if n and t and j:
            insert_data(n,t,j,s)
            st.rerun()

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df)

        if st.button("📥 Insert CSV"):
            insert_csv(df)
            st.rerun()

    fid=st.text_input("Enter Faculty ID")

    if st.button("🗑️ Delete"):
        delete_data(fid)
        st.rerun()

    with st.expander("View Data"):
        st.dataframe(data)

# ---------------- DOWNLOAD ----------------
elif menu=="Download":
    st.subheader("📥 Download Reports")

    st.download_button(
        "📊 Download Excel",
        data.to_csv(index=False),
        file_name="faculty_data.csv"
    )

    if PDF_AVAILABLE:
        if st.button("📄 Generate PDF"):
            path=generate_pdf()
            with open(path,"rb") as f:
                st.download_button("Download PDF", f, "report.pdf")

dintilo mootham aa otp antha paina changes antha chesi full code ivu
