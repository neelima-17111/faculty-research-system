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

# ---------------- DATABASE ----------------
conn = sqlite3.connect("faculty.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faculty_data (
faculty_id TEXT, faculty TEXT, title TEXT, journal TEXT, status TEXT)
""")
conn.commit()

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_sql("SELECT * FROM faculty_data", conn)
    if not df.empty:
        df.columns = df.columns.str.lower()

        # ✅ FIX: normalize status
        df["status"] = df["status"].astype(str).str.strip().str.title()

        df["text"] = df["title"].astype(str) + " " + df["journal"].astype(str)
    return df

data = load_data()

# ---------------- MODEL ----------------
model = None
vectorizer = None

if not data.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["text"])
    model = LogisticRegression(max_iter=200)
    model.fit(X, data["status"])

# ---------------- FUNCTIONS ----------------
def insert_data(n,t,j,s):
    s = s.strip().title()
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
        insert_data(
            str(row[faculty_col]),
            str(row[title_col]),
            str(row[journal_col]),
            str(row[status_col])
        )

def delete_data(fid):
    cursor.execute("DELETE FROM faculty_data WHERE faculty_id=?", (fid,))
    conn.commit()

# ---------------- SIDEBAR ----------------
menu=st.sidebar.radio("📌 Menu",[
    "Prediction","Search","Analytics","Database","Download"
])

st.title("📚 Faculty Research System")

# ---------------- ANALYTICS ----------------
if menu=="Analytics":
    st.subheader("📊 Dashboard")

    if not data.empty:
        c1,c2=st.columns(2)
        c1.metric("Total Records",len(data))
        c2.metric("Unique Faculty",data["faculty"].nunique())

        # ✅ ONLY ONE CLEAN GRAPH
        status_counts = data["status"].value_counts()

        st.bar_chart(status_counts)

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

    st.dataframe(data)

# ---------------- DOWNLOAD ----------------
elif menu=="Download":
    st.subheader("📥 Download Reports")

    st.download_button(
        "📊 Download CSV",
        data.to_csv(index=False),
        file_name="faculty_data.csv"
    )

    if PDF_AVAILABLE:
        if st.button("📄 Generate PDF"):
            path="/mnt/data/report.pdf"
            doc=SimpleDocTemplate(path)
            styles=getSampleStyleSheet()
            elements=[]

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

            with open(path,"rb") as f:
                st.download_button("Download PDF", f, "report.pdf")
