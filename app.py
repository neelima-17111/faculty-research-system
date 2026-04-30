import pandas as pd
import streamlit as st
import sqlite3
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Faculty Research System", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0b1f3a;
}
h1, h2, h3 {
    color: #0b3d91;
}
.stButton>button {
    background-color: #0b3d91;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #1456c3;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("faculty.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faculty_data (
    faculty_id TEXT,
    faculty TEXT,
    title TEXT,
    journal TEXT,
    status TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT,
    password TEXT
)
""")
conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- AUTH ----------------
def login(u, p):
    return cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (u, p)
    ).fetchone()

def signup(u, p):
    cursor.execute("INSERT INTO users VALUES (?,?)", (u, p))
    conn.commit()

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:
    st.title("📚 Faculty Research System")

    mode = st.radio("Choose Option", ["Login", "Signup"])
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if mode == "Login":
        if st.button("Login"):
            if login(user, pwd):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        if st.button("Signup"):
            if user and pwd:
                signup(user, pwd)
                st.success("Account created")
            else:
                st.warning("Enter details")
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
def generate_id():
    return "FID-" + str(uuid.uuid4())[:8]

def insert_data(name, title, journal, status):
    cursor.execute(
        "INSERT INTO faculty_data VALUES (?, ?, ?, ?, ?)",
        (generate_id(), name, title, journal, status)
    )
    conn.commit()

def delete_data(fid):
    cursor.execute("DELETE FROM faculty_data WHERE faculty_id=?", (fid,))
    conn.commit()
    return cursor.rowcount > 0

def predict_status(title, journal):
    vec = vectorizer.transform([title + " " + journal])
    pred = model.predict(vec)[0]
    return pred

def find_similar(query):
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, X)[0]
    return data[sim > 0.3]

# ---------------- PDF GENERATION ----------------
def generate_pdf():
    file_path = "/mnt/data/report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Faculty Research Report", styles["Title"]))
    content.append(Spacer(1, 12))

    for _, row in data.iterrows():
        text = f"{row['faculty']} - {row['title']} ({row['status']})"
        content.append(Paragraph(text, styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)
    return file_path

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", [
    "Prediction",
    "Search",
    "Analytics",
    "Database",
    "Download Report"
])

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>📚 Faculty Research System</h1>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if menu == "Prediction":
    st.subheader("🔮 Predict Status")

    title = st.text_input("Title")
    journal = st.text_input("Journal")

    if st.button("🔮 Predict"):
        if not data.empty:
            result = predict_status(title, journal)
            st.success(result)
        else:
            st.warning("No data")

# ---------------- SEARCH ----------------
elif menu == "Search":
    st.subheader("🔍 Search")

    name = st.text_input("Faculty Name")
    fid = st.text_input("Faculty ID")

    if st.button("Search"):
        df = data
        if name:
            df = df[df["faculty"].str.contains(name, case=False)]
        if fid:
            df = df[df["faculty_id"].str.contains(fid, case=False)]
        st.dataframe(df)

    st.subheader("🔎 Similar Papers")
    q = st.text_input("Search Title")

    if st.button("Find"):
        st.dataframe(find_similar(q))

# ---------------- ANALYTICS ----------------
elif menu == "Analytics":
    st.subheader("📊 Dashboard")

    if not data.empty:
        col1, col2 = st.columns(2)
        col1.metric("Total Records", len(data))
        col2.metric("Unique Faculty", data["faculty"].nunique())

        status_counts = data["status"].value_counts()
        st.bar_chart(status_counts)
        st.line_chart(status_counts)

        st.dataframe(data)
    else:
        st.info("No data")

# ---------------- DATABASE ----------------
elif menu == "Database":
    st.subheader("🗄️ Manage Data")

    name = st.text_input("Name")
    title = st.text_input("Title")
    journal = st.text_input("Journal")
    status = st.selectbox("Status", ["Published", "Accepted", "Rejected", "Under Review"])

    if st.button("➕ Add"):
        insert_data(name, title, journal, status)
        st.success("Added")
        st.rerun()

    st.subheader("Delete Record")
    fid = st.text_input("Enter ID")

    if st.button("🗑️ Delete"):
        if delete_data(fid):
            st.success("Deleted")
        else:
            st.error("Not found")
        st.rerun()

# ---------------- PDF DOWNLOAD ----------------
elif menu == "Download Report":
    st.subheader("📄 Generate Report")

    if st.button("Generate PDF"):
        path = generate_pdf()
        with open(path, "rb") as f:
            st.download_button("Download PDF", f, file_name="report.pdf")
