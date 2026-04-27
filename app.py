import pandas as pd
import streamlit as st
import sqlite3
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Faculty Research System", layout="centered")

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
    st.subheader("🔐 Login / Signup")

    choice = st.radio("Choose Option", ["Login", "Signup"])

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if choice == "Login":
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

    st.stop()

# ---------------- HELPERS ----------------
def generate_id():
    return "FID-" + str(uuid.uuid4())[:8]

def insert_data(name, title, journal, status):
    fid = generate_id()
    cursor.execute("INSERT INTO faculty_data VALUES (?, ?, ?, ?, ?)",
                   (fid, name, title, journal, status))
    conn.commit()

def delete_data(fid):
    cursor.execute("DELETE FROM faculty_data WHERE faculty_id=?", (fid,))
    conn.commit()

# ---------------- LIMITED DATA LOAD ----------------
@st.cache_data
def load_sample():
    df = pd.read_sql("SELECT * FROM faculty_data LIMIT 2000", conn)
    if not df.empty:
        df['text'] = df['title'] + " " + df['journal']
    return df

data = load_sample()

# ---------------- ML MODEL (LIMITED DATA ONLY) ----------------
if not data.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])

    model = LogisticRegression(max_iter=200)
    model.fit(X, data['status'])

# ---------------- FUNCTIONS ----------------
def predict_status(title, journal):
    vec = vectorizer.transform([title + " " + journal])
    probs = model.predict_proba(vec)[0]
    result = model.classes_[probs.argmax()]
    confidence = round(max(probs) * 100, 2)
    return result, confidence

def find_similar(query):
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, X)[0]
    return data[sim > 0.3]

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "🔍 Search",
    "📊 Analytics",
    "🗄️ Database"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("🔮 Predict Status")

    pt = st.text_input("Title")
    pj = st.text_input("Journal")

    if st.button("Predict"):
        if not data.empty and pt and pj:
            result, conf = predict_status(pt, pj)
            st.success(result)
            st.info(f"Confidence: {conf}%")
        else:
            st.warning("Need data")

# ---------------- TAB 2 (DB SEARCH OPTIMIZED) ----------------
with tab2:
    st.subheader("🔍 Search Faculty")

    name = st.text_input("Faculty Name")
    fid_search = st.text_input("Faculty ID")

    if st.button("Search"):
        query = "SELECT * FROM faculty_data WHERE 1=1"
        params = []

        if name:
            query += " AND faculty LIKE ?"
            params.append(f"%{name}%")

        if fid_search:
            query += " AND faculty_id LIKE ?"
            params.append(f"%{fid_search}%")

        result_df = pd.read_sql(query, conn, params=params)
        st.dataframe(result_df.head(100))  # LIMIT UI

    st.subheader("🔎 Similar Papers")

    q = st.text_input("Keyword")

    if st.button("Find"):
        if not data.empty:
            st.dataframe(find_similar(q))

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("📊 Analytics")

    df = pd.read_sql("SELECT status, COUNT(*) as count FROM faculty_data GROUP BY status", conn)
    st.bar_chart(df.set_index("status"))

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("🗄️ Database")

    dn = st.text_input("Name")
    dt = st.text_input("Title")
    dj = st.text_input("Journal")
    ds = st.selectbox("Status", ["Published","Accepted","Rejected","Under Review"])

    if st.button("Add"):
        if dn and dt and dj:
            insert_data(dn, dt, dj, ds)
            st.success("Added")
            st.rerun()

    st.subheader("📁 Upload CSV (ONE TIME ONLY)")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file, chunksize=5000)
        for chunk in df:
            chunk.to_sql("faculty_data", conn, if_exists="append", index=False)
        st.success("Inserted Large CSV Safely")

    st.subheader("📋 Preview Data")

    preview = pd.read_sql("SELECT * FROM faculty_data LIMIT 100", conn)
    st.dataframe(preview)

    did = st.text_input("Enter ID")

    if st.button("Delete"):
        delete_data(did)
        st.success("Deleted")
        st.rerun()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
