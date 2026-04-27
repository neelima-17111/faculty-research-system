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
            else:
                st.warning("Enter details")

    st.stop()

# ---------------- MAIN TITLE ----------------
st.title("📚 Faculty Research System")

# ---------------- LOAD DATA (CACHED) ----------------
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT * FROM faculty_data", conn)
    if not df.empty:
        df.columns = df.columns.str.lower()
        df['faculty'] = df['faculty'].astype(str).str.strip()
        df['status'] = df['status'].str.capitalize()
        df['text'] = df['title'].astype(str) + " " + df['journal'].astype(str)
    return df

data = load_data()

# ---------------- ML MODEL ----------------
if not data.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])

    model = LogisticRegression(class_weight='balanced', max_iter=200)
    model.fit(X, data['status'])

# ---------------- FUNCTIONS ----------------
def generate_id():
    return "FID-" + str(uuid.uuid4())[:8]

def insert_data(name, title, journal, status):
    fid = generate_id()
    cursor.execute("INSERT INTO faculty_data VALUES (?, ?, ?, ?, ?)",
                   (fid, name, title, journal, status.capitalize()))
    conn.commit()

def delete_data(fid):
    cursor.execute("DELETE FROM faculty_data WHERE faculty_id=?", (fid,))
    conn.commit()

# 🔥 CHUNK + BULK INSERT (MAIN FIX)
def insert_csv(file):
    chunksize = 5000
    progress = st.progress(0)
    total = 0

    for chunk in pd.read_csv(file, chunksize=chunksize):
        chunk.columns = chunk.columns.str.lower()

        chunk['faculty_id'] = ["FID-" + str(uuid.uuid4())[:8] for _ in range(len(chunk))]
        chunk = chunk[['faculty_id', 'faculty', 'title', 'journal', 'status']]

        chunk.to_sql('faculty_data', conn, if_exists='append', index=False)

        total += len(chunk)
        progress.progress(min(total / 100000, 1.0))  # approximate progress

    progress.empty()
    st.success("CSV Inserted Successfully 🚀")

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

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("🔍 Search Faculty")

    name = st.text_input("Search by Faculty Name")
    fid_search = st.text_input("Search by Faculty ID")

    if st.button("Search"):
        result_df = data

        if name:
            result_df = result_df[result_df['faculty'].str.contains(name, case=False)]

        if fid_search:
            result_df = result_df[result_df['faculty_id'].str.contains(fid_search, case=False)]

        st.dataframe(result_df)

    st.subheader("🔎 Similar Papers")

    q = st.text_input("Keyword")

    if st.button("Find"):
        if not data.empty:
            st.dataframe(find_similar(q))

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("📊 Analytics")

    if not data.empty:
        st.bar_chart(data['status'].value_counts())
        st.line_chart(data['status'].value_counts())
    else:
        st.info("No data")

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

    st.subheader("📁 Upload CSV")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        if st.button("Insert CSV"):
            insert_csv(file)
            st.rerun()

    st.subheader("📋 Data")
    st.dataframe(data)

    did = st.text_input("Enter ID")

    if st.button("Delete"):
        if did:
            delete_data(did)
            st.success("Deleted")
            st.rerun()

    # LOGOUT
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
