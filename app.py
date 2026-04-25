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

# ---------------- LOGIN UI ----------------
if not st.session_state.logged_in:
    st.title("🔐 Login / Signup")

    choice = st.radio("Choose Option", ["Login", "Signup"], key="auth_choice")

    user = st.text_input("Username", key="login_user")
    pwd = st.text_input("Password", type="password", key="login_pass")

    if choice == "Login":
        if st.button("Login", key="login_btn"):
            if login(user, pwd):
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        if st.button("Signup", key="signup_btn"):
            if user and pwd:
                signup(user, pwd)
                st.success("Account created! Please login.")
            else:
                st.warning("Enter details")

    st.stop()

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_sql("SELECT * FROM faculty_data", conn)
    if not df.empty:
        df.columns = df.columns.str.lower()
        df['faculty'] = df['faculty'].astype(str).str.strip()
        df['text'] = df['title'] + " " + df['journal']
    return df

data = load_data()

# ---------------- ML ----------------
if not data.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['status']

    model = LogisticRegression()
    model.fit(X, y)

# ---------------- FUNCTIONS ----------------
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

def insert_csv(df):
    for _, row in df.iterrows():
        insert_data(row['faculty'], row['title'], row['journal'], row['status'])

def predict_status(title, journal):
    vec = vectorizer.transform([title + " " + journal])
    return model.predict(vec)[0]

def find_similar(query):
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, X)[0]
    return data[sim > 0.5]

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

    t = st.text_input("Title", key="pred_title")
    j = st.text_input("Journal", key="pred_journal")

    if st.button("Predict", key="pred_btn"):
        if not data.empty and t and j:
            st.success(predict_status(t, j))
        else:
            st.warning("Need data")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("🔍 Search")

    name = st.text_input("Faculty Name", key="search_name")

    if st.button("Search", key="search_btn"):
        res = data[data['faculty'].str.contains(name, case=False, na=False)]
        st.dataframe(res if not res.empty else pd.DataFrame())

    st.subheader("🔎 Similar Papers")

    q = st.text_input("Keyword", key="search_keyword")

    if st.button("Find", key="find_btn"):
        if not data.empty:
            res = find_similar(q)
            st.dataframe(res)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("📊 Analytics")

    if not data.empty:
        st.bar_chart(data['status'].value_counts())
        st.line_chart(data['status'].value_counts())
        st.write("Pie (counts):")
        st.write(data['status'].value_counts())
    else:
        st.info("No data")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("🗄️ Database Management")

    # ADD
    n = st.text_input("Name", key="db_name")
    t = st.text_input("Title", key="db_title")
    j = st.text_input("Journal", key="db_journal")
    s = st.selectbox("Status", ["published","accepted","rejected"], key="db_status")

    if st.button("Add", key="add_btn"):
        if n and t and j:
            insert_data(n, t, j, s)
            st.success("Added")
            st.rerun()

    # CSV UPLOAD
    st.subheader("📁 Upload CSV")
    file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df)
        if st.button("Insert CSV", key="csv_btn"):
            insert_csv(df)
            st.success("Inserted")
            st.rerun()

    # VIEW
    st.subheader("📋 Data")
    st.dataframe(data)

    # DELETE
    did = st.text_input("Enter ID", key="delete_id")

    if st.button("Delete", key="delete_btn"):
        if did:
            delete_data(did)
            st.success("Deleted")
            st.rerun()

# ---------------- LOGOUT ----------------
if st.button("Logout", key="logout_btn"):
    st.session_state.logged_in = False
    st.rerun()