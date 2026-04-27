import pandas as pd
import streamlit as st
import sqlite3
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
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

# ---------------- MAIN ----------------
st.title("📚 Faculty Research System")

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_sql("SELECT * FROM faculty_data", conn)
    if not df.empty:
        df.columns = df.columns.str.lower()
        df["faculty"] = df["faculty"].astype(str).str.strip()
        df["status"] = df["status"].str.capitalize()
        df["text"] = df["title"].astype(str) + " " + df["journal"].astype(str)
    return df

data = load_data()

# ---------------- ML MODEL ----------------
if not data.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["text"])

    model = LogisticRegression(class_weight="balanced", max_iter=200)
    model.fit(X, data["status"])

# ---------------- FUNCTIONS ----------------
def generate_id():
    return "FID-" + str(uuid.uuid4())[:8]

def insert_data(name, title, journal, status):
    fid = generate_id()
    cursor.execute(
        "INSERT INTO faculty_data VALUES (?, ?, ?, ?, ?)",
        (fid, name, title, journal, status)
    )
    conn.commit()

def delete_data(fid):
    fid = fid.strip().upper()

    cursor.execute(
        "DELETE FROM faculty_data WHERE UPPER(faculty_id)=?",
        (fid,)
    )
    conn.commit()
    return cursor.rowcount > 0

def insert_csv(df):
    for _, row in df.iterrows():
        insert_data(row["faculty"], row["title"], row["journal"], row["status"])

def predict_status(title, journal):
    vec = vectorizer.transform([title + " " + journal])
    probs = model.predict_proba(vec)[0]
    result = model.classes_[probs.argmax()]
    confidence = round(max(probs) * 100, 2)
    return result, confidence

def find_similar(query):
    if data.empty:
        return pd.DataFrame()
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

    title = st.text_input("Title")
    journal = st.text_input("Journal")

    if st.button("Predict"):
        if not data.empty and title and journal:
            result, conf = predict_status(title, journal)
            st.success(result)
            st.info(f"Confidence: {conf}%")
        else:
            st.warning("Not enough data")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("🔍 Search Faculty")

    name = st.text_input("Faculty Name")
    fid = st.text_input("Faculty ID")

    if st.button("Search"):
        df = data

        if name:
            df = df[df["faculty"].str.contains(name, case=False, na=False)]

        if fid:
            df = df[df["faculty_id"].str.contains(fid, case=False, na=False)]

        st.dataframe(df)

    st.subheader("🔎 Similar Papers")

    q = st.text_input("Search Title")

    if st.button("Find"):
        st.dataframe(find_similar(q))

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("📊 Analytics Dashboard")

    if not data.empty:

        # Count status
        status_counts = data["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        # Bar chart
        st.bar_chart(status_counts.set_index("Status"))

        # Line chart (FIXED)
        st.line_chart(status_counts.set_index("Status"))

        # Extra info
        st.write("Status Distribution Table")
        st.dataframe(status_counts)

    else:
        st.info("No data available")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("🗄️ Database")

    name = st.text_input("Name", key="name")
    title = st.text_input("Title", key="title")
    journal = st.text_input("Journal", key="journal")
    status = st.selectbox("Status", ["Published", "Accepted", "Rejected", "Under Review"])

    if st.button("Add Record"):
        if name and title and journal:
            insert_data(name, title, journal, status)
            st.success("Record Added")
            st.rerun()

    st.subheader("📁 Upload CSV")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df)

        if st.button("Insert CSV"):
            insert_csv(df)
            st.success("Uploaded")
            st.rerun()

    st.markdown(f"📊 Total Records: {len(data)}")

    with st.expander("View Data"):
        st.dataframe(data)

    # ---------------- DELETE ----------------
    st.subheader("🗑️ Delete Record")

    delete_id = st.text_input("Enter Faculty ID")

    if st.button("Delete"):
        if delete_id:
            if delete_data(delete_id):
                st.success("Deleted Successfully")
            else:
                st.error("ID not found")
            st.rerun()

    # ---------------- LOGOUT ----------------
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
