import pandas as pd
import streamlit as st
import sqlite3
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Faculty Research System", layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("faculty.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faculty_data (
faculty_id TEXT, faculty TEXT, title TEXT, journal TEXT, status TEXT, owner TEXT)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
username TEXT, password TEXT)
""")
conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

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
                st.session_state.username=u
                st.rerun()
    else:
        if st.button("Signup"):
            if u and p:
                signup(u,p)
                st.session_state.logged_in=True
                st.session_state.username=u
                st.rerun()

    st.stop()

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_sql("SELECT * FROM faculty_data", conn)
    if not df.empty:
        df.columns = df.columns.str.lower()

        # 🔥 FIX: normalize status
        df["status"] = df["status"].astype(str).str.strip().str.title()

        df["text"] = df["title"].astype(str) + " " + df["journal"].astype(str)
    return df

data = load_data()

# ---------------- MODEL ----------------
model=None
vectorizer=None

if not data.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["text"])
    model = LogisticRegression(max_iter=200)
    model.fit(X, data["status"])

# ---------------- FUNCTIONS ----------------
def insert_data(n,t,j,s):
    # also normalize while inserting
    s = s.strip().title()

    cursor.execute("INSERT INTO faculty_data VALUES (?,?,?,?,?,?)",
                   ("FID-"+str(uuid.uuid4())[:8], n,t,j,s, st.session_state.username))
    conn.commit()

def delete_data(fid):
    cursor.execute("DELETE FROM faculty_data WHERE faculty_id=? AND owner=?",
                   (fid, st.session_state.username))
    conn.commit()
    return cursor.rowcount > 0

def predict_status_with_confidence(t,j):
    text = t + " " + j
    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    return prediction, round(max(prob)*100,2)

# ---------------- SIDEBAR ----------------
menu=st.sidebar.radio("📌 Menu",[
    "Prediction","Analytics","Database"
])

st.title("📚 Faculty Research System")

# ---------------- PREDICTION ----------------
if menu=="Prediction":
    t=st.text_input("Title")
    j=st.text_input("Journal")

    if st.button("Predict"):
        if model and t and j:
            pred,conf = predict_status_with_confidence(t,j)
            st.success(pred)
            st.info(f"Confidence: {conf}%")

# ---------------- ANALYTICS ----------------
elif menu=="Analytics":
    st.subheader("📊 Dashboard")

    if not data.empty:
        c1,c2=st.columns(2)
        c1.metric("Total Records",len(data))
        c2.metric("Unique Faculty",data["faculty"].nunique())

        # ✅ CLEAN GRAPH (NO DUPLICATES)
        status_counts = data["status"].value_counts()
        st.bar_chart(status_counts)

        st.dataframe(data)

# ---------------- DATABASE ----------------
elif menu=="Database":
    n=st.text_input("Name")
    t=st.text_input("Title")
    j=st.text_input("Journal")
    s=st.selectbox("Status",["Published","Accepted","Rejected","Under Review"])

    if st.button("Add"):
        insert_data(n,t,j,s)
        st.rerun()

    fid=st.text_input("Enter Faculty ID")

    if st.button("Delete"):
        if delete_data(fid):
            st.success("Deleted")
        else:
            st.error("Not allowed")

    st.dataframe(data)
