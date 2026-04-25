import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Faculty Research System", layout="centered")

st.title("📚 Faculty Research & Publication Management System")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("data.csv")

# Fill missing values
data['title'] = data['title'].fillna("")
data['journal'] = data['journal'].fillna("")
data['status'] = data['status'].fillna("Unknown")

# Combine text
data['text'] = data['title'] + " " + data['journal']

# ---------------- ML MODEL ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])

# IMPORTANT FIX: balanced model
model = LogisticRegression(class_weight='balanced', max_iter=200)
model.fit(X, data['status'])

# ---------------- FUNCTIONS ----------------

def predict_status(title, journal):
    text = [title + " " + journal]
    vec = vectorizer.transform(text)
    
    # probability check
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    
    # get best class
    result = classes[probs.argmax()]
    confidence = round(max(probs) * 100, 2)
    
    return result, confidence


def find_similar_papers(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X)[0]

    results = []
    for i, score in enumerate(similarity):
        if score > 0.3:   # reduced threshold
            results.append({
                "Faculty ID": data.iloc[i]['faculty_id'],
                "Faculty Name": data.iloc[i]['faculty'],
                "Paper Title": data.iloc[i]['title'],
                "Similarity": round(score, 2)
            })

    return results


def check_duplicate(title):
    return data[data['title'].str.lower() == title.lower()]

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🔍 Search", "📊 Analytics"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Predict Publication Status")

    faculty_id = st.text_input("Faculty ID", key="pred_id")
    title = st.text_input("Paper Title", key="pred_title")
    journal = st.text_input("Journal Name", key="pred_journal")

    if st.button("Predict", key="btn_pred"):
        if faculty_id and title and journal:

            dup = check_duplicate(title)
            if not dup.empty:
                st.warning("⚠️ Duplicate paper found")
                st.dataframe(dup[['faculty_id','faculty','title']])

            result, confidence = predict_status(title, journal)

            st.success(f"Prediction: {result}")
            st.info(f"Confidence: {confidence}%")

        else:
            st.warning("Enter all fields")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Search Faculty Papers")

    search_id = st.text_input("Enter Faculty ID", key="search_id")

    if st.button("Search", key="btn_search"):
        if search_id:
            results = data[data['faculty_id'].astype(str) == search_id]

            if not results.empty:
                st.dataframe(results[['faculty','title','journal','status']])
            else:
                st.warning("No records found")

    st.subheader("Find Similar Papers")

    query = st.text_input("Enter keyword", key="sim_query")

    if st.button("Find Similar", key="btn_sim"):
        if query:
            res = find_similar_papers(query)
            if res:
                st.dataframe(pd.DataFrame(res))
            else:
                st.info("No similar papers found")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Analytics")

    if st.checkbox("Show Status Distribution", key="chk_graph"):
        st.bar_chart(data['status'].value_counts())
