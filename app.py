import streamlit as st
import pickle
import numpy as np
from nltk.corpus import stopwords

# ==========================================
# 1. LOAD MODEL AND VECTORIZER
# ==========================================
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('cv.pkl', 'rb') as f:
            cv = pickle.load(f)
        return model, cv
    except FileNotFoundError:
        st.error("Error: 'model.pkl' or 'cv.pkl' not found. Ensure they are in the same folder as app.py.")
        st.stop()

rf, cv = load_models()

# ==========================================
# 2. DEFINE FEATURE FUNCTIONS (EXACT LOGIC)
# ==========================================

def preprocess(text):
    """
    Paste your EXACT preprocessing logic from Kaggle here.
    Example below (replace with your actual logic if different):
    """
    # Example: Lowercase and strip
    return text.lower().strip()

def test_common_words(q1, q2):
    s1 = set(q1.split())
    s2 = set(q2.split())
    return len(s1.intersection(s2))

def test_total_words(q1, q2):
    return len(q1.split()) + len(q2.split())

def test_fetch_token_features(q1, q2):
    """
    Returns exactly 8 token-based features.
    """
    SAFE_DIV = 0.0001
    try:
        STOP_WORDS = stopwords.words("english")
    except LookupError:
        import nltk
        nltk.download('stopwords')
        STOP_WORDS = stopwords.words("english")

    token_features = [0.0] * 8
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])
    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    # 1. cwc_min
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    # 2. cwc_max
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    # 3. csc_min
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    # 4. csc_max
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    # 5. ctc_min
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    # 6. ctc_max
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    # 7. Last word equal
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    # 8. First word equal
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features

def query_point_creator(q1, q2):
    input_query = []
    
    # Preprocess
    q1_proc = preprocess(q1)
    q2_proc = preprocess(q2)
    
    # --- Basic Features (7 items) ---
    input_query.append(len(q1_proc))
    input_query.append(len(q2_proc))
    input_query.append(len(q1_proc.split(" ")))
    input_query.append(len(q2_proc.split(" ")))
    
    common = test_common_words(q1_proc, q2_proc)
    total = test_total_words(q1_proc, q2_proc)
    
    input_query.append(common)
    input_query.append(total)
    input_query.append(round(common/total, 2) if total > 0 else 0)
    
    # --- Token Features (8 items) ---
    token_feats = test_fetch_token_features(q1_proc, q2_proc)
    input_query.extend(token_feats)
    
    # Total Base Features should be 15 (7 + 8)
    base = np.array(input_query).reshape(1, -1)
    
    # --- BOW Features (3000 + 3000 items) ---
    q1_bow = cv.transform([q1_proc]).toarray()
    q2_bow = cv.transform([q2_proc]).toarray()
    
    # Concatenate: 15 + 3000 + 3000 = 6015
    final_vector = np.hstack((base, q1_bow, q2_bow))
    
    return final_vector

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Question Duplicate Detector", page_icon="🔍")
st.title("🔍 Question Duplicate Detector")
st.write("Enter two questions to check if they are duplicates.")

q1 = st.text_area("Question 1", "What is your first name?")
q2 = st.text_area("Question 2", "What is your last name?")

if st.button("Check Duplicate"):
    if q1 and q2:
        try:
            vector = query_point_creator(q1, q2)
            
            # Optional: Debug print to terminal
            # print(f"Vector shape: {vector.shape}, Expected: {rf.n_features_in_}")
            
            prediction = rf.predict(vector)[0]
            proba = rf.predict_proba(vector)[0]
            
            if prediction == 1:
                st.success(f"**✅ Duplicate** (Confidence: {proba[1]:.2%})")
                st.write("The questions likely have the same meaning.")
            else:
                st.info(f"**❌ Not Duplicate** (Confidence: {proba[0]:.2%})")
                st.write("The questions have different meanings.")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Ensure `model.pkl`, `cv.pkl`, and `app.py` are consistent.")
    else:
        st.warning("Please enter both questions.")   