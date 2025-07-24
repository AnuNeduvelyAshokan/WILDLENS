import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- API Endpoints ---
MAMMAL_API_URL = "http://127.0.0.1:8000/predict_mammal/"
INSECT_API_URL = "http://127.0.0.1:8000/predict_insect/"
T5_API_URL = "http://127.0.0.1:8000/t5_ask/"
ARTICLE_API_URL = "http://127.0.0.1:8000/generate_article/"
SIGNUP_URL = "http://127.0.0.1:8000/signup/"
LOGIN_URL = "http://127.0.0.1:8000/login/"
TOTAL_USERS_URL = "http://127.0.0.1:8000/admin/total_users/"
TOTAL_PREDICTIONS_URL = "http://127.0.0.1:8000/admin/total_predictions/"
ALL_PREDICTIONS_URL = "http://127.0.0.1:8000/admin/all_predictions/"
DELETE_HISTORY_URL = "http://127.0.0.1:8000/admin/delete_user_history/"
USER_HISTORY_URL = "http://127.0.0.1:8000/user/history/"

# --- Session State Init ---
for k, v in {
    "user_id": None, "is_admin": False, "username": "", "prediction_counts": {},
    "prediction_history": [], "chat_history": [],
    "t5_question": "", "t5_species": "", "article_text": "", "article_species": "",
    "article_map_regions": [], "article_map_countries": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Map Helper Data ---
import re
import pycountry
import pandas as pd
import plotly.express as px

# ----------- CONTINENTS (with ISO3 codes!) -----------
CONTINENTS = {
    "Africa": ["DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD", "COM", "COG", "COD", "CIV",
               "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR",
               "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MYT", "MAR", "MOZ", "NAM", "NER", "NGA", "REU", "RWA",
               "SHN", "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "TUN", "UGA", "ESH",
               "ZMB", "ZWE"],
    "Asia": ["AFG", "ARM", "AZE", "BHR", "BGD", "BTN", "BRN", "KHM", "CHN", "CXR", "CCK", "GEO", "HKG", "IND",
             "IDN", "IRN", "IRQ", "ISR", "JPN", "JOR", "KAZ", "PRK", "KOR", "KWT", "KGZ", "LAO", "LBN", "MAC",
             "MYS", "MDV", "MNG", "MMR", "NPL", "OMN", "PAK", "PSE", "PHL", "QAT", "SAU", "SGP", "LKA", "SYR",
             "TWN", "TJK", "THA", "TLS", "TUR", "TKM", "ARE", "UZB", "VNM", "YEM"],
    "Europe": ["ALB", "AND", "AUT", "BLR", "BEL", "BIH", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FRO", "FIN",
               "FRA", "DEU", "GIB", "GRC", "GGY", "HUN", "ISL", "IRL", "IMN", "ITA", "JEY", "LVA", "LIE", "LTU",
               "LUX", "MLT", "MDA", "MCO", "MNE", "NLD", "MKD", "NOR", "POL", "PRT", "ROU", "RUS", "SMR", "SRB",
               "SVK", "SVN", "ESP", "SWE", "CHE", "UKR", "GBR", "VAT"],
    "North America": ["AIA", "ATG", "ABW", "BHS", "BRB", "BLZ", "BMU", "BES", "VGB", "CAN", "CYM", "CRI", "CUB",
                      "CUW", "DMA", "DOM", "SLV", "GRL", "GRD", "GLP", "GTM", "HTI", "HND", "JAM", "MTQ", "MEX",
                      "MSR", "NIC", "PAN", "PRI", "BLM", "KNA", "LCA", "MAF", "SPM", "VCT", "SXM", "TTO", "TCA",
                      "USA", "VIR"],
    "South America": ["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "FLK", "GUF", "GUY", "PRY", "PER", "SUR", "URY", "VEN"],
    "Australia": ["AUS", "FJI", "GUM", "KIR", "MHL", "FSM", "NRU", "NZL", "PLW", "PNG", "SLB", "TKL", "TON", "TUV", "VUT", "WLF"],
    "Oceania": ["AUS", "FJI", "FSM", "GUM", "KIR", "MHL", "NRU", "NZL", "PLW", "PNG", "SLB", "TKL", "TON", "TUV", "VUT", "WLF"],
    "Antarctica": ["ATA"]
}
SUBCONTINENT = ["IND", "PAK", "BGD", "NPL", "LKA", "BTN", "MDV"]

# ----------- COUNTRY NAME TO ISO3 CONVERSION -----------
def country_name_to_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except Exception:
        return None

# ----------- REGION AND COUNTRY EXTRACTION -----------
def extract_regions_and_countries(article):
    found_continents, found_countries = set(), set()
    text = article.lower()
    # Find continents in article
    for continent in CONTINENTS:
        if continent.lower() in text:
            found_continents.add(continent)
    # Subcontinent logic
    if "subcontinent" in text:
        found_countries.update(SUBCONTINENT)
    # Find country names in article and add ISO3 code
    for country in pycountry.countries:
        if re.search(rf"\b{country.name.lower()}\b", text):
            found_countries.add(country.alpha_3)
    return list(found_continents), list(found_countries)

# ----------- PLOT MAP -----------
def plot_map(continents, countries):
    highlight_codes = []
    colors = []
    if continents:
        for cont in continents:
            highlight_codes += CONTINENTS[cont]
            colors += ['red'] * len(CONTINENTS[cont])
    if countries:
        highlight_codes += countries
        colors += ['blue'] * len(countries)
    if not highlight_codes:
        return None
    df = pd.DataFrame({'iso_alpha': highlight_codes, 'Color': colors})
    fig = px.choropleth(
        df,
        locations='iso_alpha',
        color='Color',
        color_discrete_map={'red': 'red', 'blue': 'blue'},
        projection="natural earth",
        title="Distribution Map",
    )
    fig.update_geos(showcoastlines=True, showcountries=True, showframe=False)
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    return fig

# ----------- USAGE EXAMPLE -----------
# article = "The tiger is found in Asia, India and China. It also occurs in the subcontinent."
# continents, countries = extract_regions_and_countries(article)
# fig = plot_map(continents, countries)
# if fig:
#     st.plotly_chart(fig)


def handle_t5_chat_dual():
    question = st.session_state.t5_question.strip()
    species = st.session_state.t5_species.strip()
    if question and species:
        try:
            response = requests.post(T5_API_URL, json={"question": question, "species_name": species})
            answer = response.json().get('answer', 'Sorry, no answer found.')
            st.session_state.chat_history.append({"sender": "user", "message": f"{question} ({species})"})
            st.session_state.chat_history.append({"sender": "bot", "message": answer})
        except Exception:
            st.error("❌ T5 backend not responding.")
        st.session_state.t5_question = ""
        st.session_state.t5_species = ""
    else:
        st.warning("Please fill both question and species name.")

def clear_chat():
    st.session_state.chat_history = []

# --- Sidebar Navigation ---
st.set_page_config(page_title="Animal Detection & Chatbot", layout="wide")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "📈 Analytics"])

# --- Home/Login Page ---
if page == "Home":
    st.title("🔑 Login / Signup")
    tab1, tab2 = st.tabs(["User Login", "Admin Login"])

    # --- User Login ---
    with tab1:
        username = st.text_input("Username", key="user_username")
        password = st.text_input("Password", type="password", key="user_password")
        if st.button("User Login"):
            r = requests.post(LOGIN_URL, data={"username": username, "password": password})
            if r.status_code == 200:
                st.session_state.user_id = r.json()["user_id"]
                st.session_state.username = username
                st.session_state.is_admin = False
                # Fetch user history from backend
                hist_resp = requests.get(USER_HISTORY_URL, params={"user_id": st.session_state.user_id})
                if hist_resp.status_code == 200:
                    st.session_state.prediction_history = hist_resp.json().get("history", [])
                st.success("User login successful!")
            else:
                st.error("Invalid credentials.")

        new_username = st.text_input("New Username", key="new_user_username")
        new_password = st.text_input("New Password", type="password", key="new_user_password")
        if st.button("Register User"):
            r = requests.post(SIGNUP_URL, data={"username": new_username, "password": new_password})
            st.success("Signup successful!" if r.status_code == 200 else "Signup failed.")

    # --- Admin Login ---
    with tab2:
        admin_username = st.text_input("Admin Username", key="admin_username")
        admin_password = st.text_input("Admin Password", type="password", key="admin_password")
        if st.button("Admin Login"):
            r = requests.post(LOGIN_URL, data={"username": admin_username.strip(), "password": admin_password.strip()})
            if r.status_code == 200 and admin_username == "admin":
                st.session_state.user_id = r.json()["user_id"]
                st.session_state.username = admin_username
                st.session_state.is_admin = True
                st.success("Admin login successful!")
            else:
                st.error("Admin login failed. Check credentials.")

# --- Dashboard (Detection, Article, Chatbot, History) ---
elif page == "Dashboard":
    if not st.session_state.user_id:
        st.warning("Please login first.")
        st.stop()
    is_admin = st.session_state.is_admin
    st.title("🐾 Animal Detection & Educational Chatbot")

    # --- Prediction & Article ---
    st.subheader("🦁 / 🐝 Image Detection")
    detection_type = st.radio("Choose Type", ["Mammal 🦁", "Insect 🐝"])
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Preview", use_container_width=True)
        if st.button("Detect"):
            api_url = MAMMAL_API_URL if detection_type == "Mammal 🦁" else INSECT_API_URL
            files = {"file": uploaded_file.getvalue()}
            data = {"user_id": str(st.session_state.user_id)}
            try:
                r = requests.post(api_url, files=files, data=data)
                res = r.json()
                species = res.get("predicted_class")
                confidence = res.get("confidence")
                if species:
                    st.success(f"✅ Prediction: {species} (Confidence: {confidence:.2f})")
                    # Save history
                    hist_entry = {"Class": species, "Confidence (%)": round(confidence * 100, 2)}
                    st.session_state.prediction_history.append(hist_entry)
                    st.session_state.prediction_counts[species] = st.session_state.prediction_counts.get(species, 0) + 1
                    # Generate Article
                    data_article = {"species_name": species}
                    article_response = requests.post(ARTICLE_API_URL, data=data_article)
                    full_article = article_response.json().get("article", "")
                    # --- Remove model prompt, add h3 title ---
                    import re
                    # ...in your detect/image code after getting full_article:
                    match = re.search(r'write a short educational article about\s*([^.:\n]+)', full_article, re.IGNORECASE)
                    if match:
                        species_name = match.group(1).strip().capitalize()
                        art_title = f"Educational Article about {species_name}"
                        # Remove ONLY the first line (title) from the article, keep the rest as body
                        art_body = re.sub(r'^.*?(\.|\:)\s*', '', full_article, count=1, flags=re.IGNORECASE | re.DOTALL).strip()
                    else:
                        art_title = "Educational Article"
                        art_body = full_article.strip()

                    # --- Display with styled black border ---
                    st.markdown(
                        f"""
                        <div style='border:2px solid #000; border-radius:12px; padding:16px; margin:10px 0; background:#111; color:#fff'>
                            <h3 style='margin-top:0;color:#fff'>{art_title}</h3>
                            <div style='font-size:1.1em; color:#eee;'>{art_body}</div>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    # --- Map ---
                    continents, countries = extract_regions_and_countries(full_article)
                    st.session_state.article_map_regions = continents
                    st.session_state.article_map_countries = countries
                    map_fig = plot_map(continents, countries)
                    if map_fig:
                        st.plotly_chart(map_fig, use_container_width=True)
                else:
                    st.error("Prediction failed.")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Chatbot QA --- 
    st.subheader("💬 Species QA Chatbot")
    st.markdown("#### Chat History")
    for chat in st.session_state.chat_history:
        if chat["sender"] == "user":
            st.markdown(
                f"<div style='text-align:right; margin-bottom:8px;'>"
                f"<span style='vertical-align:middle; font-size:1.5em;'>🧑‍💻</span>"
                f"<span style='display:inline-block; background:#2196F3; color:#fff; padding:10px 16px; border-radius:16px 0 16px 16px; max-width:70%; word-break:break-word;'>{chat['message']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left; margin-bottom:8px;'>"
                f"<span style='display:inline-block; background:#4CAF50; color:#fff; padding:10px 16px; border-radius:0 16px 16px 16px; max-width:70%; word-break:break-word;'>"
                f"<span style='vertical-align:middle; font-size:1.5em;'>🤖</span> {chat['message']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.text_input("Question", key="t5_question", placeholder="What does a tiger eat?")
    st.text_input("Species", key="t5_species", placeholder="Tiger")
    ask_col, clear_col = st.columns([2, 1])
    with ask_col:
        st.button("Ask", on_click=handle_t5_chat_dual)
    with clear_col:
        st.button("Clear Chat", on_click=clear_chat)
    # --- History ---
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📜 Detection History")
        st.table(pd.DataFrame(st.session_state.prediction_history))

    # --- Admin Dashboard ---
    if is_admin:
        st.markdown("---")
        st.header("🛡️ Admin Dashboard")
        u = requests.get(TOTAL_USERS_URL).json()
        p = requests.get(TOTAL_PREDICTIONS_URL).json()
        d = requests.get(ALL_PREDICTIONS_URL).json()
        df_pred = pd.DataFrame(d)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Users", "CNN Mammals", "YOLO Insects", "T5 Chatbot", "Article Model"
        ])
        with tab1:
            st.subheader(f"👥 Total Users: {u.get('total_users', 0)}")
            if not df_pred.empty:
                users = df_pred.groupby("user_id").size().reset_index(name="Predictions")
                st.dataframe(users)
                st.download_button("Download as CSV", users.to_csv(index=False), "users.csv")
                user_to_del = st.selectbox("Select user to delete history", users["user_id"])
                if st.button("Delete User History"):
                    requests.post(DELETE_HISTORY_URL, data={"user_id": user_to_del})
                    st.success("User history deleted. Please refresh page.")
        with tab2:
            st.subheader("CNN Mammal Model Stats")
            mammal_list = [
                "Cheetah", "Fox", "Hyena", "Lion", "Tiger",
                "Wolf", "Rhino", "Elephant", "Buffalo", "Zebra"
            ]
            mammal_df = df_pred[df_pred["predicted_class"].isin(mammal_list)]
            if not mammal_df.empty:
                st.bar_chart(mammal_df["predicted_class"].value_counts())
                st.dataframe(mammal_df)
                st.download_button("Download as CSV", mammal_df.to_csv(index=False), "cnn_predictions.csv")
        with tab3:
            st.subheader("YOLO Insect Model Stats")
            mammal_list = [
                "Cheetah", "Fox", "Hyena", "Lion", "Tiger",
                "Wolf", "Rhino", "Elephant", "Buffalo", "Zebra"
            ]
            insect_df = df_pred[~df_pred["predicted_class"].isin(mammal_list)]
            if not insect_df.empty:
                st.bar_chart(insect_df["predicted_class"].value_counts())
                st.dataframe(insect_df)
                st.download_button("Download as CSV", insect_df.to_csv(index=False), "yolo_predictions.csv")
        with tab4:
                st.subheader("T5 Chatbot QA History")
    # Replace with your real data fetching code below:
    t5_qa_data = [
        {"question": "What does a lion eat?", "species": "lion", "answer": "Lions eat prey.", "timestamp": "2025-07-23T10:00:00"},
        {"question": "Where does a fox live?", "species": "fox", "answer": "Foxes live everywhere.", "timestamp": "2025-07-23T10:01:00"},
        {"question": "How fast is a cheetah?", "species": "cheetah", "answer": "Cheetahs run up to 100 km/h.", "timestamp": "2025-07-23T10:05:00"},
    ]
    t5_df = pd.DataFrame(t5_qa_data)
    t5_df["timestamp"] = pd.to_datetime(t5_df["timestamp"])
    t5_df["question_length"] = t5_df["question"].str.split().str.len()
    t5_df["answer_length"] = t5_df["answer"].str.split().str.len()

    if not t5_df.empty:
        st.write("**QAs Over Time**")
        qa_counts = t5_df.groupby(t5_df['timestamp'].dt.date).size()
        st.line_chart(qa_counts, use_container_width=True)

        st.write("**Most Popular Species**")
        st.bar_chart(t5_df["species"].value_counts(), use_container_width=True)

        st.write("**Question Length Distribution**")
        fig_q_len = px.histogram(t5_df, x="question_length", nbins=10, title="Question Word Count Distribution")
        st.plotly_chart(fig_q_len, use_container_width=True)

        st.write("**Avg Answer Length per Species**")
        fig_a_len = px.bar(t5_df.groupby("species")["answer_length"].mean().reset_index(),
                           x="species", y="answer_length", title="Avg Answer Length")
        st.plotly_chart(fig_a_len, use_container_width=True)

        st.download_button("Download QA Data CSV", t5_df.to_csv(index=False), "t5_qa_analytics.csv")
    else:
        st.info("No T5 QA Data available.")
        with tab5:
            st.subheader("Article Generation Model Usage")

    # Replace with your real data fetching code below:
    article_data = [
        {"species": "lion", "article": "Lions are large cats found in Africa...", "timestamp": "2025-07-23T09:50:00"},
        {"species": "fox", "article": "Foxes are small, dog-like mammals...", "timestamp": "2025-07-23T09:55:00"},
        {"species": "cheetah", "article": "Cheetahs are the fastest land animals...", "timestamp": "2025-07-23T09:58:00"},
    ]
    article_df = pd.DataFrame(article_data)
    article_df["timestamp"] = pd.to_datetime(article_df["timestamp"])
    article_df["word_count"] = article_df["article"].str.split().str.len()
    article_df["unique_word_count"] = article_df["article"].apply(lambda x: len(set(x.split())))

    if not article_df.empty:
        st.write("**Article Count Over Time**")
        article_counts = article_df.groupby(article_df['timestamp'].dt.date).size()
        st.line_chart(article_counts, use_container_width=True)

        st.write("**Most Generated Species**")
        st.bar_chart(article_df["species"].value_counts(), use_container_width=True)

        st.write("**Article Word Count Distribution**")
        fig_wordcount = px.histogram(article_df, x="word_count", nbins=10, title="Article Word Count")
        st.plotly_chart(fig_wordcount, use_container_width=True)

        st.write("**Vocabulary Diversity**")
        fig_vocab = px.scatter(article_df, x="word_count", y="unique_word_count",
                               hover_data=["species"], title="Unique Words per Article")
        st.plotly_chart(fig_vocab, use_container_width=True)

        st.download_button("Download Article Data CSV", article_df.to_csv(index=False), "article_analytics.csv")
    else:
        st.info("No Article Analytics available.")


# --- Analytics Page ---
elif page == "📈 Analytics":
    st.title("📈 User Analytics")
    df = pd.DataFrame(st.session_state.prediction_history)
    if not df.empty:
        st.subheader("Temporal Analysis")
        st.line_chart(df["Confidence (%)"])  # or timestamp if you have it
        st.subheader("Class Distribution")
        st.bar_chart(df["Class"].value_counts())
        # Add more as needed!
    else:
        st.info("No predictions yet.")

