import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "processed"
if not DATA.exists(): DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
CLEAN = DATA / "cleaned_housing.csv"
ML = DATA / "ml_housing.csv"
CM = MODELS / "best_classification_model.pkl"
CP = MODELS / "classification_preprocessor.pkl"
CN = MODELS / "best_classification_model_name.txt"
RM = MODELS / "best_regression_model.pkl"
RP = MODELS / "regression_preprocessor.pkl"
RN = MODELS / "best_regression_model_name.txt"

@st.cache_data
def clean_df(): return pd.read_csv(CLEAN)

@st.cache_data
def ml_df(): return pd.read_csv(ML)

@st.cache_resource
def class_objects():
    m, p = joblib.load(CM), joblib.load(CP); n = CN.read_text().strip() if CN.exists() else "Best Classification Model"; return m, p, n

@st.cache_resource
def reg_objects():
    m, p = joblib.load(RM), joblib.load(RP); n = RN.read_text().strip() if RN.exists() else "Best Regression Model"; return m, p, n

def bar(s, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(10, 5)); s.plot(kind="bar", ax=ax); ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.tick_params(axis="x", rotation=45); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def hist(s, title, xlabel):
    fig, ax = plt.subplots(); ax.hist(s, bins=50); ax.set(title=title, xlabel=xlabel, ylabel="Count"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def feature_input(vals):
    df = clean_df(); age = max(0, 2025 - int(vals["Year_Built"])); amen = len(set(x.strip() for x in str(vals["Amenities"]).split(",") if x.strip())); floor = min(int(vals["Floor_No"]), int(vals["Total_Floors"]))
    med_pp, med_am, med_age = df["Price_per_SqFt"].median(), df["Amenity_Count"].median(), df["Age_of_Property"].median()
    x = {"State": vals["State"], "City": vals["City"], "Locality": vals["Locality"], "Property_Type": vals["Property_Type"], "BHK": vals["BHK"], "Size_in_SqFt": vals["Size_in_SqFt"], "Price_per_SqFt": vals["Price_per_SqFt"], "Year_Built": vals["Year_Built"], "Furnished_Status": vals["Furnished_Status"], "Floor_No": floor, "Total_Floors": vals["Total_Floors"], "Age_of_Property": age, "Nearby_Schools": vals["Nearby_Schools"], "Nearby_Hospitals": vals["Nearby_Hospitals"], "Public_Transport_Accessibility": vals["Public_Transport_Accessibility"], "Parking_Space": vals["Parking_Space"], "Security": vals["Security"], "Amenities": vals["Amenities"], "Facing": vals["Facing"], "Owner_Type": vals["Owner_Type"], "Availability_Status": vals["Availability_Status"], "Amenity_Count": amen}
    x.update({"Price_per_SqFt_Score": int(float(vals["Price_per_SqFt"]) <= med_pp), "BHK_Score": int(int(vals["BHK"]) >= 3), "Availability_Score": int(str(vals["Availability_Status"]).strip().lower() == "ready_to_move"), "Transport_Score": int(str(vals["Public_Transport_Accessibility"]).strip().lower() == "high"), "Parking_Score": int(str(vals["Parking_Space"]).strip().lower() in ["yes", "available"]), "Security_Score": int(str(vals["Security"]).strip().lower() in ["yes", "high", "gated", "cctv", "guard"]), "Amenity_Score": int(amen >= med_am), "Age_Score": int(age <= med_age), "Expected_Appreciation_Percentage": (((1.08) ** 5) - 1) * 100})
    return pd.DataFrame([x])

def align(x, p):
    if not hasattr(p, "feature_names_in_"): raise ValueError("Saved preprocessor does not contain feature_names_in_.")
    cols = list(p.feature_names_in_); miss = [c for c in cols if c not in x.columns]
    if miss: raise ValueError(f"Columns required by the saved preprocessor are missing: {miss}")
    return x[cols].copy()

def importance(model, prep, title):
    try:
        names = prep.get_feature_names_out(); vals = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(np.asarray(model.coef_)[0]) if hasattr(model, "coef_") else None
        if vals is None or len(names) != len(vals): return
        d = pd.Series(vals, index=names).sort_values(ascending=False).head(5); st.subheader(title); bar(d, "Top 5 Important Features", "Feature", "Importance")
    except Exception as e: st.info(f"Feature importance unavailable: {e}")

def inputs(prefix=""):
    df = clean_df(); states = sorted(df["State"].dropna().unique()); state = st.selectbox("State", states, index=states.index("Telangana") if "Telangana" in states else 0, key=prefix + "state")
    city_df = df[df["State"] == state]; cities = sorted(city_df["City"].dropna().unique()); city = st.selectbox("City", cities, index=cities.index("Hyderabad") if "Hyderabad" in cities else 0, key=prefix + "city")
    localities = sorted(city_df[city_df["City"] == city]["Locality"].dropna().unique()); locality = st.selectbox("Locality", localities, key=prefix + "locality")
    c1, c2, c3 = st.columns(3)
    with c1:
        ptype = st.selectbox("Property Type", sorted(df["Property_Type"].dropna().unique()), key=prefix + "ptype"); bhk = st.number_input("BHK", 1, 5, 2, 1, key=prefix + "bhk"); size = st.number_input("Size (Sq Ft)", int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max()), 1000, 50, key=prefix + "size")
    with c2:
        pps_max = float(df["Price_per_SqFt"].max()); pps_default = float(df["Price_per_SqFt"].median()); pps = st.number_input("Price per Sq Ft (dataset units)", 0.0, pps_max, pps_default, 0.01, key=prefix + "pps"); year = st.number_input("Year Built", 1990, 2025, 2018, 1, key=prefix + "year"); furnished = st.selectbox("Furnished Status", sorted(df["Furnished_Status"].dropna().unique()), key=prefix + "furnished"); total = st.number_input("Total Floors", 1, int(df["Total_Floors"].max()), 5, 1, key=prefix + "total")
    with c3:
        floor = st.number_input("Floor No", 0, int(total), min(1, int(total)), 1, key=prefix + "floor"); schools = st.number_input("Nearby Schools", int(df["Nearby_Schools"].min()), int(df["Nearby_Schools"].max()), 2, 1, key=prefix + "schools"); hospitals = st.number_input("Nearby Hospitals", int(df["Nearby_Hospitals"].min()), int(df["Nearby_Hospitals"].max()), 2, 1, key=prefix + "hospitals"); transport = st.selectbox("Public Transport Accessibility", sorted(df["Public_Transport_Accessibility"].dropna().unique()), key=prefix + "transport"); parking = st.selectbox("Parking Space", sorted(df["Parking_Space"].dropna().unique()), key=prefix + "parking"); security = st.selectbox("Security", sorted(df["Security"].dropna().unique()), key=prefix + "security"); amenities = st.text_input("Amenities", "Gym, Pool", key=prefix + "amenities"); facing = st.selectbox("Facing", sorted(df["Facing"].dropna().unique()), key=prefix + "facing"); owner = st.selectbox("Owner Type", sorted(df["Owner_Type"].dropna().unique()), key=prefix + "owner"); availability = st.selectbox("Availability Status", sorted(df["Availability_Status"].dropna().unique()), key=prefix + "availability")
    return {"State": state, "City": city, "Locality": locality, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size, "Price_per_SqFt": pps, "Year_Built": year, "Furnished_Status": furnished, "Floor_No": floor, "Total_Floors": total, "Nearby_Schools": schools, "Nearby_Hospitals": hospitals, "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security, "Amenities": amenities, "Facing": facing, "Owner_Type": owner, "Availability_Status": availability}

def apply_filters(df):
    st.sidebar.header("Filters")
    states = st.sidebar.multiselect("State", sorted(df["State"].unique()))
    cities = st.sidebar.multiselect("City", sorted(df["City"].unique()))
    types = st.sidebar.multiselect("Property Type", sorted(df["Property_Type"].unique()))
    size_min, size_max = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max()); area = st.sidebar.slider("Area / Size (Sq Ft)", size_min, size_max, (size_min, size_max), 50)
    price_min, price_max = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max()); price = st.sidebar.slider("Price (Lakhs)", price_min, price_max, (price_min, price_max), 1.0)
    bhks = st.sidebar.multiselect("BHK", sorted(df["BHK"].unique()))
    if states: df = df[df["State"].isin(states)]
    if cities: df = df[df["City"].isin(cities)]
    if types: df = df[df["Property_Type"].isin(types)]
    df = df[df["Size_in_SqFt"].between(*area) & df["Price_in_Lakhs"].between(*price)]
    if bhks: df = df[df["BHK"].isin(bhks)]
    st.sidebar.caption(f"Filtered properties: {len(df):,}")
    return df

page = st.sidebar.radio("Go to", ["Introduction", "EDA Visualizations", "Classification", "Regression"])

if page == "Introduction":
    st.title("🏠 Real Estate Investment Advisor"); st.subheader("Predicting Property Profitability & Future Value"); st.markdown("---"); st.header("Project Overview"); st.write("This application analyzes residential property data and provides machine-learning based investment classification and five-year future price estimation.")
    a, b = st.columns(2)
    with a: st.markdown("**Analytics**\n- Exploratory Data Analysis\n- Property price analysis\n- Property size analysis\n- Location analysis\n- Amenities and accessibility analysis\n- Area, price and BHK filters")
    with b: st.markdown("**Machine Learning**\n- Good Investment classification\n- Five-year future price prediction\n- Model confidence\n- Model feature insights")
    a, b = st.columns(2)
    with a: st.metric("Classification Target", "Good_Investment")
    with b: st.metric("Regression Target", "Future_Price_5Y")
    st.header("Five-Year Growth Assumption"); st.info(f"The project uses an 8% annual growth assumption for five years, resulting in {(((1.08) ** 5) - 1) * 100:.2f}% compound appreciation.")
    st.header("Technologies Used"); st.write("Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, MLflow and Streamlit.")

elif page == "EDA Visualizations":
    st.title("📊 Exploratory Data Analysis")
    try: df = apply_filters(clean_df()); st.success(f"Dataset loaded: {len(df):,} filtered rows")
    except Exception as e: st.error(f"Unable to load cleaned dataset: {e}"); st.stop()
    if df.empty: st.warning("No properties match the selected filters."); st.stop()
    st.subheader("1. Distribution of Property Prices"); hist(df["Price_in_Lakhs"], "Q1 - Distribution of Property Prices", "Price (Lakhs)")
    st.subheader("2. Distribution of Property Sizes"); hist(df["Size_in_SqFt"], "Q2 - Distribution of Property Sizes", "Size (Sq Ft)")
    st.subheader("3. Price per Sq Ft by Property Type"); bar(df.groupby("Property_Type")["Price_per_SqFt"].mean().sort_values(ascending=False), "Q3 - Average Price per Sq Ft by Property Type", "Property Type", "Average Price per Sq Ft")
    st.subheader("4. Relationship Between Property Size and Price"); fig, ax = plt.subplots(); ax.scatter(df["Size_in_SqFt"], df["Price_in_Lakhs"], alpha=.3); ax.set(title="Q4 - Property Size vs Property Price", xlabel="Size (Sq Ft)", ylabel="Price (Lakhs)"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("5. Outliers in Price per Sq Ft and Property Size"); fig, ax = plt.subplots(figsize=(10, 5)); ax.boxplot([df["Size_in_SqFt"].dropna(), df["Price_per_SqFt"].dropna()], tick_labels=["Size (Sq Ft)", "Price per Sq Ft"]); ax.set_title("Q5 - Outlier Detection"); ax.set_ylabel("Value"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("6. Average Price per Sq Ft by State"); bar(df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False), "Q6 - Average Price per Sq Ft by State", "State", "Average Price per Sq Ft")
    st.subheader("7. Average Property Price by City"); bar(df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False), "Q7 - Average Property Price by City", "City", "Average Price (Lakhs)")
    st.subheader("8. Median Age of Properties by Locality"); d = df.groupby("Locality")["Age_of_Property"].median().sort_values(ascending=False).head(20).sort_values(); fig, ax = plt.subplots(figsize=(10, 7)); d.plot(kind="barh", ax=ax); ax.set(title="Q8 - Top 20 Localities by Median Property Age", xlabel="Median Property Age", ylabel="Locality"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("9. BHK Distribution Across Cities"); d = pd.crosstab(df["City"], df["BHK"]); fig, ax = plt.subplots(figsize=(12, 6)); d.plot(kind="bar", stacked=True, ax=ax); ax.set(title="Q9 - BHK Distribution Across Cities", xlabel="City", ylabel="Number of Properties"); ax.tick_params(axis="x", rotation=75); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("10. Top 5 Most Expensive Localities"); d = df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(5).sort_values(); fig, ax = plt.subplots(figsize=(10, 5)); d.plot(kind="barh", ax=ax); ax.set(title="Q10 - Top 5 Most Expensive Localities", xlabel="Average Property Price (Lakhs)", ylabel="Locality"); plt.tight_layout(); st.pyplot(fig); plt.close(fig); st.caption("The dataset has no date/time column, so Q10 compares the top 5 expensive localities rather than a historical time trend.")
    st.subheader("11. Correlation Between Numeric Features"); cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built", "Age_of_Property", "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count"]; fig, ax = plt.subplots(figsize=(11, 8)); sns.heatmap(df[cols].corr(), cmap="coolwarm", ax=ax); ax.set_title("Q11 - Correlation Matrix of Numeric Features"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("12. Nearby Schools vs Price per Sq Ft"); fig, ax = plt.subplots(); ax.scatter(df["Nearby_Schools"], df["Price_per_SqFt"], alpha=.3); ax.set(title="Q12 - Nearby Schools vs Price per Sq Ft", xlabel="Nearby Schools", ylabel="Price per Sq Ft"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("13. Nearby Hospitals vs Price per Sq Ft"); fig, ax = plt.subplots(); ax.scatter(df["Nearby_Hospitals"], df["Price_per_SqFt"], alpha=.3); ax.set(title="Q13 - Nearby Hospitals vs Price per Sq Ft", xlabel="Nearby Hospitals", ylabel="Price per Sq Ft"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("14. Price by Furnished Status"); bar(df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().sort_values(ascending=False), "Q14 - Average Price by Furnished Status", "Furnished Status", "Average Price (Lakhs)")
    st.subheader("15. Price per Sq Ft by Facing Direction"); bar(df.groupby("Facing")["Price_per_SqFt"].mean().sort_values(ascending=False), "Q15 - Price per Sq Ft by Facing Direction", "Facing Direction", "Average Price per Sq Ft")
    st.subheader("16. Number of Properties by Owner Type"); bar(df["Owner_Type"].value_counts(), "Q16 - Number of Properties by Owner Type", "Owner Type", "Number of Properties")
    st.subheader("17. Number of Properties by Availability Status"); bar(df["Availability_Status"].value_counts(), "Q17 - Properties by Availability Status", "Availability Status", "Number of Properties")
    st.subheader("18. Parking Space vs Property Price"); bar(df.groupby("Parking_Space")["Price_in_Lakhs"].mean().sort_index(), "Q18 - Parking Space vs Property Price", "Parking Space", "Average Property Price (Lakhs)")
    st.subheader("19. Amenities vs Price per Sq Ft"); fig, ax = plt.subplots(); ax.scatter(df["Amenity_Count"], df["Price_per_SqFt"], alpha=.3); ax.set(title="Q19 - Amenities vs Price per Sq Ft", xlabel="Number of Amenities", ylabel="Price per Sq Ft"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("20. Public Transport Accessibility vs Price per Sq Ft"); bar(df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean().sort_values(ascending=False), "Q20 - Public Transport Accessibility vs Price per Sq Ft", "Transport Accessibility", "Average Price per Sq Ft")

elif page == "Classification":
    st.title("🏠 Property Investment Prediction"); st.write("Enter the property details to predict the Good_Investment classification.")
    try: model, prep, name = class_objects()
    except Exception as e: st.error(f"Unable to load the classification model or preprocessor: {e}"); st.stop()
    st.info(f"Loaded model: {name}"); st.subheader("Property Details"); vals = inputs("cls_")
    if st.button("Predict Investment", type="primary"):
        try:
            x = align(feature_input(vals), prep); z = prep.transform(x); pred = model.predict(z)[0]; conf = float(np.max(model.predict_proba(z)[0]) * 100) if hasattr(model, "predict_proba") else None
            st.subheader("Investment Decision"); (st.success if pred == 1 else st.warning)("Good Investment" if pred == 1 else "Not a Good Investment")
            if conf is not None: st.info(f"Model Confidence: {conf:.2f}%")
            st.subheader("Engineered Features Used"); st.dataframe(x.T.rename(columns={0: "Value"}), use_container_width=True); importance(model, prep, "Classification Feature Insights")
        except Exception as e: st.error(f"Prediction error: {e}")

elif page == "Regression":
    st.title("📈 Future Property Price Prediction"); st.write("Enter the same property information to estimate the property value after five years.")
    try: model, prep, name = reg_objects()
    except Exception as e: st.error(f"Unable to load the regression model or preprocessor: {e}"); st.stop()
    st.info(f"Loaded model: {name}"); st.subheader("Property Details"); vals = inputs("reg_")
    if st.button("Predict Future Price", type="primary"):
        try:
            x = align(feature_input(vals), prep); z = prep.transform(x); future = float(model.predict(z)[0]); current = vals["Price_per_SqFt"] * vals["Size_in_SqFt"]
            st.subheader("Estimated Property Value After 5 Years"); st.success(f"₹ {future:,.2f} Lakhs")
            a, b, c = st.columns(3)
            with a: st.metric("Input Price per Sq Ft", f"{vals['Price_per_SqFt']:.2f} dataset units")
            with b: st.metric("Estimated Current Price", f"₹ {current:,.2f} Lakhs")
            with c: st.metric("5-Year Appreciation Assumption", "46.93%")
            st.info("The project uses an 8% annual growth assumption for the five-year forecast."); importance(model, prep, "Regression Feature Insights")
        except Exception as e: st.error(f"Prediction error: {e}")
