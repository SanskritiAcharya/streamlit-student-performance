"""Student Performance Analytics - Streamlit dashboard.

Run with:  streamlit run app.py
Dataset:   UCI Student Performance (Cortez, 2008)
"""
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
FILES = {"Math": "student-mat.csv", "Portuguese": "student-por.csv"}
PASS_MARK = 10
LABELS = {
    "studytime": "Weekly study time",
    "failures": "Past failures",
    "absences": "Absences",
    "goout": "Going out",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "higher": "Wants higher education",
    "internet": "Internet at home",
    "G1": "Period 1 grade",
    "G2": "Period 2 grade",
    "G3": "Final grade",
}

st.set_page_config(page_title="Student Performance Analytics", layout="wide")


# ---------- Data loading ----------
@st.cache_data
def load_sample() -> pd.DataFrame:
    frames = []
    for subject, file in FILES.items():
        df = pd.read_csv(DATA_DIR / file, sep=";")
        df["subject"] = subject
        frames.append(df)
    return prepare(pd.concat(frames, ignore_index=True))


@st.cache_data
def load_external(source, sep: str) -> pd.DataFrame:
    return prepare(pd.read_csv(source, sep=sep))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["G1", "G2", "G3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "subject" not in df:
        df["subject"] = "All"
    df = df.dropna(subset=["G3"])
    df["result"] = (df["G3"] >= PASS_MARK).map({True: "Pass", False: "Fail"})
    return df


# ---------- Model ----------
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = pd.get_dummies(df.drop(columns=["G3", "result"]), drop_first=True)
    y = df["G3"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred)}
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    return model, list(X.columns), metrics, importance


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Data source")
    source = st.radio("Source", ["Sample dataset", "Upload CSV", "CSV URL"], label_visibility="collapsed")
    try:
        if source == "Upload CSV":
            file = st.file_uploader("CSV file", type="csv")
            sep = st.selectbox("Delimiter", [";", ","])
            data = load_external(file, sep) if file else load_sample()
        elif source == "CSV URL":
            url = st.text_input("URL")
            sep = st.selectbox("Delimiter", [";", ","])
            data = load_external(url, sep) if url else load_sample()
        else:
            data = load_sample()
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    st.header("Filters")
    subjects = st.multiselect("Subject", sorted(data["subject"].unique()), default=sorted(data["subject"].unique()))
    schools = st.multiselect("School", sorted(data["school"].unique()), default=sorted(data["school"].unique()))
    sexes = st.multiselect("Sex", sorted(data["sex"].unique()), default=sorted(data["sex"].unique()))
    age = st.slider("Age", int(data["age"].min()), int(data["age"].max()), (int(data["age"].min()), int(data["age"].max())))

df = data[
    data["subject"].isin(subjects)
    & data["school"].isin(schools)
    & data["sex"].isin(sexes)
    & data["age"].between(*age)
]

# ---------- Header ----------
st.title("Student Performance Analytics")
st.caption("UCI Student Performance dataset (Cortez, 2008). Grades range from 0 to 20; a final grade of 10 or above is a pass.")

if df.empty:
    st.warning("No records match the current filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Students", f"{len(df):,}")
k2.metric("Mean final grade", f"{df['G3'].mean():.2f}")
k3.metric("Pass rate", f"{(df['result'] == 'Pass').mean():.1%}")
k4.metric("Mean absences", f"{df['absences'].mean():.1f}")

overview, explore, query, predict = st.tabs(["Overview", "Explore", "SQL Query", "Predict"])

RESULT_COLORS = alt.Scale(domain=["Pass", "Fail"], range=["#2E6F9E", "#C8553D"])

# ---------- Overview ----------
with overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Final grade distribution")
        hist = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("G3:Q", bin=alt.Bin(step=1), title="Final grade (G3)"),
                y=alt.Y("count():Q", title="Students"),
                color=alt.Color("result:N", scale=RESULT_COLORS, title=None),
                tooltip=["count():Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(hist, use_container_width=True)
    with right:
        st.subheader("Pass rate by subject and school")
        rate = (
            df.groupby(["subject", "school"])["result"]
            .apply(lambda s: (s == "Pass").mean())
            .reset_index(name="pass_rate")
        )
        bars = (
            alt.Chart(rate)
            .mark_bar()
            .encode(
                x=alt.X("school:N", title="School", axis=alt.Axis(labelAngle=0)),
                xOffset="subject:N",
                y=alt.Y("pass_rate:Q", title="Pass rate", axis=alt.Axis(format="%")),
                color=alt.Color("subject:N", title="Subject", scale=alt.Scale(range=["#2E6F9E", "#8FB8DE"])),
                tooltip=["subject", "school", alt.Tooltip("pass_rate:Q", format=".1%")],
            )
            .properties(height=320)
        )
        st.altair_chart(bars, use_container_width=True)

    with st.expander("View records"):
        st.dataframe(df, use_container_width=True, height=300)
        st.download_button("Download filtered data", df.to_csv(index=False), "students_filtered.csv", "text/csv")

# ---------- Explore ----------
with explore:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Final grade by factor")
        factor = st.selectbox(
            "Factor",
            ["studytime", "failures", "goout", "Medu", "Fedu", "higher", "internet", "romantic", "Dalc", "health"],
            format_func=lambda c: LABELS.get(c, c),
        )
        box = (
            alt.Chart(df)
            .mark_boxplot(size=28, color="#2E6F9E")
            .encode(x=alt.X(f"{factor}:O", title=LABELS.get(factor, factor), axis=alt.Axis(labelAngle=0)), y=alt.Y("G3:Q", title="Final grade"))
            .properties(height=340)
        )
        st.altair_chart(box, use_container_width=True)
    with c2:
        st.subheader("Relationship with final grade")
        x_var = st.selectbox("X axis", ["G2", "G1", "absences", "age"], format_func=lambda c: LABELS.get(c, c))
        brush = alt.selection_interval()
        scatter = (
            alt.Chart(df)
            .mark_circle(size=45, opacity=0.6)
            .encode(
                x=alt.X(f"{x_var}:Q", title=LABELS.get(x_var, x_var)),
                y=alt.Y("G3:Q", title="Final grade"),
                color=alt.condition(brush, alt.Color("result:N", scale=RESULT_COLORS, title=None), alt.value("lightgray")),
                tooltip=["school", "sex", "age", "studytime", "absences", "G1", "G2", "G3"],
            )
            .add_params(brush)
            .properties(height=340)
        )
        st.altair_chart(scatter, use_container_width=True)

    st.subheader("Correlation matrix")
    num_cols = ["age", "Medu", "Fedu", "studytime", "failures", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
    corr = df[num_cols].corr().reset_index().melt("index", var_name="var2", value_name="r")
    heat = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            x=alt.X("index:N", sort=num_cols, title=None),
            y=alt.Y("var2:N", sort=num_cols, title=None),
            color=alt.Color("r:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1]), title="r"),
            tooltip=["index", "var2", alt.Tooltip("r:Q", format=".2f")],
        )
        .properties(height=380)
    )
    text = heat.mark_text(fontSize=10).encode(text=alt.Text("r:Q", format=".1f"), color=alt.value("black"))
    st.altair_chart(heat + text, use_container_width=True)

# ---------- SQL Query ----------
with query:
    st.caption("The filtered data is available as the table `students`.")
    default_sql = (
        "SELECT subject, studytime,\n"
        "       COUNT(*) AS students,\n"
        "       ROUND(AVG(G3), 2) AS avg_final_grade,\n"
        "       ROUND(AVG(CASE WHEN G3 >= 10 THEN 1 ELSE 0 END) * 100, 1) AS pass_pct\n"
        "FROM students\n"
        "GROUP BY subject, studytime\n"
        "ORDER BY subject, studytime;"
    )
    sql = st.text_area("SQL", default_sql, height=180, label_visibility="collapsed")
    if st.button("Run query", type="primary"):
        try:
            con = duckdb.connect()
            con.register("students", df)
            st.dataframe(con.execute(sql).df(), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))

# ---------- Predict ----------
with predict:
    model, feature_cols, metrics, importance = train_model(data)
    m1, m2, _ = st.columns([1, 1, 2])
    m1.metric("Model R² (test set)", f"{metrics['r2']:.3f}")
    m2.metric("Mean absolute error", f"{metrics['mae']:.2f} points")

    form_col, imp_col = st.columns([1, 1])
    with form_col:
        st.subheader("Predict a student's final grade")
        with st.form("predict_form"):
            a, b = st.columns(2)
            g1 = a.number_input("Period 1 grade (G1)", 0, 20, 11)
            g2 = b.number_input("Period 2 grade (G2)", 0, 20, 11)
            studytime = a.select_slider("Weekly study time", [1, 2, 3, 4], 2,
                                        format_func=lambda v: {1: "<2 h", 2: "2-5 h", 3: "5-10 h", 4: ">10 h"}[v])
            failures = b.number_input("Past failures", 0, 4, 0)
            absences = a.number_input("Absences", 0, 93, 4)
            goout = b.slider("Going out (1-5)", 1, 5, 3)
            higher = a.selectbox("Wants higher education", ["yes", "no"])
            subject = b.selectbox("Subject", sorted(data["subject"].unique()))
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            # Start from the most typical student, then override with form inputs
            base = data.drop(columns=["G3", "result"]).mode().iloc[[0]].copy()
            for col, val in {"G1": g1, "G2": g2, "studytime": studytime, "failures": failures,
                             "absences": absences, "goout": goout, "higher": higher, "subject": subject}.items():
                base[col] = val
            row = pd.get_dummies(pd.concat([data.drop(columns=["G3", "result"]), base]), drop_first=True).tail(1)
            pred = float(model.predict(row.reindex(columns=feature_cols, fill_value=0))[0])
            outcome = "Pass" if pred >= PASS_MARK else "At risk"
            r1, r2 = st.columns(2)
            r1.metric("Predicted final grade", f"{pred:.1f} / 20")
            r2.metric("Outcome", outcome)

    with imp_col:
        st.subheader("Top predictors")
        imp = (
            alt.Chart(importance)
            .mark_bar(color="#2E6F9E")
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(height=340)
        )
        st.altair_chart(imp, use_container_width=True)
