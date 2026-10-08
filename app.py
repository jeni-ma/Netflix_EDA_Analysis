
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Netflix EDA Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("Netflix Movies & TV Shows - EDA Dashboard")
st.write("Explore Netflix data through statistics, charts and filters.")
@st.cache_data
def load_data():
    for filename in ["netflix_cleaned.csv", "netflix_titles.csv"]:
        if Path(filename).exists():
            data = pd.read_csv(filename)

            if "release_year" in data.columns:
                data["release_year"] = pd.to_numeric(
                    data["release_year"], errors="coerce"
                )

            for col in [
                "type", "country", "rating",
                "duration", "listed_in", "director", "cast"
            ]:
                if col in data.columns:
                    data[col] = data[col].fillna("Unknown").astype(str)

            return data, filename

    return None, None


df, source_file = load_data()

if df is None:
    st.error(
        "Dataset not found. Keep netflix_titles.csv "
        "in the same folder as app.py."
    )
    st.stop()

st.caption(f"Dataset: {source_file} | Rows: {len(df)} | Columns: {df.shape[1]}")
st.sidebar.header("Dashboard Filters")

types = sorted(df["type"].unique()) if "type" in df.columns else []
selected_types = st.sidebar.multiselect(
    "Content Type", types, default=types
)

if "release_year" in df.columns and df["release_year"].notna().any():
    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())

    if min_year < max_year:
        year_range = st.sidebar.slider(
            "Release Year",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = (min_year, max_year)
else:
    year_range = None

country_options = []
if "country" in df.columns:
    country_options = (
        df["country"]
        .str.split(", ")
        .explode()
        .replace("Unknown", pd.NA)
        .dropna()
        .value_counts()
        .head(30)
        .index.tolist()
    )

selected_country = st.sidebar.selectbox(
    "Country", ["All Countries"] + country_options
)
filtered = df.copy()

if selected_types and "type" in filtered.columns:
    filtered = filtered[filtered["type"].isin(selected_types)]
elif "type" in filtered.columns:
    filtered = filtered.iloc[0:0]

if year_range and "release_year" in filtered.columns:
    filtered = filtered[
        filtered["release_year"].between(
            year_range[0], year_range[1]
        )
        | filtered["release_year"].isna()
    ]

if selected_country != "All Countries" and "country" in filtered.columns:
    filtered = filtered[
        filtered["country"].str.contains(
            selected_country, case=False, regex=False, na=False
        )
    ]

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()
movies = (filtered["type"] == "Movie").sum()
shows = (filtered["type"] == "TV Show").sum()
years = (
    filtered["release_year"].nunique()
    if "release_year" in filtered.columns else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Titles", f"{len(filtered):,}")
c2.metric("Movies", f"{movies:,}")
c3.metric("TV Shows", f"{shows:,}")
c4.metric("Unique Release Years", f"{years:,}")

st.divider()
st.subheader("1. Statistical Summary")
st.write(
    "Count, mean, standard deviation, minimum, quartiles and maximum "
    "for numeric columns."
)

numeric_df = filtered.select_dtypes(include="number")

if not numeric_df.empty:
    st.dataframe(
        numeric_df.describe().round(2),
        use_container_width=True
    )
else:
    st.info("No numeric columns available.")
st.subheader("2. Categorical Summary")

summary_rows = []

for col in ["type", "rating", "country", "listed_in"]:
    if col in filtered.columns:
        mode_values = filtered[col].mode()

        summary_rows.append({
            "Column": col,
            "Unique Values": filtered[col].nunique(),
            "Most Frequent Value": (
                str(mode_values.iloc[0])
                if not mode_values.empty else "N/A"
            )
        })

if summary_rows:
    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True
    )
st.subheader("3. Content Type Distribution")

if "type" in filtered.columns:
    type_counts = filtered["type"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Movies vs TV Shows")
    st.pyplot(fig)
    plt.close(fig)
st.subheader("4. Titles by Release Year")

if "release_year" in filtered.columns:
    year_counts = (
        filtered["release_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .tail(20)
    )

    if not year_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(year_counts.index.astype(str), year_counts.values)
        ax.set_xlabel("Release Year")
        ax.set_ylabel("Number of Titles")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
left, right = st.columns(2)

with left:
    st.subheader("5. Top Ratings")

    if "rating" in filtered.columns:
        rating_counts = filtered["rating"].value_counts().head(10)
        st.bar_chart(rating_counts)

with right:
    st.subheader("6. Top Genres")

    if "listed_in" in filtered.columns:
        genre_counts = (
            filtered["listed_in"]
            .str.split(", ")
            .explode()
            .value_counts()
            .head(10)
        )
        st.bar_chart(genre_counts)
st.subheader("7. Top Countries")

if "country" in filtered.columns:
    country_counts = (
        filtered[filtered["country"] != "Unknown"]["country"]
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
    )

    st.bar_chart(country_counts)
st.subheader("8. Correlation Heatmap")

corr_data = pd.DataFrame(index=filtered.index)

if "release_year" in filtered.columns:
    corr_data["Release Year"] = filtered["release_year"]

if "duration" in filtered.columns:
    corr_data["Duration Number"] = pd.to_numeric(
        filtered["duration"].str.extract(r"(\d+)")[0],
        errors="coerce"
    )

corr_data = corr_data.dropna()

if len(corr_data) >= 2 and corr_data.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        corr_data.corr(),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title("Correlation Between Numeric Fields")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info(
        "Not enough complete numeric data for the correlation heatmap."
    )

st.caption(
    "Note: Duration mixes minutes for movies and seasons for TV shows. "
    "Interpret its correlation with release year cautiously."
)
st.subheader("9. Filtered Dataset Preview")
st.dataframe(filtered.head(100), use_container_width=True)
st.subheader("10. Download Filtered Data")

st.download_button(
    label="Download CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="netflix_filtered_data.csv",
    mime="text/csv"
)