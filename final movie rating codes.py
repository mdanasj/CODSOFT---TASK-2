import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

df = pd.read_csv("C:/Users/Anas/codesoft/movie rating.csv.csv", encoding="latin1")

df.drop(["Name"], axis=1, inplace=True)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
if df['Year'].notna().sum() > 0:
    if "Year" in df.columns and df["Year"].notna().sum() > 0:
        df["Year"] = df["Year"].fillna(df["Year"].median())
else:
    print("⚠️ 'Year' column is missing or empty—skipping imputation.")
    if 'Year' in df.columns:
        df.drop('Year', axis=1, inplace=True)
df["Duration"] = df["Duration"].str.extract(r"(\d+)").astype(float)
df["Genre"] = df["Genre"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Unknown")
df[["Actor 1", "Actor 2", "Actor 3"]] = df[["Actor 1", "Actor 2", "Actor 3"]].fillna("Unknown")

def parse_votes(value):
    try:
        value = str(value).replace("$", "").replace(",", "").strip()
        if "M" in value:
            return float(value.replace("M", "")) * 1_000_000
        elif "K" in value:
            return float(value.replace("K", "")) * 1_000
        return float(value)
    except:
        return np.nan

df["Votes"] = df["Votes"].apply(parse_votes)
df["Votes"] = df["Votes"].fillna(df["Votes"].median())
df["Votes"] = np.log1p(df["Votes"]) 

if 'Year' in df.columns:
    df.drop('Year', axis=1, inplace=True)
df["Duration"] = df["Duration"].fillna(df["Duration"].median())

df["All_Actors"] = df[["Actor 1", "Actor 2", "Actor 3"]].agg(" ".join, axis=1)

top_actors = ["Shah Rukh Khan", "Salman Khan", "Deepika Padukone", "Priyanka Chopra"]
for actor in top_actors:
    df[f"has_{actor.replace(' ', '_')}"] = df["All_Actors"].apply(lambda x: int(actor in x))

top_directors = df["Director"].value_counts().nlargest(10).index
df["Director"] = df["Director"].apply(lambda x: x if x in top_directors else "Other")

df = pd.get_dummies(df, columns=["Genre", "Director"], drop_first=True)

df.drop(["Actor 1", "Actor 2", "Actor 3", "All_Actors"], axis=1, inplace=True)

df = df[df['Rating'].notna()]

print(df.columns)
X_raw = df.drop(columns=[col for col in ['Rating', 'Year'] if col in df.columns])
y = df["Rating"]

if 'Year' in df.columns:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    if df['Year'].notna().sum() > 0:
        df['Year'] = df['Year'].fillna(df['Year'].median())
    else:
        df.drop('Year', axis=1, inplace=True)
else:
    print("🔍 'Year' column not found — skipping processing.")

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=150, max_depth=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"✅ Model RMSE: {rmse:.2f}")
print(f"✅ Model R² Score: {r2:.2f}")

import plotly.express as px
import pandas as pd

importances = pd.Series(model.feature_importances_, index=X_raw.columns).sort_values()
df_plot = pd.DataFrame({
    "Feature": importances.index,
    "Impact": importances.values
})

fig = px.bar(
    df_plot,
    x="Impact",
    y="Feature",
    orientation="h",
    title="📊 Feature Impact on Movie Ratings",
    text=df_plot["Impact"].round(3),
    color="Feature",  
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig.update_layout(
    height=900,
    title_font_size=24,
    margin=dict(l=120, r=40, t=60, b=40),
    xaxis_title="Importance Score",
    yaxis_title="Feature",
    yaxis=dict(tickfont=dict(size=12)),
    xaxis=dict(tickfont=dict(size=12))
)

fig.update_traces(textposition="outside")

fig.show()
