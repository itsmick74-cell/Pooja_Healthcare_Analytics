"""Healthcare Analytics for Doctor Visits - AICTE TIRTC DIY Project"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

PATH = "1776250375-P2-Healthcare_Analytics_for_Doctor_Visits__1_.csv"
import os
os.makedirs("out/charts/", exist_ok=True)
OUT = "out/charts/"
sns.set_theme(style="whitegrid", palette="Blues_r")

# 1. Load & clean
df = pd.read_csv(PATH).drop(columns=["Unnamed: 0"])
df["age_years"] = (df["age"] * 100).round().astype(int)
for c in ["private", "freepoor", "freerepat", "nchronic", "lchronic"]:
    df[c] = (df[c] == "yes").astype(int)
df["gender"] = (df["gender"] == "female").astype(int)
df["visited"] = (df["visits"] > 0).astype(int)
print(df.shape, df.isna().sum().sum())

# 2. EDA
fig, ax = plt.subplots(figsize=(7, 4))
df["visits"].value_counts().sort_index().plot.bar(ax=ax, color="#1C7293")
ax.set(title="Distribution of Doctor Visits", xlabel="Visits", ylabel="People")
plt.tight_layout(); plt.savefig(OUT + "visits_dist.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
df.groupby("illness")["visits"].mean().plot.bar(ax=ax, color="#065A82")
ax.set(title="Average Visits by Number of Illnesses", xlabel="Illnesses", ylabel="Avg visits")
plt.tight_layout(); plt.savefig(OUT + "illness_visits.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
df.groupby(pd.cut(df["reduced"], [-1, 0, 2, 5, 20], labels=["0", "1-2", "3-5", "6+"]), observed=True)["visits"].mean().plot.bar(ax=ax, color="#1C7293")
ax.set(title="Average Visits by Days of Reduced Activity", xlabel="Reduced-activity days", ylabel="Avg visits")
plt.tight_layout(); plt.savefig(OUT + "reduced_visits.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
df.groupby(pd.cut(df["age_years"], [0, 25, 35, 50, 65, 100], labels=["<=25", "26-35", "36-50", "51-65", "65+"]), observed=True)["visits"].mean().plot.bar(ax=ax, color="#065A82")
ax.set(title="Average Visits by Age Group", xlabel="Age group", ylabel="Avg visits")
plt.tight_layout(); plt.savefig(OUT + "age_visits.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
df.groupby(pd.qcut(df["income"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]), observed=True)["visits"].mean().plot.bar(ax=ax, color="#1C7293")
ax.set(title="Average Visits by Income Quartile", xlabel="Income quartile", ylabel="Avg visits")
plt.tight_layout(); plt.savefig(OUT + "income_visits.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.drop(columns=["age_years", "visited"]).corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(OUT + "corr_heatmap.png", dpi=160); plt.close()

# 3. Poisson regression (visits is a count variable)
features = ["gender", "age", "income", "illness", "reduced", "health", "private", "freepoor", "freerepat", "nchronic", "lchronic"]
X = sm.add_constant(df[features])
poisson = sm.GLM(df["visits"], X, family=sm.families.Poisson()).fit()
print(poisson.summary())
irr = np.exp(poisson.params).drop("const").sort_values()
print(irr)

# 4. Predict whether a person visits a doctor
Xtr, Xte, ytr, yte = train_test_split(df[features], df["visited"], test_size=0.2, random_state=42, stratify=df["visited"])
rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42).fit(Xtr, ytr)
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
for name, m in [("Random Forest", rf), ("Logistic Regression", lr)]:
    p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]
    print(name, "acc=%.3f auc=%.3f" % (accuracy_score(yte, p), roc_auc_score(yte, pr)))
print(classification_report(yte, rf.predict(Xte)))

imp = pd.Series(rf.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.5))
imp.plot.barh(ax=ax, color="#1C7293")
ax.set(title="Random Forest Feature Importance", xlabel="Importance")
plt.tight_layout(); plt.savefig(OUT + "feature_importance.png", dpi=160); plt.close()
