# IMPORTS NECESARIOS --------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib
import json

# VARIABLES -------------------------------------------------------------------------------------------

ROOT = Path.cwd()
SAMPLE_DIR = ROOT / "src" / "data_raw"

def load_dataset():
    csv_files = sorted(SAMPLE_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No hay CSV en: {SAMPLE_DIR}")
    df = pd.read_csv(csv_files[0])
    print(f"✅ Dataset cargado: {csv_files[0].name}")
    return df

# FUNCIONES -------------------------------------------------------------------------------------------

def audit_clean(df, drop_missing=0.6, cat_unique_max=7):
    """
    Auditoría + limpieza básica:
    - Convierte strings vacíos a NaN
    - Elimina duplicados
    - Muestra missing (%)
    - Elimina columnas con muchos missing
    - Dice qué columnas son numéricas o categóricas (simple)
    - Muestra df.info() final
    """
    df = df.copy()

    print("=== AUDITORÍA ===")
    print("Shape inicial:", df.shape)

    # vacíos -> NaN
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    # duplicados
    dup = df.duplicated().sum()
    print("Duplicados:", dup)
    df = df.drop_duplicates()
    print("Shape tras quitar duplicados:", df.shape)

    # clasificación numérica / categórica
    print("\n=== TIPOS DE COLUMNAS (simple) ===")
    col_type = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            nuniq = df[col].nunique(dropna=True)
            if nuniq <= cat_unique_max:
                col_type[col] = "categórica"
            else:
                col_type[col] = "numérica"
        else:
            col_type[col] = "categórica"

    # imprimir tipo por columna
    for col, t in col_type.items():
        print(f"{col} -> {t}")

    # listas finales
    num_cols = [c for c, t in col_type.items() if t == "numérica"]
    cat_cols = [c for c, t in col_type.items() if t == "categórica"]

    print("\nNuméricas:", num_cols)
    print("\nCategóricas:", cat_cols)

    # missing por columna (top 10)
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    print("\nTop 10 columnas con más missing (%):")
    print(missing.head(10).round(2))

    # eliminar columnas con muchos missing
    cols_drop = missing[missing >= drop_missing * 100].index.tolist()
    if len(cols_drop) > 0:
        print(f"\nEliminando columnas con >= {int(drop_missing*100)}% missing:")
        print(cols_drop)
        df = df.drop(columns=cols_drop)
        print("Shape tras eliminar columnas:", df.shape)
    else:
        print(f"\nNo se elimina ninguna columna (umbral {int(drop_missing*100)}%).")

    print("\n=== INFO FINAL ===")
    df.info()
    return df

def save_clean_dataset(df, filename="clean_dataset.csv", folder="src/data_processed"):
    """
    Guarda el dataframe limpio en un CSV dentro de la carpeta indicada.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / filename
    df.to_csv(path, index=False)

    print(f"✅ Dataset guardado en: {path}")
    return

def make_xy(df, target_col, min_class=25):
    df = df.copy()
    df = df.dropna(subset=[target_col])

    y = df[target_col].astype(str).str.strip()
    X = df.drop(columns=[target_col])

    # filtrar clases raras
    vc = y.value_counts()
    keep = vc[vc >= min_class].index
    mask = y.isin(keep)
    X, y = X.loc[mask].copy(), y.loc[mask].copy()

    print("Target:", target_col)
    print("Clases tras filtrar (>= %d):" % min_class, y.nunique())
    print(y.value_counts().head(10))
    return X, y

def split_data(X, y, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    return X_train, X_test, y_train, y_test

def save_show(fig_name, img_dir):
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(img_dir / fig_name, dpi=150, bbox_inches="tight")
    plt.show()

def data_report(df):
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=True),
    })
    rep["cardin_%"] = (rep["unique"] / len(df) * 100).round(2)
    return rep.sort_values(["missing_%", "cardin_%"], ascending=False)

def outlier_report(X_train, max_cols=6):
    """
    Reporte simple de outliers por IQR (solo numéricas).
    Imprime cuántos outliers hay por columna y enseña boxplots de algunas.
    """
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    print("=== OUTLIERS (IQR) - TRAIN ===")

    if len(num_cols) == 0:
        print("No hay columnas numéricas.")
        return

    for col in num_cols:
        s = X_train[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        out = ((s < low) | (s > high)).sum()
        pct = out / len(s) * 100
        print(f"{col}: outliers = {out} ({pct:.2f}%)")

    # Boxplots de algunas
    for col in num_cols[:max_cols]:
        plt.figure(figsize=(6,3))
        plt.boxplot(X_train[col].dropna(), vert=False)
        plt.title(f"Boxplot (TRAIN): {col}")
        plt.tight_layout()
        plt.show()


def clip_outliers_iqr(X_train, X_test, k=1.5):
    """
    Tratamiento simple: recorta (clip) outliers usando límites calculados SOLO en TRAIN.
    Devuelve X_train_clip, X_test_clip.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    print("\n=== CLIPPING OUTLIERS (IQR) ===")

    for col in num_cols:
        s = X_train[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr

        before_train = ((X_train[col] < low) | (X_train[col] > high)).sum()
        before_test = ((X_test[col] < low) | (X_test[col] > high)).sum()

        X_train[col] = X_train[col].clip(low, high)
        X_test[col] = X_test[col].clip(low, high)

        if before_train > 0 or before_test > 0:
            print(f"{col}: recortados train={int(before_train)} | test={int(before_test)}")

    return X_train, X_test

def feature_relevance_mi(X_train, y_train, top=10):
    """
    Ranking simple de relevancia con Mutual Information.
    - Numéricas: se usan tal cual (rellenando NaN con mediana por si acaso)
    - Categóricas: se codifican con factorize (simple)
    """
    X = X_train.copy()

    # rellenar NaN por si acaso (aunque ya lo imputéis)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna("MISSING")

    # codificar categóricas a enteros
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    for col in cat_cols:
        X[col] = pd.factorize(X[col])[0]

    mi = mutual_info_classif(X.values, y_train.values, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    print("\n=== FEATURES MÁS RELEVANTES (Mutual Information) ===")
    print(mi_series.head(top).round(4).to_string())

    # plot simple
    mi_series.head(top).sort_values().plot(kind="barh", figsize=(7,4), title="Top features (MI)")
    plt.tight_layout()
    plt.show()

    return mi_series

def eda_basic(X_train, y_train, target_name="target", img_dir="src/img"):
    """
    EDA básico (como el main):
    - distribución del target
    - missings
    - hist numéricas
    - conteos de categóricas (pocas, para que se vea)
    - correlación numéricas
    - boxplot de 3 numéricas vs target
    - stacked bar de 2 categóricas vs target (si hay)
    """
    impute_missing_max = 0.06
    cat_unique_max=7
    X_train = X_train.copy()
    
    print("\n=== EDA (solo TRAIN) ===")

    # 1) Target distribution
    plt.figure(figsize=(10,4))
    y_train.value_counts().head(15).plot(kind="bar")
    plt.title("Distribución del target (Top 15) - TRAIN")
    save_show("target_top15_train.png", img_dir)

    # 2) Missing %
    miss = (X_train.isna().mean() * 100).sort_values(ascending=False)
    plt.figure(figsize=(10,4))
    miss.head(20).plot(kind="bar")
    plt.title("Top 20 missing (%) - TRAIN")
    save_show("missing_top20_train.png", img_dir)
    # imputación para columnas con poco missing
    col_type = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            nuniq = X_train[col].nunique(dropna=True)
            if nuniq <= cat_unique_max:
                col_type[col] = "categórica"
            else:
                col_type[col] = "numérica"
        else:
            col_type[col] = "categórica"
    missing_ratio = X_train.isna().mean()
    cols_impute = [c for c in X_train.columns if 0 < missing_ratio[c] <= impute_missing_max]

    if cols_impute:
        print(f"Imputando columnas con <= {impute_missing_max*100:.1f}% missing:")
        for col in cols_impute:
            if col_type[col] == "numérica":
                value = X_train[col].median()
                X_train[col] = X_train[col].fillna(value)
                print(f"- {col}: mediana = {value}")
            else:
                # moda (si hay empate, coge la primera)
                mode_vals = X_train[col].mode(dropna=True)
                value = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
                X_train[col] = X_train[col].fillna(value)
                print(f"- {col}: moda = {value}")
    else:
        print(f"No hay columnas con missing <= {impute_missing_max*100:.1f}% para imputar.")

    # resumen de missing final
    total_missing = int(X_train.isna().sum().sum())
    print("\nMissing total (celdas) tras imputar:", total_missing)

    # 3) Report tabla
    rep = data_report(X_train)
    print("\nData report (top 20):")
    print(rep.head(20).to_string())

    # 4) num / cat
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    print("\nNuméricas:", len(num_cols), "| Categóricas:", len(cat_cols))

    # 5) Hist numéricas (primeras 12)
    if len(num_cols) > 0:
        X_train[num_cols[:12]].hist(figsize=(14,8), bins=30)
        plt.suptitle("Histogramas numéricas (primeras 12) - TRAIN", y=1.02)
        save_show("hist_numericas_train.png", img_dir)

    # 6) Conteos categóricas (elegimos 4 con menos cardinalidad)
    if len(cat_cols) > 0:
        cat_for_plot = X_train[cat_cols].nunique().sort_values().head(4).index.tolist()
        for c in cat_for_plot:
            plt.figure(figsize=(8,4))
            X_train[c].value_counts(dropna=False).head(20).plot(kind="bar")
            plt.title(f"Conteo {c} (Top 20) - TRAIN")
            save_show(f"count_{c}_train.png", img_dir)

    # 7) Correlación
    if len(num_cols) > 1:
        plt.figure(figsize=(10,7))
        sns.heatmap(X_train[num_cols].corr(), cmap="coolwarm", center=0)
        plt.title("Matriz de correlación (numéricas) - TRAIN")
        save_show("corr_train.png", img_dir)

    # 8) Boxplot num vs target (primeras 3 numéricas)
    for col in num_cols[:3]:
        plt.figure(figsize=(10,4))
        sns.boxplot(x=y_train, y=X_train[col])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{col} vs {target_name} - TRAIN")
        save_show(f"box_{col}_vs_{target_name}_train.png", img_dir)

    # 9) Categórica vs target (stacked) (2 con menos niveles)
    if len(cat_cols) > 0:
        cat_for_stacked = X_train[cat_cols].nunique().sort_values().head(2).index.tolist()
        for c in cat_for_stacked:
            plt.figure(figsize=(9,4))
            tmp = pd.crosstab(X_train[c].fillna("MISSING"), y_train, normalize="index")
            tmp.plot(kind="bar", stacked=True)
            plt.title(f"{c} vs {target_name} (proporciones) - TRAIN")
            plt.ylabel("Proporción")
            save_show(f"stacked_{c}_vs_{target_name}_train.png", img_dir)

def preliminary_feature_reduction(X_train, X_test,
                                  max_miss_pct=40.0, max_cardin_pct=60.0, max_cat_levels=200):
    """
    Muy parecido a la guía:
    - elimina constantes
    - elimina columnas con muchos missings
    - elimina columnas tipo ID (casi únicas)
    - elimina categóricas con demasiados niveles
    """
    rep = data_report(X_train)

    const_cols = rep[rep["unique"] <= 1].index.tolist()
    miss_cols = rep[rep["missing_%"] > max_miss_pct].index.tolist()
    id_like_cols = rep[rep["cardin_%"] > max_cardin_pct].index.tolist()

    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    high_levels = [c for c in cat_cols if X_train[c].nunique(dropna=True) > max_cat_levels]

    drop_cols = sorted(set(const_cols + miss_cols + id_like_cols + high_levels))

    print("\n=== Reducción preliminar ===")
    print("Constantes:", len(const_cols))
    print("Missing > %.1f%%:" % max_miss_pct, len(miss_cols))
    print("ID-like (cardin > %.1f%%):" % max_cardin_pct, len(id_like_cols))
    print("Cat con > %d niveles:" % max_cat_levels, len(high_levels))
    print("TOTAL a eliminar:", len(drop_cols))
    if drop_cols:
        print("Drop cols:", drop_cols)

    X_train_red = X_train.drop(columns=drop_cols, errors="ignore")
    X_test_red = X_test.drop(columns=drop_cols, errors="ignore")
    return X_train_red, X_test_red, drop_cols

def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols),
    ])
    print("Preprocessor listo. Num:", len(num_cols), "| Cat:", len(cat_cols))
    return pre


def compare_models_cv(X_train, y_train, preprocessor, seed=42):
    """
    CV + comparación como el ejemplo.
    Métrica principal: f1_macro (por desbalanceo).
    """
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogReg": LogisticRegression(max_iter=3000),
        "RF": RandomForestClassifier(random_state=seed),
        "GBC": GradientBoostingClassifier(random_state=seed),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    pipes = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipes[name] = pipe

        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"f1":"f1_macro", "acc":"accuracy"},
            n_jobs=-1
        )

        results.append({
            "model": name,
            "f1_macro": scores["test_f1"].mean(),
            "accuracy": scores["test_acc"].mean()
        })

    leaderboard = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    print("\n=== Leaderboard (CV) ===")
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["model"]
    print("Mejor por f1_macro:", best_name)
    return pipes, leaderboard, best_name

def tune_model(best_name, pipe, X_train, y_train, seed=42):
    """RandomizedSearchCV sencillo (según el modelo ganador)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    if best_name == "LogReg":
        params = {"model__C": np.logspace(-3, 2, 25)}
    elif best_name == "RF":
        params = {
            "model__n_estimators": [300, 600, 900],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        }
    else:  # GBC (o lo que sea)
        params = {
            "model__n_estimators": [150, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        }

    search = RandomizedSearchCV(
        pipe, params,
        n_iter=20,
        cv=cv,
        scoring="f1_macro",
        random_state=seed,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    print("\nMejor f1_macro CV:", search.best_score_)
    print("Mejores params:", search.best_params_)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def evaluate_model(model, X_test, y_test, img_dir="src/img"):
    preds = model.predict(X_test)

    print("\n=== TEST report ===")
    print(classification_report(y_test, preds))

    plt.figure(figsize=(8,6))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, xticks_rotation=45)
    save_show("conf_matrix_test.png", img_dir)

    return preds


def save_model_and_card(model, model_path="src/models/best_model.joblib", card_path="src/models/model_card.json", info=None):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    print("✅ Modelo guardado en:", model_path)

    if info is None:
        info = {}
    card_path = Path(card_path)
    card_path.parent.mkdir(parents=True, exist_ok=True)

    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("🧾 Model card guardada en:", card_path)

def train_detection_method(X_train, y_train, X_test, y_test, seed=42):
    """
    Wrapper específico para detection_method:
    - modelos base + CV
    - tuning
    - test
    - guardado
    """
    pre = build_preprocessor(X_train)
    pipes, leaderboard, best_name = compare_models_cv(X_train, y_train, pre, seed=seed)

    # si gana Dummy, forzamos LogReg (para que sea un modelo "de verdad")
    if best_name == "Dummy":
        best_name = "LogReg"

    best_model, best_params, best_cv = tune_model(best_name, pipes[best_name], X_train, y_train, seed=seed)
    evaluate_model(best_model, X_test, y_test, img_dir="src/img")

    info = {
        "target": "detection_method",
        "best_model": best_name,
        "best_params": best_params,
        "best_cv_f1_macro": best_cv,
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
    }
    save_model_and_card(
        best_model,
        model_path="src/models/best_model_detection_method.joblib",
        card_path="src/models/model_card_detection_method.json",
        info=info
    )

    return best_model, leaderboard