import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(file_path):
    """Load and clean the dataset."""
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df


def engineer_features(df):
    """Create essential features for the model."""
    data = df.copy()
    data["Transmission_Type"] = data["Transmission"].str.extract(r"([A-Z]+)")
    data["Gear_Count"] = data["Transmission"].str.extract(r"(\d+)").astype(float)
    data["Gear_Count"] = data["Gear_Count"].fillna(
        data["Gear_Count"].median(skipna=True)
    )
    data["Power_to_Weight"] = (
        data["Engine Size (L)"] * 1000 / data["Vehicle Class"].map(len)
    )
    data["Fuel_Efficiency"] = 200 / data["Comb (L/100 km)"]  # Simplified proxy
    data["City_Hwy_Ratio"] = data["Fuel Consumption (L/100Km)"] / data["Hwy (L/100 km)"]
    data["Engine_Size_Per_Cylinder"] = data["Engine Size (L)"] / data[
        "Cylinders"
    ].replace(0, np.nan)
    data["Engine_Size_Per_Cylinder"] = data["Engine_Size_Per_Cylinder"].fillna(
        data["Engine Size (L)"]
    )

    encoders = {}
    categorical_cols = [
        "Model",
        "Vehicle Class",
        "Transmission_Type",
        "Fuel Type",
        "Make",
    ]
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[f"{col}_Encoded"] = le.fit_transform(data[col])
            encoders[col] = le

    def get_size_category(vehicle_class):
        if any(term in vehicle_class.lower() for term in ["small", "compact", "mini"]):
            return 0
        elif any(term in vehicle_class.lower() for term in ["mid", "medium"]):
            return 1
        return 2

    data["Vehicle_Size"] = data["Vehicle Class"].apply(get_size_category)
    data["Is_4WD"] = data["Model"].apply(
        lambda x: 1 if any(term in x for term in ["4WD", "4X4", "AWD"]) else 0
    )
    return data, encoders


def create_emission_categories(df, num_categories=3):
    """Create emission categories based on CO2 percentiles."""
    thresholds = [
        df["CO2 Emissions (g/km)"].quantile(i / num_categories)
        for i in range(1, num_categories)
    ]
    df["Emission_Label"] = df["CO2 Emissions (g/km)"].apply(
        lambda x: next((i for i, t in enumerate(thresholds) if x <= t), len(thresholds))
    )
    categories = ["Low Emission", "Moderate Emission", "High Emission"]
    emission_mapping = {i: categories[i] for i in range(len(categories))}
    df["Emission_Category"] = df["Emission_Label"].map(emission_mapping)
    return df, emission_mapping


def select_features(df, target="Emission_Label"):
    """Select features based on correlation with the target."""
    potential_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [
        target,
        "CO2 Emissions (g/km)",
        "Comb (mpg)",
        "Year",
    ]
    potential_features = [col for col in potential_features if col not in exclude_cols]

    correlations = []
    for col in potential_features:
        if col in df.columns:
            corr = abs(np.corrcoef(df[col], df[target])[0, 1])
            correlations.append((col, corr))

    correlations.sort(key=lambda x: x[1], reverse=True)
    selected_features = [feat for feat, _ in correlations[:10]]  # Top 10 features
    return selected_features, correlations


def train_model(X_train, X_test, y_train, y_test):
    """Train a RandomForestClassifier."""
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(
        model, np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)), cv=5
    )
    return model, accuracy, cv_scores.mean()


def process_data(file_path):
    """Main function to process data and train the model."""
    # Load and prepare data
    df = load_and_prepare_data(file_path)

    # Feature engineering
    processed_df, encoders = engineer_features(df)

    # Create emission categories
    processed_df, emission_mapping = create_emission_categories(processed_df)
    category_distribution = dict(
        processed_df["Emission_Category"].value_counts().items()
    )

    # Feature selection
    selected_features, feature_correlations = select_features(processed_df)

    # Prepare data
    X = processed_df[selected_features].values
    y = processed_df["Emission_Label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train model
    model, accuracy, cv_accuracy = train_model(X_train, X_test, y_train, y_test)

    # Feature importance
    feature_importance = list(zip(selected_features, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return {
        "df": df,
        "selected_features": selected_features,
        "feature_correlations": feature_correlations,
        "best_model": model,
        "best_model_name": "Random Forest",
        "best_accuracy": accuracy,
        "cv_accuracy": cv_accuracy,
        "feature_importance": feature_importance,
        "category_distribution": category_distribution,
        "emission_mapping": emission_mapping,
        "scaler": scaler,
        "encoders": encoders,
        "all_models_performance": {"Random Forest": (cv_accuracy, accuracy)},
    }
