import pandas as pd
import pickle
import os
from main import process_data

# Step 1: Load the dataset
file_path = "data/Fuel_Consumption_Ratings_2023.csv"
try:
    df = pd.read_csv(file_path)
    os.system("cls")
    print(f"Dataset loaded: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
except Exception as e:
    raise Exception(f"Failed to load dataset: {e}")

# Step 2: Process data and train model
try:
    results = process_data(file_path)
    print("Model training completed")
except Exception as e:
    raise Exception(f"Error during processing/training: {e}")

# Step 3: Save the model and components
output_file = "model/vehicle_emission_model.pkl"
try:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "best_model": results["best_model"],
                "best_model_name": results["best_model_name"],
                "selected_features": results["selected_features"],
                "feature_importance": results["feature_importance"],
                "emission_mapping": results["emission_mapping"],
                "scaler": results["scaler"],
                "encoders": results["encoders"],
                "feature_correlations": results["feature_correlations"],
                "category_distribution": results["category_distribution"],
                "best_accuracy": results["best_accuracy"],
                "cv_accuracy": results["cv_accuracy"],
                "makes": results["df"]["Make"].unique().tolist(),
                "vehicle_classes": results["df"]["Vehicle Class"].unique().tolist(),
                "fuel_types": results["df"]["Fuel Type"].unique().tolist(),
                "df": results["df"],
            },
            f,
        )
    print(f"Model saved to: {output_file}")
except Exception as e:
    raise Exception(f"Error saving model: {e}")

# Step 4: Print summary
print("\nTraining Summary:")
print("\nSelected Features:")
for feature in results["selected_features"]:
    print(f"  - {feature}")
print(
    f"\nBest Model: {results['best_model_name']}\n"
    f"Test Accuracy: {results['best_accuracy']*100:.2f}%\n"
    f"Cross-Validation Accuracy: {results['cv_accuracy']*100:.2f}%"
)
print("\nEmission Category Distribution:")
for category, count in results["category_distribution"].items():
    print(f"  - {category}: {count}")
