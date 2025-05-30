import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def main(data_path):
    df = pd.read_csv(data_path, delimiter=';')

    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    
    transform_data(df)
    fill_empty_values(df)
    feature_engineering(df)

    # Save the cleaned dataset
    # df.to_csv('data/heart_cleaned.csv', index=False)

def feature_engineering(df):
    ever_claims(df)

def ever_claims(df):


def fill_empty_values(df):
    fill_the_fuel_type(df)
    fill_the_length(df)
    fill_the_lapse(df)

def fill_the_length(df):
    features = ['Power', 'Cylinder_capacity', 'Weight', 'Type_risk', 'N_doors', 'Type_fuel',]
    X = df[~df['Length'].isna()][features]
    y = df[~df['Length'].isna()]['Length']

    model = LinearRegression()
    model.fit(X, y)

    missing_length_mask = df['Length'].isna()
    X_predict = df[missing_length_mask][features]
    predicted_lengths = model.predict(X_predict)

    df.loc[missing_length_mask, 'Length'] = predicted_lengths

    df['Length'] = df['Length'].round(3)

def fill_the_fuel_type(df):
    feature_to_consider = ['Type_risk', 'Value_vehicle', 'Power', 'Cylinder_capacity', 'N_doors', 'Weight']
    X = df[~df['Type_fuel'].isna()][feature_to_consider]
    y = df[~df['Type_fuel'].isna()]['Type_fuel']

    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)

    print("Best parameters found: ", grid_search.best_params_)
    print("Mean CV score with best params: ", grid_search.best_score_)

    missing_fuel_mask = df['Type_fuel'].isna()
    X_predict = df[missing_fuel_mask][feature_to_consider]
    predicted_fuel = grid_search.best_estimator_.predict(X_predict)

    df.loc[missing_fuel_mask, 'Type_fuel'] = predicted_fuel

def fill_the_lapse(df):
    df['in_lapse'] = df['Date_lapse'].notna().astype(int)

def transform_data(df):
    transform_date_data(df)
    transform_categorical_data(df)

def transform_categorical_data(df):
    fuel_type_map = {'P': 0, 'D': 1}
    df['Type_fuel'] = df['Type_fuel'].map(fuel_type_map)

def transform_date_data(df):
    df["Date_start_contract"] = pd.to_datetime(df["Date_start_contract"], format='%d/%m/%Y')
    df["Date_last_renewal"] = pd.to_datetime(df["Date_last_renewal"], format='%d/%m/%Y')
    df["Date_next_renewal"] = pd.to_datetime(df["Date_next_renewal"], format='%d/%m/%Y')
    df["Date_birth"] = pd.to_datetime(df["Date_birth"], format='%d/%m/%Y')
    df['Date_driving_licence'] = pd.to_datetime(df["Date_driving_licence"], format='%d/%m/%Y')

    df["contract_duration"] = ((df["Date_last_renewal"] - df["Date_start_contract"]).dt.days)/365.25
    df['age'] = ((df['Date_next_renewal'] - df['Date_birth']).dt.days)/365.25
    df['licence_duration'] = ((df['Date_start_contract'] - df['Date_driving_licence']).dt.days)/365.25


if __name__ == "__main__":
    data_path = 'data/Dataset/Motor vehicle insurance data.csv'
    main(data_path)