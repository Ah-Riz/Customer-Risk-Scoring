import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import SplineTransformer, RobustScaler

def main(data_path):
    snapshot_date = pd.to_datetime('2018-12-31')
    
    df = pd.read_csv(data_path, delimiter=';')
    
    feature_engineering(df, snapshot_date)

    # Save the cleaned dataset
    # df.to_csv('data/heart_cleaned.csv', index=False)

def feature_engineering(df, snapshot_date):
    print(df.isnull().sum())
    transform_data(df, snapshot_date)
    engineer_temporal_features(df, snapshot_date)
    engineer_vehicle_features(df, snapshot_date)
    engineer_claim_features(df)

    drop_columns(df)
    to_drop = flag_high_correlation(df.select_dtypes(include=['number']))
    drop_columns(df, to_drop)
    feature_scaling(df, df.select_dtypes(include=['number']).columns)
    target_engineering(df)

def target_engineering(df):
    df['risk_score'] = (
        0.6 * (df['Cost_claims_year'] / df['Premium']) + 
        0.4 * df['claims_per_year']
    )

def feature_scaling(df, numeric_features):
    # numeric_features = ['Power', 'Cylinder_capacity', 'Value_vehicle', 'Premium', 'N_claims_year', 'N_claims_history', 'R_Claims_history', 'Cost_claims_year', 'power_weight_ratio', 'value_per_kg', 'value_per_power', 'policy_age', 'driving_experience']
    scaler = RobustScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

def flag_high_correlation(numeric_data):
    
    corr_matrix = numeric_data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    print(f"Kill these redundant features: {to_drop}")
    return to_drop

def transform_data(df, snapshot_date):
    transform_date_data(df, snapshot_date)
    transform_categorical_data(df)

def engineer_temporal_features(df, snapshot_date):
    df["policy_age"] = (snapshot_date - df["Date_start_contract"]).dt.days
    df['driving_experience'] = (snapshot_date - df['Date_driving_licence']).dt.days
    df['customer_age'] = ((snapshot_date - df['Date_birth']).dt.days)/365.25

    df['vehicle_age'] = (snapshot_date.year - df['Year_matriculation'].values.reshape(-1, 1))
    spline = SplineTransformer(n_knots=5, degree=2, include_bias=False)
    df['vehicle_age'] = spline.fit_transform(df['vehicle_age'].values.reshape(-1, 1))
    
    fill_the_lapse(df)

def engineer_vehicle_features(df,snapshot_date):
    df['power_weight_ratio'] = df['Power'] / df['Weight']
    df['premum_to_value'] = df['Premium'] / df['Value_vehicle']
    df['value_per_kg'] = df['Value_vehicle'] / df['Weight']
    df['value_per_power'] = df['Value_vehicle'] / (df['Power'] + 1e-6)
    df['vehicle_age'] = df['Year_matriculation'].apply(lambda x: snapshot_date.year - x)
    df['urban_high_risk'] = ((df['Area'] == 1) & (df['Type_risk'] == 1)).astype(int)
    
    fill_the_fuel_type(df)
    fill_the_length(df)

def engineer_claim_features(df):
    df['claims_per_year'] = df['N_claims_history'] / (df['policy_age'] / 365.25 + 1e-6)
    df['recent_claims_rate'] = df['N_claims_history'] / (df['policy_age'] / 365.25 + 1e-6)
    df['avg_claim_cost'] = df['Cost_claims_year'] / (df['N_claims_year'] + 1e-6)
    df['log_claim_cost'] = np.log1p(df['Cost_claims_year'])
    df['claim_frequency_increase'] = (df['recent_claims_rate'] > df['claims_per_year']).astype(int)

def fill_the_length(df):
    features = ['power_weight_ratio', 'Power', 'Cylinder_capacity', 'Weight', 'Type_risk', 'N_doors', 'Type_fuel',]
    X = df[~df['Length'].isna()][features]
    y = df[~df['Length'].isna()]['Length']

    model = ElasticNetCV(cv=5, random_state=42, n_jobs=-1)
    model.fit(X, y)

    if model.score(X, y) < 0.5:
        raise ValueError("Imputation model performance unacceptable")
    else:
        missing_mask = df['Length'].isna()
        df.loc[missing_mask, 'Length'] = model.predict(df[missing_mask][features])

    df['Length'] = df['Length'].round(3)

def fill_the_fuel_type(df):
    df.loc[(df['Type_fuel'].isna()) & (df['Type_risk'] == 1), 'Type_fuel'] = 0
    df.loc[(df['Type_fuel'].isna()) & (df['Type_risk'] == 4), 'Type_fuel'] = 1
    if df['Type_fuel'].isna().any():
        df['Type_fuel'].fillna(0, inplace=True)

def fill_the_lapse(df):
    df['in_lapse'] = df['Date_lapse'].notna().astype(int)

def transform_categorical_data(df):
    fuel_type_map = {'P': 0, 'D': 1}
    df['Type_fuel'] = df['Type_fuel'].map(fuel_type_map)

def transform_date_data(df, snapshot_date):
    date_cols = ["Date_start_contract", "Date_last_renewal", "Date_birth", "Date_driving_licence"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y')
    
def drop_columns(df, columns_to_drop = ['ID', "Date_start_contract", "Date_last_renewal", "Date_birth", "Date_driving_licence", 'Date_next_renewal']):
    df.drop(columns=columns_to_drop, inplace=True)

if __name__ == "__main__":
    data_path = 'data/Dataset/Motor vehicle insurance data.csv'
    main(data_path)