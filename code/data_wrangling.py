import pandas as pd
import numpy as np

print("OREGON COUNTY DATA WRANGLING")
print("="*50)

def wrangle_oregon_data():
    """Create the 14-indicator development dataset"""
    
    # Load data sources
    health_data = pd.read_csv(r"C:\Users\joshu\Downloads\HDI Modeling Data\Health_Data.csv")
    education_data = pd.read_csv(r"C:\Users\joshu\Downloads\HDI Modeling Data\education.csv") 
    acs_data = pd.read_csv(r"C:\Users\joshu\Downloads\HDI Modeling Data\oregon_acs_5Y_2023_consolidated.csv")
    
    # Filter to Oregon counties only - also remove rows with NaN county names
    health_data = health_data[health_data['County'] != 'Oregon'].copy()
    health_data = health_data.dropna(subset=['County_1'])  # Remove rows with missing county names
    
    # Remove duplicate counties - keep first occurrence with valid data
    health_data = health_data.drop_duplicates(subset=['County_1'], keep='first')
    
    education_data = education_data[education_data['County'] != 'Oregon'].copy()
    
    counties = education_data['County'].tolist()
    # Using education data for county list since it has exactly 36 Oregon counties we need
    
    # Create lookup dictionaries - health data uses 'County_1' for actual county names
    health_lookup = {row['County_1']: row for _, row in health_data.iterrows()}
    edu_lookup = {row['County']: row for _, row in education_data.iterrows()}
    acs_lookup = {row['county']: row for _, row in acs_data.iterrows()}
    # Health data has FIPS codes in 'County' column, actual names in 'County_1'
    
    data_rows = []
    
    for county in counties:
        row = {'county': county}
        health_row = health_lookup.get(county, {})
        edu_row = edu_lookup.get(county, {})
        acs_row = acs_lookup.get(county, {})
        
        # ECONOMIC INDICATORS (4)
        
        row['household_income'] = acs_row.get('household_median_income')
        # Just grabbing median household income straight from the ACS data - this is the most solid measure of what families actually have to spend
        
        unemployment = health_row.get('% Unemployed')
        row['employment_rate'] = 100 - unemployment if unemployment is not None else None
        # Flipping unemployment to employment because we want everything to follow "higher = better" - makes the analysis way cleaner
        
        snap_pct = acs_row.get('pct_with_snap_assistance')
        row['economic_security'] = 100 - snap_pct if snap_pct is not None else None
        # Economic security = 100 minus food assistance percentage because fewer people needing help = more secure community economically
        
        income_ratio = health_row.get('Income Ratio')
        row['income_equality'] = 1 / (income_ratio + 0.1) if income_ratio is not None else None
        # Income ratio is rich vs poor (80th percentile / 20th percentile) so we flip it with 1/x to get equality instead of inequality
        
        # HEALTH INDICATORS (3)
        
        uninsured = health_row.get('% Uninsured')
        row['healthcare_access'] = 100 - uninsured if uninsured is not None else None
        # Healthcare access = 100 minus uninsured because you can't get care if you can't pay for it, period
        
        fair_poor = health_row.get('% Fair or Poor Health')
        row['health_outcomes'] = 100 - fair_poor if fair_poor is not None else None
        # Health outcomes = 100 minus bad health to keep our "higher = better" thing going
        
        pc_ratio = health_row.get('Primary Care Physicians Ratio')
        if pc_ratio:
            try:
                ratio_num = float(str(pc_ratio).split(':')[0])
                row['provider_access'] = 10000 / ratio_num
            except:
                row['provider_access'] = None
        else:
            row['provider_access'] = None
        # Provider ratios come as "1,234:1" which means 1 doctor per 1,234 people - we convert to doctors per 10k so higher = better access
        
        # EDUCATION INDICATORS (3)
        
        row['graduation_rate'] = edu_row.get('23-24 Graduation Rate')
        # Using most recent graduation rate because that's what matters right now for these communities
        
        row['higher_education'] = health_row.get('% Some College')
        # Any college shows the community actually values education beyond just high school
        
        dropout = edu_row.get('23-24 Dropout Rate')
        row['educational_retention'] = 100 - dropout if dropout is not None else None
        # Retention = 100 minus dropout because keeping kids in school is what we actually want to measure here
        
        # COMMUNITY INDICATORS (4)
        
        severe_housing = health_row.get('% Severe Housing Problems')
        row['housing_affordability'] = 100 - severe_housing if severe_housing is not None else None
        # Housing affordability = 100 minus housing problems because affordable housing is what builds stable communities
        
        crime_rate = health_row.get('Violent Crime Rate')
        if crime_rate is not None and not pd.isna(crime_rate) and crime_rate >= 0:
            row['community_safety'] = 1000 / (crime_rate + 1)
        else:
            row['community_safety'] = None
        # Safety score using 1000/(crime rate + 1) so lower crime = higher safety score - the math just works better this way
        
        row['social_cohesion'] = health_row.get('Social Association Rate')
        # Social cohesion = civic orgs per 10k people - more groups = stronger community bonds, pretty straightforward
        
        child_poverty = health_row.get('% Children in Poverty')
        row['child_welfare'] = 100 - child_poverty if child_poverty is not None else None
        # Child welfare = 100 minus child poverty because communities that take care of kids today will be stronger tomorrow
        
        data_rows.append(row)
    
    return pd.DataFrame(data_rows)

def clean_and_standardize_data(df):
    """Handle missing values and standardize using pandas/numpy"""
    
    feature_cols = [col for col in df.columns if col != 'county']
    print(f"Processing {len(feature_cols)} indicators...")
    
    # Fill missing values with median
    for col in feature_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {missing_count} missing values in {col} with median: {median_val:.2f}")
    # Using median because it's more robust than mean and neural networks need complete data to work properly
    
    # Standardize using numpy
    feature_data = df[feature_cols].values
    means = np.mean(feature_data, axis=0)
    stds = np.std(feature_data, axis=0)
    stds = np.where(stds == 0, 1, stds)
    standardized_data = (feature_data - means) / stds
    # Standardizing so all indicators are on same scale - otherwise income (big numbers) would dominate over percentages
    
    # Create standardized dataframe
    df_standardized = df[['county']].copy()
    for i, col in enumerate(feature_cols):
        df_standardized[col] = standardized_data[:, i]
    
    return df_standardized, feature_cols

def save_dataset(df_raw, df_standardized, feature_cols):
    """Save both raw and standardized datasets"""
    
    # Save to the same directory as our source data
    data_dir = r"C:\Users\joshu\Downloads\HDI Modeling Data"
    
    # Save raw data
    raw_path = f"{data_dir}\\oregon_counties_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"✓ Saved raw data to: {raw_path}")
    
    # Save standardized data (ready for neural network)
    standardized_path = f"{data_dir}\\oregon_counties_standardized.csv"
    df_standardized.to_csv(standardized_path, index=False)
    print(f"✓ Saved standardized data to: {standardized_path}")
    
    # Save feature list for reference
    feature_info = pd.DataFrame({
        'feature_name': feature_cols,
        'category': ['Economic']*4 + ['Health']*3 + ['Education']*3 + ['Community']*4
    })
    feature_path = f"{data_dir}\\oregon_feature_list.csv"
    feature_info.to_csv(feature_path, index=False)
    print(f"✓ Saved feature reference to: {feature_path}")

# Execute the data wrangling
if __name__ == "__main__":
    print("Starting Oregon county data preparation...")
    
    # Create the dataset
    df_raw = wrangle_oregon_data()
    print(f"Raw dataset created: {df_raw.shape}")
    
    # Clean and standardize
    df_standardized, features = clean_and_standardize_data(df_raw)
    print(f"Standardized dataset ready: {df_standardized.shape}")
    
    # Save everything
    save_dataset(df_raw, df_standardized, features)
    
    print(f"\nComplete! Dataset has {len(features)} indicators:")
    print("Economic (4): Income, Employment, Security, Equality")
    print("Health (3): Access, Outcomes, Provider Access") 
    print("Education (3): Attainment, Higher Ed, Retention")
    print("Community (4): Housing, Safety, Social Cohesion, Child Welfare")
    print("\nAll indicators follow 'higher = better' logic")
    print("Ready for PyTorch neural network analysis!")