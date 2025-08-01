---
title: "Oregon County Data Wrangling Documentation"
format: html
---

# Oregon County Development Data Wrangling

## Overview

This script processes raw government data from three sources to create a standardized 14-indicator dataset for Oregon county development analysis.

## Data Sources

- **Health Data**: County Health Rankings (265 variables)
- **Education Data**: Oregon Department of Education (13 variables) 
- **ACS Data**: American Community Survey 5-year estimates (21 variables)

## Feature Engineering

### Calculated Features

Several indicators required transformation from raw data to meaningful development measures:

#### Economic Security
```python
snap_pct = acs_row.get('pct_with_snap_assistance')
row['economic_security'] = 100 - snap_pct if snap_pct is not None else None
```
**Rationale**: Economic security represents a community's ability to meet basic needs without assistance. We invert SNAP assistance percentage because fewer people needing food assistance indicates greater community-wide economic stability.

#### Income Equality  
```python
income_ratio = health_row.get('Income Ratio')
row['income_equality'] = 1 / (income_ratio + 0.1) if income_ratio is not None else None
```
**Rationale**: Income ratio measures inequality (80th percentile ÷ 20th percentile income). We invert using 1/x transformation to convert inequality to equality measure. The +0.1 prevents division by zero while preserving the relative ordering.

#### Provider Access
```python
pc_ratio = health_row.get('Primary Care Physicians Ratio')
if pc_ratio:
    try:
        ratio_num = float(str(pc_ratio).split(':')[0])
        row['provider_access'] = 10000 / ratio_num
    except:
        row['provider_access'] = None
```
**Rationale**: Raw data provides ratios like "1,234:1" (1 physician per 1,234 people). We parse the ratio and convert to physicians per 10,000 population for intuitive interpretation where higher values indicate better access.

#### Community Safety
```python
crime_rate = health_row.get('Violent Crime Rate')
if crime_rate is not None and not pd.isna(crime_rate) and crime_rate >= 0:
    row['community_safety'] = 1000 / (crime_rate + 1)
```
**Rationale**: Violent crime rate requires inverse transformation since lower crime equals safer communities. We use 1000/(rate+1) to create a safety score where higher values indicate safer communities. The +1 prevents division by zero.

#### Employment Rate
```python
unemployment = health_row.get('% Unemployed')
row['employment_rate'] = 100 - unemployment if unemployment is not None else None
```
**Rationale**: Simple inversion of unemployment to employment rate maintains the "higher = better" paradigm across all indicators.

#### Housing Affordability
```python
severe_housing = health_row.get('% Severe Housing Problems')
row['housing_affordability'] = 100 - severe_housing if severe_housing is not None else None
```
**Rationale**: Severe housing problems include cost burden >50% of income, overcrowding, and inadequate facilities. Inversion creates an affordability measure where higher values indicate more affordable housing markets.

#### Child Welfare
```python
child_poverty = health_row.get('% Children in Poverty')
row['child_welfare'] = 100 - child_poverty if child_poverty is not None else None
```
**Rationale**: Child welfare inverts child poverty percentage to create a positive indicator of community investment in future generations.

## Data Quality Handling

### County Name Matching
```python
health_data = health_data.dropna(subset=['County_1'])
health_data = health_data.drop_duplicates(subset=['County_1'], keep='first')
```
**Issue**: Health data contains FIPS codes in 'County' column and actual names in 'County_1', with duplicate entries.  
**Solution**: Use 'County_1' for matching and keep first occurrence of each county to ensure clean joins.

### Missing Value Imputation
```python
median_val = df[col].median()
df[col] = df[col].fillna(median_val)
```
**Rationale**: Median imputation is robust against outliers and appropriate for small datasets. Neural networks require complete data, and with <1% missing values, median substitution provides conservative estimates.

### Standardization
```python
means = np.mean(feature_data, axis=0)
stds = np.std(feature_data, axis=0)
standardized_data = (feature_data - means) / stds
```
**Purpose**: Ensures all indicators contribute equally to neural network training regardless of their original scale (income in thousands vs. percentages).

## Output Files

- **oregon_counties_raw.csv**: Original scale data for human interpretation
- **oregon_counties_standardized.csv**: Standardized data ready for neural network input
- **oregon_feature_list.csv**: Feature metadata and category classifications

## Validation

Final dataset contains 36 Oregon counties × 14 indicators with complete data following consistent "higher = better" scaling for all variables.