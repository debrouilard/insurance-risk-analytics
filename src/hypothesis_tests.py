import pandas as pd
from scipy import stats

def calculate_claim_frequency(df):
    """Calculate claim frequency"""
    df = df.copy()
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    return df

def test_provinces(df, alpha=0.05):
    """Test H0: No risk differences across provinces"""
    groups = [df[df['Province'] == p]['LossRatio'].dropna() 
              for p in df['Province'].unique() 
              if len(df[df['Province'] == p]) > 30]
    
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        return {
            "Hypothesis": "No risk differences across provinces",
            "Test": "ANOVA / F-test",
            "p_value": round(p_value, 6),
            "Decision": "Reject" if p_value < alpha else "Fail to Reject"
        }
    return None

def test_gender_risk(df, alpha=0.05):
    """Test H0: No significant risk difference between Women and Men"""
    male = df[df['Gender'] == 'Male']['LossRatio'].dropna()
    female = df[df['Gender'] == 'Female']['LossRatio'].dropna()
    
    if len(male) > 30 and len(female) > 30:
        t_stat, p_value = stats.ttest_ind(male, female)
        return {
            "Hypothesis": "No risk difference between Women and Men",
            "Test": "Independent t-test",
            "p_value": round(p_value, 6),
            "Decision": "Reject" if p_value < alpha else "Fail to Reject"
        }
    return None

def test_zipcode_margin(df, alpha=0.05):
    """Test H0: No significant margin difference between zip codes"""
    postal_counts = df['PostalCode'].value_counts()
    if len(postal_counts) >= 2:
        zip1 = df[df['PostalCode'] == postal_counts.index[0]]['Margin'].dropna()
        zip2 = df[df['PostalCode'] == postal_counts.index[1]]['Margin'].dropna()
        
        if len(zip1) > 30 and len(zip2) > 30:
            t_stat, p_value = stats.ttest_ind(zip1, zip2)
            return {
                "Hypothesis": "No margin difference between zip codes",
                "Test": "t-test",
                "p_value": round(p_value, 6),
                "Decision": "Reject" if p_value < alpha else "Fail to Reject"
            }
    return None