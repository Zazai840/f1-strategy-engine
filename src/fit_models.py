

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor


def fit_degradation(laps_all: pd.DataFrame) -> Dict[str, Dict]:
    
    green_laps = laps_all[laps_all['TrackStatus'] == '1'].copy()
    
    if len(green_laps) == 0:
        warnings.warn("No green flag laps found, using all laps")
        green_laps = laps_all.copy()
    
   
    compounds = green_laps['Compound'].unique()
    models = {}
    
    for compound in compounds:
        compound_data = green_laps[green_laps['Compound'] == compound].copy()
        
        if len(compound_data) < 10:  
            warnings.warn(f"Insufficient data for {compound} ({len(compound_data)} laps), using defaults")
            models[compound] = {
                'base': 90.0, 
                'slope': 0.1,  
                'sigma': 1.0   
            }
            continue
        
       
        X = compound_data['StintLap'].values.reshape(-1, 1)
        y = compound_data['LapTimeSeconds'].values
        
        
        y_mean, y_std = np.mean(y), np.std(y)
        valid_mask = np.abs(y - y_mean) <= 3 * y_std
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 5:
            warnings.warn(f"Too few clean data points for {compound}, using defaults")
            models[compound] = {
                'base': np.mean(y_clean) if len(y_clean) > 0 else 90.0,
                'slope': 0.1,
                'sigma': np.std(y_clean) if len(y_clean) > 1 else 1.0
            }
            continue
        
        try:
           
            regressor = HuberRegressor(epsilon=1.35, max_iter=1000)
            regressor.fit(X_clean, y_clean)
            
            y_pred = regressor.predict(X_clean)
            residuals = y_clean - y_pred
            sigma = np.std(residuals)
            
            models[compound] = {
                'base': float(regressor.intercept_),
                'slope': float(regressor.coef_[0]),
                'sigma': float(sigma)
            }
            
        except Exception as e:
            warnings.warn(f"Failed to fit model for {compound}: {e}, using defaults")
            models[compound] = {
                'base': np.mean(y_clean) if len(y_clean) > 0 else 90.0,
                'slope': 0.1,
                'sigma': np.std(y_clean) if len(y_clean) > 1 else 1.0
            }

    default_compounds = {
        'SOFT': {'base': 88.0, 'slope': 0.15, 'sigma': 1.2},
        'MEDIUM': {'base': 90.0, 'slope': 0.10, 'sigma': 1.0},
        'HARD': {'base': 92.0, 'slope': 0.05, 'sigma': 0.8},
        'UNK': {'base': 90.0, 'slope': 0.10, 'sigma': 1.0}
    }
    
    for compound, default_params in default_compounds.items():
        if compound not in models:
            models[compound] = default_params.copy()
    
    return models


def fit_pit_loss(pits_all: pd.DataFrame) -> Dict[str, np.ndarray]:
    
    if len(pits_all) == 0:
        warnings.warn("No pit stop data available, using synthetic distributions")
        return {
            'green': np.random.normal(24.0, 1.8, 1000),
            'neutral': np.random.normal(24.0, 1.8, 1000)
        }
    
    pits_clean = pits_all.copy()
    pits_clean = pits_clean.dropna(subset=['PitLaneTime'])
    pits_clean = pits_clean[pits_clean['PitLaneTime'] > 0]
    pits_clean = pits_clean[np.isfinite(pits_clean['PitLaneTime'])]
    
    if len(pits_clean) == 0:
        warnings.warn("No valid pit stop data after cleaning, using synthetic distributions")
        return {
            'green': np.random.normal(24.0, 1.8, 1000),
            'neutral': np.random.normal(24.0, 1.8, 1000)
        }
    
    green_pits = pits_clean[pits_clean['IsGreen']]['PitLaneTime'].values
    neutral_pits = pits_clean[~pits_clean['IsGreen']]['PitLaneTime'].values
    
    if len(green_pits) == 0 and len(neutral_pits) == 0:
        warnings.warn("No pit stops in either condition, using synthetic distributions")
        return {
            'green': np.random.normal(24.0, 1.8, 1000),
            'neutral': np.random.normal(24.0, 1.8, 1000)
        }
    
    elif len(green_pits) == 0:
        warnings.warn("No green flag pit stops, using neutral flag data for both")
        return {
            'green': neutral_pits,
            'neutral': neutral_pits
        }
    
    elif len(neutral_pits) == 0:
        warnings.warn("No neutralized pit stops, using green flag data for both")
        return {
            'green': green_pits,
            'neutral': green_pits
        }
    
    else:
        return {
            'green': green_pits,
            'neutral': neutral_pits
        }


def validate_models(models: Dict[str, Dict], pit_dists: Dict[str, np.ndarray]) -> bool:
    for compound, params in models.items():
        if not all(key in params for key in ['base', 'slope', 'sigma']):
            warnings.warn(f"Invalid model structure for {compound}")
            return False
        
        if params['base'] <= 0 or params['sigma'] <= 0:
            warnings.warn(f"Invalid model parameters for {compound}: {params}")
            return False
    for condition, times in pit_dists.items():
        if len(times) == 0:
            warnings.warn(f"Empty pit distribution for {condition}")
            return False
        
        if np.any(times <= 0) or not np.all(np.isfinite(times)):
            warnings.warn(f"Invalid pit times for {condition}")
            return False
    
    return True


if __name__ == "__main__":
  
    print("Testing model fitting...")
    
    
    np.random.seed(42)
    

    n_laps = 100
    stint_laps = np.tile(np.arange(1, 11), 10)
    base_time = 90.0
    degradation = 0.1
    noise = np.random.normal(0, 1.0, n_laps)
    lap_times = base_time + degradation * stint_laps + noise
    
    test_laps = pd.DataFrame({
        'Compound': ['SOFT'] * n_laps,
        'StintLap': stint_laps,
        'LapTimeSeconds': lap_times,
        'TrackStatus': ['1'] * n_laps
    })
    
    models = fit_degradation(test_laps)
    print(f"✓ Fitted degradation model: {models['SOFT']}")
    
    
    green_times = np.random.normal(24.0, 1.5, 50)
    neutral_times = np.random.normal(25.0, 2.0, 30)
    
    test_pits = pd.DataFrame({
        'PitLaneTime': np.concatenate([green_times, neutral_times]),
        'IsGreen': [True] * 50 + [False] * 30
    })
    
    pit_dists = fit_pit_loss(test_pits)
    print(f"✓ Fitted pit distributions: green={len(pit_dists['green'])}, neutral={len(pit_dists['neutral'])}")
    
    
    is_valid = validate_models(models, pit_dists)
    print(f"✓ Model validation: {is_valid}")
    
    print("Model fitting tests completed.")
