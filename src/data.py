
import warnings
from typing import Optional, Tuple

import fastf1
import numpy as np
import pandas as pd
from fastf1.core import DataNotLoadedError

MIN_SUPPORTED_YEAR = 2018  

def load_race_session(year: int, gp_name_or_location: str):
    ses = fastf1.get_session(year, gp_name_or_location, "R")
    try:
        ses.load(laps=True, telemetry=False, weather=False, messages=True)
        _ = ses.laps  
        return ses
    except (DataNotLoadedError, SessionNotAvailableError) as e:
        raise RuntimeError(
            f"No lap data available for {year} {gp_name_or_location}. "
            f"FastF1 generally has reliable Live Timing from {MIN_SUPPORTED_YEAR}+ seasons. "
            f"Original: {e}"
        )


def ensure_core_columns(laps: pd.DataFrame, *, year: Optional[int] = None, 
                       event: Optional[str] = None) -> pd.DataFrame:
   
    df = laps.copy()
    
   
    if 'Year' not in df.columns:
        if year is not None:
            df['Year'] = year
        else:
            raise ValueError("Year must be provided if not in data")
    
    if 'EventName' not in df.columns:
        if event is not None:
            df['EventName'] = event
        else:
            df['EventName'] = 'Unknown'
    
    if 'Driver' not in df.columns:
        if 'DriverNumber' in df.columns:
            df['Driver'] = df['DriverNumber'].astype(str)
        elif 'Abbreviation' in df.columns:
            df['Driver'] = df['Abbreviation']
        else:
            raise ValueError("Cannot derive Driver column")
    
    if 'LapNumber' not in df.columns:
        if 'LapTime' in df.columns:
            df['LapNumber'] = df.groupby('Driver').cumcount() + 1
        else:
            raise ValueError("Cannot derive LapNumber column")
    
    if 'LapTimeSeconds' not in df.columns:
        if 'LapTime' in df.columns:
            df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
        else:
            raise ValueError("Cannot derive LapTimeSeconds column")
    
    df['LapTimeSeconds'] = pd.to_numeric(df['LapTimeSeconds'], errors='coerce')
    df = df.dropna(subset=['LapTimeSeconds'])
    df = df[df['LapTimeSeconds'] > 0]  # Remove invalid lap times
    
    if 'Stint' not in df.columns:
        if 'TyreLife' in df.columns:
           
            df['Stint'] = df.groupby('Driver')['TyreLife'].apply(
                lambda x: (x != x.shift()).cumsum()
            ).reset_index(0, drop=True)
        else:
   
            df['Stint'] = 1
    
    if 'StintLap' not in df.columns:
        df['StintLap'] = df.groupby(['Driver', 'Stint']).cumcount() + 1
    
    if 'Compound' not in df.columns:
        if 'TyreCompound' in df.columns:
            df['Compound'] = df['TyreCompound']
        else:
            df['Compound'] = 'UNK'
    
    df['Compound'] = df['Compound'].str.upper().str.strip()
    df['Compound'] = df['Compound'].fillna('UNK')
    
    if 'TrackStatus' not in df.columns:
        if 'TrackStatus' in df.columns:
            df['TrackStatus'] = df['TrackStatus'].fillna('1')  # Assume green flag
        else:
            df['TrackStatus'] = '1'  # Default to green flag
    
    required_cols = ['Year', 'EventName', 'Driver', 'LapNumber', 'LapTimeSeconds', 
                    'Stint', 'StintLap', 'Compound', 'TrackStatus']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df[required_cols]


def load_many_years(years: list[int], gp: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    all_laps = []
    all_pits = []
    
    for year in years:
        try:
            session = load_race_session(year, gp)
            
            
            laps_df = ensure_core_columns(
                session.laps, 
                year=year, 
                event=session.event['EventName']
            )
            all_laps.append(laps_df)
            
           
            if hasattr(session, 'pit_stops') and session.pit_stops is not None:
                pits_df = session.pit_stops.copy()
                
           
                pits_df['Year'] = year
                pits_df['EventName'] = session.event['EventName']
                
          
                if 'PitLaneTime' not in pits_df.columns:
                    if 'PitOutTime' in pits_df.columns and 'PitInTime' in pits_df.columns:
                        pits_df['PitLaneTime'] = (
                            pits_df['PitOutTime'] - pits_df['PitInTime']
                        ).dt.total_seconds()
                    else:
                       
                        pits_df['PitLaneTime'] = 25.0  
                
                if 'TrackStatus' in pits_df.columns:
                    pits_df['IsGreen'] = pits_df['TrackStatus'] == '1'
                else:
                    pits_df['IsGreen'] = True  
                
     
                pits_df = pits_df.dropna(subset=['PitLaneTime'])
                pits_df = pits_df[pits_df['PitLaneTime'] > 0]
                
                all_pits.append(pits_df[['Year', 'EventName', 'Driver', 'PitLaneTime', 'IsGreen']])
            
        except Exception as e:
            warnings.warn(f"Failed to load {year} {gp}: {str(e)}")
            continue
    
    if not all_laps:
        raise Exception(f"No data loaded for {gp} across years {years}")
    

    laps_all = pd.concat(all_laps, ignore_index=True)
    
    if all_pits:
        pits_all = pd.concat(all_pits, ignore_index=True)
    else:
        pits_all = pd.DataFrame(columns=['Year', 'EventName', 'Driver', 'PitLaneTime', 'IsGreen'])
    
    return laps_all, pits_all


if __name__ == "__main__":
    
    print("Testing data loading...")
    
   
    try:
        session = load_race_session(2023, "Monza")
        print(f"✓ Loaded session: {session.event['EventName']} {session.event['EventDate']}")
        
        laps_clean = ensure_core_columns(session.laps, year=2023, event="Monza")
        print(f"✓ Cleaned laps: {len(laps_clean)} laps, columns: {list(laps_clean.columns)}")
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
    
    print("Data module tests completed.")


