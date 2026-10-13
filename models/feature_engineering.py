import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("SeverityPrediction.FeatureEngineering")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-grounded, leak-free feature engineering.
    Can be run identically during training, evaluation, and single-instance Flask inference.

    Features generated for accident/traffic domain:
      - Speed_Delta: vehicle_speed - speed_limit
      - Speed_Ratio: vehicle_speed / (speed_limit + 1e-5)
      - Excess_Speed_Flag: 1 if speed exceeds limit, else 0
      - Adverse_Weather_Flag: 1 if weather conditions are hazardous
      - Hazardous_Road_Flag: 1 if surface is wet, icy, or flooded
      - High_Risk_Lighting_Flag: 1 if dark without lights or poor visibility
      - Vulnerable_Vehicle_Flag: 1 for motorcycles or unprotected transport
      - Driver_Impairment_Risk: 1 if alcohol or unfastened seatbelt
      - Casualties_Per_Vehicle: number_of_casualties / number_of_vehicles
      - Environmental_Hazard_Index: composite environmental friction metric

    Also provides fallback interaction features for arbitrary tabular datasets in advance mode.
    """
    df_feat = df.copy()

    # Domain specific features if columns exist
    has_speed = "Vehicle_Speed" in df_feat.columns and "Speed_Limit" in df_feat.columns
    if has_speed:
        speed = pd.to_numeric(df_feat["Vehicle_Speed"], errors="coerce").fillna(0)
        limit = pd.to_numeric(df_feat["Speed_Limit"], errors="coerce").fillna(50)
        df_feat["Speed_Delta"] = speed - limit
        df_feat["Speed_Ratio"] = (speed / (limit.replace(0, 50))).round(4)
        df_feat["Excess_Speed_Flag"] = (df_feat["Speed_Delta"] > 0).astype(int)

    if "Weather_Conditions" in df_feat.columns:
        adverse_weather = ["Rainy", "Snowy", "Foggy", "High_Winds", "Rain", "Snow", "Fog"]
        df_feat["Adverse_Weather_Flag"] = (
            df_feat["Weather_Conditions"].astype(str).str.lower().apply(
                lambda val: int(any(adv.lower() in val for adv in adverse_weather))
            )
        )

    if "Road_Surface_Conditions" in df_feat.columns:
        hazardous_road = ["Wet", "Ice", "Flooded", "Damp", "Snow"]
        df_feat["Hazardous_Road_Flag"] = (
            df_feat["Road_Surface_Conditions"].astype(str).str.lower().apply(
                lambda val: int(any(haz.lower() in val for haz in hazardous_road))
            )
        )

    if "Lighting_Conditions" in df_feat.columns:
        dark_lighting = ["Darkness_No_Lights", "Darkness", "No_Lights", "Dawn_Dusk", "Dark"]
        df_feat["High_Risk_Lighting_Flag"] = (
            df_feat["Lighting_Conditions"].astype(str).str.lower().apply(
                lambda val: int(any(dark.lower() in val for dark in dark_lighting))
            )
        )

    if "Vehicle_Type" in df_feat.columns:
        df_feat["Vulnerable_Vehicle_Flag"] = (
            df_feat["Vehicle_Type"].astype(str).str.lower().str.contains("motorcycle|bike|pedestrian").astype(int)
        )

    has_driver_risk = "Driver_Alcohol" in df_feat.columns or "Seatbelt_Restraint" in df_feat.columns
    if has_driver_risk:
        alc_risk = (
            df_feat["Driver_Alcohol"].astype(str).str.lower().isin(["yes", "true", "1"])
            if "Driver_Alcohol" in df_feat.columns
            else False
        )
        belt_risk = (
            df_feat["Seatbelt_Restraint"].astype(str).str.lower().isin(["not_worn", "no", "false", "0"])
            if "Seatbelt_Restraint" in df_feat.columns
            else False
        )
        df_feat["Driver_Impairment_Risk"] = (alc_risk | belt_risk).astype(int)

    has_counts = "Number_of_Casualties" in df_feat.columns and "Number_of_Vehicles" in df_feat.columns
    if has_counts:
        cas = pd.to_numeric(df_feat["Number_of_Casualties"], errors="coerce").fillna(1)
        veh = pd.to_numeric(df_feat["Number_of_Vehicles"], errors="coerce").fillna(1).replace(0, 1)
        df_feat["Casualties_Per_Vehicle"] = (cas / veh).round(4)

    env_cols = [c for c in ["Adverse_Weather_Flag", "Hazardous_Road_Flag", "High_Risk_Lighting_Flag"] if c in df_feat.columns]
    if env_cols:
        df_feat["Environmental_Hazard_Index"] = df_feat[env_cols].sum(axis=1)

    logger.debug("Engineered features completed. Total columns: %d", len(df_feat.columns))
    return df_feat
