def create_targets(df):

    # Regression target
    df['Future_Price'] = df['Price_in_Lakhs'] * ((1.08) ** 5)

    # ✅ Better classification logic
    score = (
        (df['BHK'] >= 3).astype(int) +
        (df['Parking_Space'] >= 1).astype(int) +
        (df['Nearby_Schools'] >= 3).astype(int) +
        (df['Public_Transport_Accessibility'] >= 5).astype(int)
    )

    df['Good_Investment'] = (score >= 2).astype(int)

    return df