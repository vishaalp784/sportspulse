import pandas as pd
import numpy as np
import os

print("⚽ SportsPulse — Loading FIFA 23 data...")

# Load only FIFA 23 rows — skip all older versions
chunks = []
chunk_size = 50000
total = 0

for chunk in pd.read_csv("data/male_players.csv", 
                          low_memory=False, 
                          chunksize=chunk_size):
    fifa23 = chunk[chunk["fifa_version"] == 23]
    if len(fifa23) > 0:
        chunks.append(fifa23)
        total += len(fifa23)

df = pd.concat(chunks, ignore_index=True)
print(f"✓ FIFA 23 players loaded: {len(df)}")

# ── KEEP USEFUL COLUMNS ───────────────────────────────────
cols = [
    "short_name", "long_name", "player_positions",
    "overall", "potential", "value_eur", "wage_eur",
    "age", "height_cm", "weight_kg",
    "league_name", "league_level",
    "club_name", "nationality_name",
    "pace", "shooting", "passing",
    "dribbling", "defending", "physic",
    "attacking_crossing", "attacking_finishing",
    "attacking_heading_accuracy", "attacking_short_passing",
    "skill_dribbling", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed",
    "power_shot_power", "power_stamina",
    "mentality_aggression", "mentality_vision",
    "defending_marking_awareness", "defending_standing_tackle",
    "goalkeeping_diving", "goalkeeping_handling",
]

# Keep only columns that exist
cols = [c for c in cols if c in df.columns]
df = df[cols].copy()

# ── CLEAN DATA ────────────────────────────────────────────
print("🧹 Cleaning data...")

# Remove players with no market value
df = df.dropna(subset=["value_eur"])
df = df[df["value_eur"] > 0]

# Fill missing numeric columns with median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Primary position
df["primary_position"] = df["player_positions"].str.split(",").str[0].str.strip()

# Position group
def pos_group(pos):
    if pos in ["GK"]:
        return "Goalkeeper"
    elif pos in ["CB","LB","RB","LWB","RWB"]:
        return "Defender"
    elif pos in ["CDM","CM","CAM","LM","RM"]:
        return "Midfielder"
    else:
        return "Forward"

df["position_group"] = df["primary_position"].apply(pos_group)

# Value in millions
df["value_eur_m"] = df["value_eur"] / 1_000_000

print(f"✓ Clean dataset: {len(df)} players")
print(f"✓ Value range: €{df['value_eur_m'].min():.2f}M — €{df['value_eur_m'].max():.2f}M")
print(f"✓ Avg value: €{df['value_eur_m'].mean():.2f}M")

# Save
os.makedirs("data", exist_ok=True)
df.to_csv("data/fifa23_clean.csv", index=False)
print(f"\n✅ Saved to data/fifa23_clean.csv")
print(df[["short_name","overall","potential","value_eur_m","position_group"]].head(10))