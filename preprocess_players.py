import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/NBA Shot Locations 1997 - 2020.csv"

PLAYERS = [
    # Original deep-dive players
    "Stephen Curry",
    "LeBron James",
    "Kevin Durant",
    # Broad study — top 30 by shot volume in window
    "James Harden",
    "Russell Westbrook",
    "Klay Thompson",
    "Damian Lillard",
    "DeMar DeRozan",
    "LaMarcus Aldridge",
    "Kemba Walker",
    "Paul George",
    "Kyrie Irving",
    "Bradley Beal",
    "Anthony Davis",
    "Carmelo Anthony",
    "Blake Griffin",
    "John Wall",
    "Dwyane Wade",
    "Kyle Lowry",
    "Chris Paul",
    "DeMarcus Cousins",
    "Marc Gasol",
    "Nikola Vucevic",
    "Kawhi Leonard",
    "Jimmy Butler",
    "CJ McCollum",
    "Serge Ibaka",
    "Tobias Harris",
    "Goran Dragic",
    "Thaddeus Young",
]

ALL_SEASONS = {2012, 2013, 2014, 2015, 2016, 2017, 2018}  # 2012-13 through 2018-19

TEST_SIZE   = 0.25
RANDOM_SEED = 42

KEEP_COLS = [
    "X Location", "Y Location", "Shot Made Flag", "is_3pt",
    "Period", "Minutes Remaining", "Seconds Remaining",
    "Action Type", "Shot Type", "Home Team", "Away Team",
    "Team Name", "Game Date", "Season Type",
    "Split",   # train / test label added below
]

# Map Shot Zone Basic -> court region
REGION_MAP = {
    "Restricted Area":          "paint",
    "In The Paint (Non-RA)":    "paint",
    "Mid-Range":                "midrange",
    "Left Corner 3":            "three_point",
    "Right Corner 3":           "three_point",
    "Above the Break 3":        "three_point",
    "Backcourt":                "backcourt",
}


def derive_season(game_date: pd.Series) -> pd.Series:
    """Return NBA season start year from YYYYMMDD integer dates.
    Seasons start in October: Jan 2016 → season 2015 (2015-16).
    """
    dt = pd.to_datetime(game_date.astype(str), format="%Y%m%d")
    return dt.apply(lambda d: d.year if d.month >= 10 else d.year - 1)


def print_stats(name: str, df: pd.DataFrame, split_label: str) -> None:
    total = len(df)
    fg_pct = df["Shot Made Flag"].mean() * 100
    region_col = df["_region"]

    print(f"\n  [{split_label}]  n={total:,}  FG%={fg_pct:.1f}%")
    for region in ["paint", "midrange", "three_point", "backcourt"]:
        count = (region_col == region).sum()
        print(f"    {region:<12}: {count:5,}  ({count/total*100:.1f}%)")
    unmapped = region_col.isna().sum()
    if unmapped:
        print(f"    unmapped      : {unmapped:5,}")


def main():
    print("Loading dataset...")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Total rows: {len(raw):,}")

    for player in PLAYERS:
        df = raw[raw["Player Name"] == player].copy()

        # Derive season and filter to the 7 chosen seasons
        df["_season"] = derive_season(df["Game Date"])
        df = df[df["_season"].isin(ALL_SEASONS)].copy()

        df["is_3pt"] = (df["Shot Type"] == "3PT Field Goal").astype(int)

        # Random 75/25 split, stratified on make/miss to preserve FG%
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=df["Shot Made Flag"],
        )
        df["Split"] = "train"
        df.loc[test_idx, "Split"] = "test"

        # Attach region for stats (not saved to CSV)
        df["_region"] = df["Shot Zone Basic"].map(REGION_MAP)

        print(f"\n{'='*60}")
        print(f"  {player}  (total in window: {len(df):,})")
        print(f"{'='*60}")

        shots_per_season = (
            df.groupby("_season").size()
            .reindex(sorted(ALL_SEASONS))
        )
        print("  Shots per season:")
        for season, count in shots_per_season.items():
            print(f"    {season}-{str(season+1)[-2:]}  {count:,}")

        for split in ["train", "test"]:
            print_stats(player, df[df["Split"] == split], split)

        # Save: keep only requested columns
        out = df[KEEP_COLS].reset_index(drop=True)
        slug = player.lower().replace(" ", "_")
        out_path = f"data/{slug}_shots.csv"
        out.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path}  ({len(out):,} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
