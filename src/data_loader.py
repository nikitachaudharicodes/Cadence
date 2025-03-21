import pandas as pd

BARK_LABELS = ["/m/0btp2", "/m/07qf0zm", "/m/05r5c", "/m/02yds9", "/m/0brhx", "/m/03s_tn", "/m/0cq_cl"]
col_name = ["YTID", "start_seconds", "end_seconds", "positive_labels"]


def load_and_filter_data(csv_path):
    """Loads AudioSet CSV and filters only bark-related sounds."""

    df = pd.read_csv(csv_path, skiprows = 3, names=col_name, dtype = str, usecols=[0,1,2,3]) #skip the header
    
    df.dropna(inplace = True)

    df["positive_labels"] = df["positive_labels"].str.strip()

    df["positive_labels"] = df["positive_labels"].str.replace('"', "").str.strip()


    df_filtered = df[df["positive_labels"].str.contains("|".join(BARK_LABELS), na=False, regex=True)]

    df_filtered = df_filtered.copy()  

    df_filtered.loc[:, "bark_labels"] = df_filtered["positive_labels"].apply(
    lambda x: ",".join([label.strip().strip('"') for label in x.split(",") if label.strip().strip('"') in BARK_LABELS])
    )


    # df_filtered.loc[:, "bark_labels"] = df_filtered["positive_labels"].apply(
    #     lambda x: ",".join([label for label in x.split(",") if label in BARK_LABELS])
    # )

    df_filtered.to_csv("data/processed/bark_segments.csv", index=False)
    
    return df_filtered

if __name__ == "__main__":
    
    df_bark = load_and_filter_data("data/raw/balanced_train_segments.csv")
    
    print(f"Filtered dataset saved with {len(df_bark)} bark-related samples.")