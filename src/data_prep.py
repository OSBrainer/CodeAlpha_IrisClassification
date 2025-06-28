import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(path: str) -> pd.DataFrame:
    """Load raw Iris CSV into a DataFrame."""
    return pd.read_csv(path)

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning (none needed) and ensure correct types."""
    df.drop('Id',axis=1,inplace=True)

    def fix_species(text):
       return text.split('-')[1].capitalize()
    
    df['Species'] = df['Species'].apply(fix_species)

    return df

def split_and_save(df: pd.DataFrame,
                   train_path: str,
                   test_path: str,
                   test_size: float = 0.33,
                   random_state: int = 42):
    """Split iris into stratified train/test and save CSVs."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['species']
    )
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

if __name__ == "__main__":
    raw = load_raw("../data/iris.csv")
    proc = clean_and_engineer(raw)
    proc.to_csv("../data/all_data.csv", index=False)
    split_and_save(proc,
                   "../data/train.csv",
                   "../data/test.csv")
