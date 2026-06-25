import pandas as pd


class DataLoader:
    """Handles loading the Kaggle dataset, column normalizations,
    row-level validations, and median BPM imputation."""

    def __init__(self, filepath: str) -> None:
        """Stores the dataset file path."""
        self.filepath: str = filepath
        self.df: pd.DataFrame = None
        self.is_loaded: bool = False

    def load(self) -> pd.DataFrame:
        """Reads raw CSV data via pd.read_csv.
        Raises FileNotFoundError if the file is missing."""
        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file not found at: {self.filepath}"
            )
        return self.df

    def clean(self) -> pd.DataFrame:
        """Normalizes columns, drops rows with missing mental-health labels,
        and imputes missing BPM values with the column median."""
        # Normalize column names by stripping whitespace
        self.df.columns = self.df.columns.str.strip()

        # Drop rows where any mental health label is missing
        self.df = self.df.dropna(subset=['Anxiety', 'Depression', 'Insomnia', 'OCD'])

        # Impute missing BPM values with the column median
        self.df['BPM'] = self.df['BPM'].fillna(self.df['BPM'].median())

        return self.df

    def validate(self) -> bool:
        """Verifies data volume is at least 500 rows.
        Raises ValueError if row count is below 500."""
        if len(self.df) < 500:
            raise ValueError(
                f"Insufficient data: expected at least 500 rows, "
                f"but got {len(self.df)}."
            )
        return True

    def get_data(self) -> pd.DataFrame:
        """Runs the full loader pipeline: load -> clean -> validate.
        Sets is_loaded flag and returns the cleaned DataFrame."""
        self.load()
        self.clean()
        self.validate()
        self.is_loaded = True
        return self.df

    def get_unique_genres(self) -> list:
        """Returns a sorted list of unique favorite genres."""
        return sorted(self.df['Fav genre'].dropna().unique().tolist())

    def get_unique_services(self) -> list:
        """Returns a sorted list of unique streaming platforms."""
        return sorted(
            self.df['Primary streaming service'].dropna().unique().tolist()
        )
