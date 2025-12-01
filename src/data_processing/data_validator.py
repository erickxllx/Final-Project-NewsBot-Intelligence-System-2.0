import pandas as pd

class DataValidator:
    def __init__(self, required_columns=None):
        self.required_columns = required_columns or []

    def check_missing_columns(self, df: pd.DataFrame):
        """Verify if required columns exist."""
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    def check_null_values(self, df: pd.DataFrame):
        """Detect null values in dataset."""
        null_counts = df.isnull().sum()
        return null_counts[null_counts > 0]

    def check_duplicates(self, df: pd.DataFrame):
        """Identify duplicate rows."""
        return df[df.duplicated()]

    def validate(self, df: pd.DataFrame):
        """Run all validation checks."""
        self.check_missing_columns(df)
        nulls = self.check_null_values(df)
        duplicates = self.check_duplicates(df)

        return {
            "null_values": nulls,
            "duplicates": duplicates
        }
