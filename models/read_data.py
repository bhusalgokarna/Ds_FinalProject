
import pandas as pd

class DataLoader:
    def __init__(self):
        self.data = DataLoader.load_data()
    @staticmethod
    def load_data():
        file_id = "1c222AbSUMn9vKcepLZDnyCKUN2B8BQtP"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        df = pd.read_csv(url, sep=',')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.ffill()  # Forward fill to handle missing values
        return df




if __name__ == "__main__":
   loader = DataLoader()
   df = loader.load_data()
   print("Data loaded successfully.")
   df.head()