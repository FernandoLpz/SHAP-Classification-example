import pandas as pd
from sklearn.model_selection import train_test_split

class Data:

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.data = pd.read_csv(self.csv_path)

        self._transform()
        self._split()

        pass

    def _transform(self) -> None:
        self.data.drop(['id'], axis=1, inplace=True)
        self.data.columns = self.data.columns.str.replace(' ', '_')

        self.classes = {0: 'M', 1: 'B'}
        self.data['diagnosis_result'].replace({'M': 0, 'B': 1}, inplace=True)

    def _split(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data.drop(['diagnosis_result'], axis=1), self.data['diagnosis_result'], test_size=0.25)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __str__(self) -> str:
        return f"Dataset with {self.data.shape[0]} samples and {self.data.shape[1] - 1} features"

    def __repr__(self) -> str:
        return f"Data({self.csv_path})"