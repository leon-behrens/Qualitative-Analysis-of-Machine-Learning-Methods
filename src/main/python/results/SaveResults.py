import os
from datetime import datetime
import pandas as pd


class SaveResults:
    def __init__(self):
        """
        Initializes the SaveResults class.
        """
        self.save_dir = "/scicore/home/bruder/behleo00/PA/src/main/resources/data/results"
        os.makedirs(self.save_dir, exist_ok=True)

    def save_as_csv(self, results):
        """
        Saves all results in a single CSV file after ensuring uniform lengths.

        Args:
            results (dict): Dictionary containing the results to save.
        """
        # Check for uniform lengths and pad with None where necessary
        max_length = max(len(v) for v in results.values())
        for key in results:
            if len(results[key]) < max_length:
                results[key].extend([None] * (max_length - len(results[key])))

        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f"results_{timestamp}.csv")
        results_df.to_csv(filepath, index=False)
        print(f"Results saved as CSV to {filepath}")

    def save_each_as_csv(self, results):
        """
        Saves each key in the results dictionary as a separate CSV file.

        Args:
            results (dict): Dictionary containing the results to save.
        """
        for key, values in results.items():
            filepath = os.path.join(self.save_dir, f"{key}.csv")
            df = pd.DataFrame({key: values})
            df.to_csv(filepath, index=False)
            print(f"{key.capitalize()} saved to {filepath}")
