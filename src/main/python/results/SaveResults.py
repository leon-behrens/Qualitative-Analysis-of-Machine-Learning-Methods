import os
from datetime import datetime




class SaveResults:
    def __init__(self):
        """
        Initializes the ResultsSaver class.

        Args:
            save_dir (str): Directory where results will be saved.
        """
        self.save_dir = "/Users/leon/Uni/Master/Projektarbeit/Qualitative-Analysis-of-Machine-Learning-Methods/src/main/resources/data/"
        self.filename = "results"
        os.makedirs(self.save_dir, exist_ok=True)

    def save_as_csv(self, results):
        """
        Saves the results as a CSV file.

        Args:
            results (dict): Dictionary containing the results to save.
        """
        import pandas as pd

        # Check for uniform lengths and pad with None where necessary
        max_length = max(len(v) for v in results.values())
        for key in results:
            if len(results[key]) < max_length:
                results[key].extend([None] * (max_length - len(results[key])))

        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f"{self.filename}_{timestamp}.csv")
        results_df.to_csv(filepath, index=False)
        print(f"Results saved as CSV to {filepath}")
