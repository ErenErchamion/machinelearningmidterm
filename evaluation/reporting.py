from pathlib import Path
from typing import Dict, List

import pandas as pd


class ResultsReporter:
    def save_validation_results(self, rows: List[Dict], output_path: Path) -> None:
        # Drop heavy 'model' objects from rows before saving
        cleaned_rows = []
        for row in rows:
            row_copy = {k: v for k, v in row.items() if k != "model"}
            cleaned_rows.append(row_copy)
        df = pd.DataFrame(cleaned_rows)
        df.to_csv(output_path, index=False)

    def save_test_results(
        self,
        best_model_info: Dict,
        test_metrics: Dict,
        output_path: Path,
    ) -> None:
        data = {**{k: v for k, v in best_model_info.items() if k != "model"}, **{f"test_{k}": v for k, v in test_metrics.items()}}
        df = pd.DataFrame([data])
        df.to_csv(output_path, index=False)
