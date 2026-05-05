"""
Module: replace_column.py

Reads:
  - sample_submission.csv
  - submission.0.csv

Replaces:
  - Column 3 in sample_submission.csv
With:
  - Column 3 from submission.0.csv

Writes:
  - t.submission.csv (Unix EOL: \n)
"""

from __future__ import annotations

import csv


def main() -> None:
    with open("sample_submission.csv", newline="") as sample_file, open(
        "submission.0.csv", newline=""
    ) as sub_file, open("t.submission.csv", "w", newline="") as out_file:

        sample_reader = csv.reader(sample_file)
        sub_reader = csv.reader(sub_file)

        # Force Unix newlines
        writer = csv.writer(out_file, lineterminator="\n")

        for sample_row, sub_row in zip(sample_reader, sub_reader):
            assert len(sample_row) >= 3, "sample_submission.csv must have at least 3 columns."
            assert len(sub_row) >= 2, "submission.0.csv must have at least 2 columns."

            # Replace column 3 (index 2)
            sample_row[2] = sub_row[1]

            writer.writerow(sample_row)


if __name__ == "__main__":
    main()
