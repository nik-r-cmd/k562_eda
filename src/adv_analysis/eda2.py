import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
import collections
warnings.filterwarnings("ignore")

class ComprehensiveHiCAnalysis:
    def __init__(self, data_files):
        self.data_files = data_files
        self.datasets = {}
        self.results = {}
        self.load_and_preprocess_all()

    def _load_file(self, filepath):
        # Determine file type and read accordingly
        try:
            if filepath.lower().endswith(".xls") or filepath.lower().endswith(".xlsx"):
                # Handle old excel formats
                # LIBRARIES: pip install openpyxl xlrd --upgrade
                if filepath.lower().endswith(".xls"):
                    return pd.read_excel(filepath, engine="xlrd")
                else:
                    return pd.read_excel(filepath, engine="openpyxl")
            else:
                # Try tab-delimited, safer for your files
                return pd.read_csv(filepath, sep="\t")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def calculate_distances(self, df):
        # Use midpoint formula per literature: midpoint = (start + end)/2
        # Estimate missing 'Feature_End' if not present
        if "Feature_End" not in df.columns:
            df["Feature_End"] = df["Feature_Start"] + 1000  # default fragment length assumption

        df["Feature_Midpoint"] = (df["Feature_Start"] + df["Feature_End"]) / 2
        df["Interactor_Midpoint"] = (df["Interactor_Start"] + df["Interactor_End"]) / 2

        df["Distance_Midpoint"] = np.abs(df["Feature_Midpoint"] - df["Interactor_Midpoint"])
        df["Distance_Standard"] = np.abs(df["Feature_Start"] - df["Interactor_Start"])

        # Nullify distance for inter-chromosomal
        df.loc[df["Feature_Chr"] != df["Interactor_Chr"], ["Distance_Midpoint","Distance_Standard"]] = np.nan
        return df

    def classify_interactions(self, df):
        df["Interaction_Type"] = np.where(df["Feature_Chr"] == df["Interactor_Chr"], "Intra-chromosomal", "Inter-chromosomal")
        return df

    def load_and_preprocess_all(self):
        print("Loading and preprocessing datasets...")
        for label, path in self.data_files.items():
            df = self._load_file(path)
            if df is None:
                print(f"Skipping {label} due to load failure.")
                continue

            df.columns = df.columns.str.strip()
            # Convert relevant columns to numeric
            for c in df.columns:
                if any(x in c.lower() for x in ["start", "end", "distance", "pair", "pvalue", "supp"]):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df = self.calculate_distances(df)
            df = self.classify_interactions(df)
            self.datasets[label] = df

            # Print summary
            intra_ct = (df["Interaction_Type"] == "Intra-chromosomal").sum()
            total_ct = len(df)
            print(f"{label}: Loaded {total_ct:,} rows, {intra_ct/total_ct*100:.2f}% intra-chromosomal")

    # Question 1
    def question_1(self):
        print("\n=== Question 1: Count of Intra- vs Inter-Chromosomal Interactions ===")
        summary = []
        for label, df in self.datasets.items():
            cts = df["Interaction_Type"].value_counts()
            intra = cts.get("Intra-chromosomal", 0)
            inter = cts.get("Inter-chromosomal", 0)
            total = len(df)
            print(f"{label}: Total={total:,} | Intra={intra:,} ({intra/total*100:.2f}%) | Inter={inter:,} ({inter/total*100:.2f}%) | Ratio={intra/(inter if inter>0 else 1):.1f}:1")
            summary.append({"Label": label, "Intra": intra, "Inter": inter, "Total": total,
                            "Intra%": intra/total*100, "Inter%": inter/total*100})

        # Plot
        df_s = pd.DataFrame(summary)
        x = np.arange(len(df_s))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(x - width/2, df_s["Intra"], width, label="Intra-chromosomal", color="skyblue")
        ax.bar(x + width/2, df_s["Inter"], width, label="Inter-chromosomal", color="lightcoral")

        ax.set_xticks(x)
        ax.set_xticklabels(df_s["Label"])
        ax.set_yscale("log")
        ax.set_ylabel("Number of interactions (log scale)")
        ax.set_title("Intra- vs Inter-chromosomal Interaction Counts")
        ax.legend()
        plt.tight_layout()
        plt.savefig("results/figures/question1_counts.png", dpi=300)
        plt.show()

        return summary

    # Question 2
    def question_2(self):
        print("\n=== Question 2: Compare Distance, Pairs, P-values Between Interaction Types ===")
        for label, df in self.datasets.items():
            print(f"\n{label}:")
            intra = df[df["Interaction_Type"] == "Intra-chromosomal"]
            inter = df[df["Interaction_Type"] == "Inter-chromosomal"]

            # Distance stats (only intra since inter dist is NaN)
            dist = intra["Distance_Midpoint"].dropna()
            print(f"Intra-chromosomal distance (Midpoint) median: {dist.median():,.0f} bp, mean: {dist.mean():,.0f} bp, count: {len(dist):,}")

            # Supporting pairs stat - pick first found pair col
            pair_cols = [c for c in df.columns if "supp" in c.lower() and "pair" in c.lower()]
            if len(pair_cols) > 0:
                pair_col = pair_cols[0]
                intra_pairs = intra[pair_col].dropna()
                inter_pairs = inter[pair_col].dropna()
                print(f"Supporting pairs ({pair_col}): Intra median={intra_pairs.median()}, Inter median={inter_pairs.median()}")

                # Statistical test
                if len(intra_pairs)>5 and len(inter_pairs)>5:
                    stat, pval = stats.mannwhitneyu(intra_pairs, inter_pairs, alternative="two-sided")
                    print(f"Mann-Whitney U test p-value between intra & inter supporting pairs: {pval:.3e}")

            # p-value stats - pick first pvalue col
            pcols = [c for c in df.columns if "pvalue" in c.lower() or "p_value" in c.lower()]
            if len(pcols) > 0:
                pcol = pcols[0]
                intra_pval = intra[pcol].dropna()
                inter_pval = inter[pcol].dropna()
                print(f"P-values ({pcol}): Intra median={intra_pval.median()}, Inter median={inter_pval.median()}")

                if len(intra_pval)>5 and len(inter_pval)>5:
                    stat, pval = stats.mannwhitneyu(intra_pval, inter_pval, alternative="two-sided")
                    print(f"Mann-Whitney U test p-value between intra & inter p-values: {pval:.3e}")

    # Question 3
    def question_3(self):
        print("\n=== Question 3: Interaction Presence Across Treatments ===")
        for label, df in self.datasets.items():
            print(f"\n{label}:")
            treatment_cols = [c for c in df.columns if any(x in c.lower() for x in ["normal", "carboplatin", "gemcitabine"])]
            if len(treatment_cols) == 0:
                print("No treatment-related columns found.")
                continue
            for col in treatment_cols:
                treated = df[df[col] == 1]
                untreated = df[df[col] == 0]

                # Counts by interaction type
                treated_cts = treated["Interaction_Type"].value_counts()
                untreated_cts = untreated["Interaction_Type"].value_counts()

                print(f"Treatment: {col}")
                print(f"  Treated - Intra: {treated_cts.get('Intra-chromosomal',0)}, Inter: {treated_cts.get('Inter-chromosomal',0)}")
                print(f"  Untreated - Intra: {untreated_cts.get('Intra-chromosomal',0)}, Inter: {untreated_cts.get('Inter-chromosomal',0)}")

                # Chi-square test
                crosstab = pd.crosstab(df["Interaction_Type"], df[col])
                if crosstab.shape==(2,2):
                    stat, p, dof, expected = stats.chi2_contingency(crosstab)
                    print(f"  Chi-square p-value: {p:.3e}")

    # Question 4
    def question_4(self):
        print("\n=== Question 4: Chromosomes Involved Most in Interactions (Inter-Chromosomal) ===")
        import collections
        for label, df in self.datasets.items():
            print(f"\n{label}:")
            inter = df[df["Interaction_Type"] == "Inter-chromosomal"]

            if len(inter) == 0:
                print("No inter-chromosomal interactions")
                continue

            # Count chromosomes on both sides
            chrs = list(inter["Feature_Chr"]) + list(inter["Interactor_Chr"])
            counts = collections.Counter(chrs)

            total_slots = len(inter)*2
            most_common = counts.most_common()

            for i, (chr_, count) in enumerate(most_common[:10], 1):
                print(f"  {i}. {chr_}: {count} ({count/total_slots*100:.2f}%)")

            # Compute concentration in top 5
            top5_sum = sum(x[1] for x in most_common[:5])
            conc = top5_sum / total_slots * 100
            print(f"  Top 5 chromosomes = {conc:.2f}% of all inter-chromosomal interactions")

    # Run all questions
    def run_all(self):
        res1 = self.question_1()
        self.question_2()
        self.question_3()
        self.question_4()
        return res1

if __name__ == "__main__":
    # Define your (corrected) data paths here
    data_files = {
        "FDR_0.1": "data/raw/k562_FDR1.tsv",
        "FDR_0.01": "data/raw/k562_FDR01.tsv",
        "FDR_0.001": "data/raw/k562_FDR001.tsv"
    }
    
    analysis = ComprehensiveHiCAnalysis(data_files)
    analysis.run_all()
