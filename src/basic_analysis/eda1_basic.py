import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class ReferenceAnalysis:
    def __init__(self, data_files):
        """Initialize with dictionary of data files {threshold: filepath}"""
        self.data_files = data_files
        self.datasets = {}
        self.load_all_datasets()

    def load_all_datasets(self):
        """Load all threshold datasets"""
        for threshold, filepath in self.data_files.items():
            print(f"Loading {threshold} data...")
            try:
                df = pd.read_csv(filepath, sep='\t' if filepath.endswith('.xls') else ',')
                self.datasets[threshold] = df
                print(f"  Loaded: {len(df):,} interactions")
            except Exception as e:
                print(f"  Failed to load {filepath}: {e}")

    def question_1_total_interactions(self):
        results = {}
        print("=== QUESTION 1: Total Interactions Analysis ===")
        for threshold, df in self.datasets.items():
            p_cols = [col for col in df.columns if 'p_value' in col]
            df[p_cols] = df[p_cols].apply(pd.to_numeric, errors='coerce')
            total = len(df)
            significant = (df[p_cols] < 0.05).any(axis=1).sum()
            results[threshold] = {
                'total': total,
                'significant': significant,
                'significance_rate': significant / total if total > 0 else 0
            }
            print(f"  {threshold}: {total:,} total, {significant:,} significant ({significant/total*100:.1f}%)")
        return results

    def question_2_unique_genes(self):
        results = {}
        print("\n=== QUESTION 2: Unique Genes Analysis ===")
        for threshold, df in self.datasets.items():
            unique_genes = df['RefSeqName'].nunique()
            total_interactions = len(df)
            gene_freq = df['RefSeqName'].value_counts()
            top_1_percent = max(1, int(0.01 * len(gene_freq)))
            hub_genes = gene_freq.head(top_1_percent)
            results[threshold] = {
                'unique_genes': unique_genes,
                'avg_interactions_per_gene': total_interactions / unique_genes,
                'hub_genes': hub_genes.to_dict(),
                'gene_frequency_stats': {
                    'median': gene_freq.median(),
                    'mean': gene_freq.mean(),
                    'std': gene_freq.std()
                }
            }
            print(f"  {threshold}: {unique_genes:,} unique genes")
            print(f"    Avg interactions per gene: {total_interactions/unique_genes:.1f}")
            print(f"    Top connected gene: {gene_freq.index[0]} ({gene_freq.iloc[0]} interactions)")
        return results

    def question_3_intgroup_analysis(self):
        results = {}
        print("\n=== QUESTION 3: Interaction Type Analysis ===")
        for threshold, df in self.datasets.items():
            intgroup_counts = df['IntGroup'].value_counts()
            pp_data = df[df['IntGroup'] == 'PP']
            pd_data = df[df['IntGroup'] == 'PD']
            if len(pp_data) > 0 and len(pd_data) > 0:
                supp_cols = [col for col in df.columns if 'SuppPairs' in col]
                if supp_cols:
                    col = supp_cols[0]
                    pp_supp = pp_data[col].dropna()
                    pd_supp = pd_data[col].dropna()
                    if len(pp_supp) > 0 and len(pd_supp) > 0:
                        stat, p_val = stats.mannwhitneyu(pp_supp, pd_supp, alternative='two-sided')
                        results[threshold] = {
                            'counts': intgroup_counts.to_dict(),
                            'pp_median_supp': pp_supp.median(),
                            'pd_median_supp': pd_supp.median(),
                            'statistical_test': {
                                'statistic': stat,
                                'p_value': p_val,
                                'significant': p_val < 0.05
                            }
                        }
                        print(f"  {threshold}:")
                        for int_type, count in intgroup_counts.items():
                            pct = count / len(df) * 100
                            print(f"    {int_type}: {count:,} ({pct:.1f}%)")
                        print(f"    PP vs PD supporting pairs: p={p_val:.2e}")
        return results

    def question_4_distance_analysis(self):
            print("\n=== QUESTION 4: Distance Pattern Analysis ===")
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            for threshold, df in self.datasets.items():
                if 'distance' not in df.columns:
                    continue

                # Filter valid distances
                distances = df['distance'].dropna()
                distances = distances[distances > 0]
                if len(distances) > 0:
                    axes[0, 0].hist(np.log10(distances + 1), bins=50, alpha=0.7, label=threshold, density=True)

                # Filter sample for scatter plot
                sample_df = df.sample(n=10000, random_state=42) if len(df) > 10000 else df.copy()
                supp_cols = [col for col in df.columns if 'SuppPairs' in col]
                if supp_cols and 'distance' in sample_df.columns:
                    col = supp_cols[0]
                    sample_df = sample_df.dropna(subset=['distance', col])
                    sample_df = sample_df[(sample_df['distance'] > 0) & (sample_df[col] > 0)]
                    if not sample_df.empty:
                        axes[0, 1].scatter(
                            np.log10(sample_df['distance'] + 1),
                            np.log10(sample_df[col] + 1),
                            alpha=0.5, s=1, label=threshold
                        )

            axes[0, 0].set_xlabel('Log10(Distance + 1)')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Distance Distribution')
            axes[0, 0].legend()

            axes[0, 1].set_xlabel('Log10(Distance + 1)')
            axes[0, 1].set_ylabel('Log10(Supporting Pairs + 1)')
            axes[0, 1].set_title('Distance vs Interaction Strength')
            axes[0, 1].legend()

            distance_bins = [0, 50000, 100000, 500000, 1000000, 5000000, np.inf]
            distance_labels = ['<50kb', '50-100kb', '100-500kb', '500kb-1Mb', '1-5Mb', '>5Mb']
            category_data = []

            for threshold, df in self.datasets.items():
                if 'distance' not in df.columns:
                    continue
                df = df.copy()
                df = df[df['distance'] > 0]
                df['distance_category'] = pd.cut(df['distance'], bins=distance_bins, labels=distance_labels)
                cat_counts = df['distance_category'].value_counts()
                for cat, count in cat_counts.items():
                    category_data.append({'Threshold': threshold, 'Category': cat, 'Count': count})

            cat_df = pd.DataFrame(category_data)
            cat_pivot = cat_df.pivot(index='Category', columns='Threshold', values='Count').fillna(0)
            cat_pivot.plot(kind='bar', ax=axes[1, 0], stacked=False)
            axes[1, 0].set_title('Interactions by Distance Category')
            axes[1, 0].set_xlabel('Distance Category')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend(title='Threshold')
            axes[1, 0].tick_params(axis='x', rotation=45)

            stats_data = []
            for threshold, df in self.datasets.items():
                if 'distance' not in df.columns:
                    continue
                distances = df['distance'].dropna()
                distances = distances[distances > 0]
                stats_data.append({
                    'Threshold': threshold,
                    'Median': distances.median(),
                    'Mean': distances.mean(),
                    'Q75': distances.quantile(0.75),
                    'Q95': distances.quantile(0.95)
                })

            stats_df = pd.DataFrame(stats_data)
            x_pos = np.arange(len(stats_df))
            axes[1, 1].bar(x_pos - 0.2, stats_df['Median'], 0.4, label='Median', alpha=0.7)
            axes[1, 1].bar(x_pos + 0.2, stats_df['Mean'], 0.4, label='Mean', alpha=0.7)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(stats_df['Threshold'])
            axes[1, 1].set_ylabel('Distance (bp)')
            axes[1, 1].set_title('Distance Statistics by Threshold')
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')

            plt.tight_layout()
            os.makedirs('results/figures', exist_ok=True)
            plt.savefig('results/figures/enhanced_distance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            return stats_df


    def recalculate_distance(self):
        print("\n=== Distance Recalculation Analysis ===")
        for threshold, df in self.datasets.items():
            if not {'Interactor_Start', 'Feature_Start', 'Feature_Chr', 'Interactor_Chr', 'distance'}.issubset(df.columns):
                continue
            calculated_distance = np.abs(df['Interactor_Start'] - df['Feature_Start'])
            original_distance = df['distance']
            diff = np.abs(calculated_distance - original_distance)
            print(f"\n{threshold} Distance Verification:")
            print(f"  Perfect matches: {(diff < 1).sum():,} ({(diff < 1).mean()*100:.1f}%)")
            print(f"  Close matches (<1kb): {(diff < 1000).sum():,} ({(diff < 1000).mean()*100:.1f}%)")
            print(f"  Large discrepancies (>10kb): {(diff > 10000).sum():,}")
            if (diff > 10000).sum() > 0:
                print("  Investigating large discrepancies...")
                large_diff = df[diff > 10000].head()
                print("  Sample cases:")
                for idx, row in large_diff.iterrows():
                    print(f"    {row['Feature_Chr']}:{row['Feature_Start']} <-> {row['Interactor_Chr']}:{row['Interactor_Start']} "
                          f"(Original: {row['distance']}, Calculated: {calculated_distance.iloc[idx]})")

def main():
    data_files = {
        'FDR_0.1': r'data/raw/k562_FDR1.xls',
        'FDR_0.01': r'data/raw/k562_FDR01.xls',
        'FDR_0.001': r'data/raw/k562_FDR001.xls'
    }
    analyzer = ReferenceAnalysis(data_files)
    analyzer.question_1_total_interactions()
    analyzer.question_2_unique_genes()
    analyzer.question_3_intgroup_analysis()
    analyzer.question_4_distance_analysis()
    analyzer.recalculate_distance()
    return analyzer

if __name__ == "__main__":
    analyzer = main()
