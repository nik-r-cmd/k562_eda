import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class HiddenPatternDiscovery:
    def __init__(self, datasets):
        self.datasets = datasets

    def discover_interaction_hotspots(self):
        print("=== DISCOVERING INTERACTION HOTSPOTS ===")
        hotspots = {}
        for threshold, df in self.datasets.items():
            print(f"\nAnalyzing {threshold}...")
            feature_density = df.groupby('Feature_Chr').size()
            interactor_density = df.groupby('Interactor_Chr').size()
            feature_hotspots = feature_density.quantile(0.8)
            hot_chroms = feature_density[feature_density >= feature_hotspots].index.tolist()
            hotspot_data = df[df['Feature_Chr'].isin(hot_chroms) | df['Interactor_Chr'].isin(hot_chroms)]
            total_interactions = len(df)
            hotspot_interactions = len(hotspot_data)
            hotspot_percentage = hotspot_interactions / total_interactions * 100
            intgroup_counts = hotspot_data['IntGroup'].value_counts(normalize=True) * 100
            supp_cols = [col for col in df.columns if 'SuppPairs' in col]
            avg_supporting = hotspot_data[supp_cols].mean().mean() if supp_cols else np.nan
            hotspots[threshold] = {
                'hot_chromosomes': hot_chroms,
                'hotspot_percentage': hotspot_percentage,
                'interaction_types': intgroup_counts.to_dict(),
                'avg_distance': hotspot_data['distance'].median(),
                'avg_supporting_pairs': avg_supporting
            }
            print(f"  Hotspot chromosomes: {hot_chroms}")
            print(f"  {hotspot_percentage:.1f}% of interactions involve hotspot regions")
        return hotspots

    def find_distance_anomalies(self):
        print("\n=== FINDING DISTANCE ANOMALIES ===")
        anomalies = {}
        for threshold, df in self.datasets.items():
            print(f"\nAnalyzing {threshold}...")
            supp_cols = [col for col in df.columns if 'SuppPairs' in col and 'KN' in col]
            if not supp_cols:
                supp_cols = [col for col in df.columns if 'SuppPairs' in col]
            if not supp_cols:
                continue
            df_clean = df.dropna(subset=['distance', supp_cols[0]])
            distances = df_clean['distance']
            supporting_pairs = df_clean[supp_cols[0]]
            distance_bins = np.logspace(3, 8, 20)
            bin_indices = np.digitize(distances, distance_bins)
            expected_supp, actual_supp = [], []
            for bin_idx in range(1, len(distance_bins)):
                mask = bin_indices == bin_idx
                if mask.sum() > 10:
                    expected = supporting_pairs[mask].median()
                    expected_supp.extend([expected] * mask.sum())
                    actual_supp.extend(supporting_pairs[mask].values)
            if len(expected_supp) == 0:
                continue
            expected_array = np.array(expected_supp)
            actual_array = np.array(actual_supp)
            residuals = actual_array - expected_array
            z_scores = stats.zscore(residuals)
            strong_anomalies = np.abs(z_scores) > 3
            anomaly_count = strong_anomalies.sum()
            anomalies[threshold] = {
                'total_anomalies': anomaly_count,
                'anomaly_percentage': anomaly_count / len(actual_array) * 100,
                'extreme_positive': (z_scores > 3).sum(),
                'extreme_negative': (z_scores < -3).sum()
            }
            print(f"  Found {anomaly_count:,} distance anomalies ({anomaly_count/len(actual_array)*100:.2f}%)")
            print(f"    Strong outliers: {(z_scores > 3).sum():,}")
            print(f"    Weak outliers: {(z_scores < -3).sum():,}")
        return anomalies

    def discover_drug_specific_networks(self):
        print("\n=== DISCOVERING DRUG-SPECIFIC NETWORKS ===")
        networks = {}
        for threshold, df in self.datasets.items():
            print(f"\nAnalyzing {threshold}...")
            conditions = ['Normal', 'CarboplatinTreated', 'GemcitabineTreated']
            condition_specific = {}
            for condition in conditions:
                if condition in df.columns:
                    other_conditions = [c for c in conditions if c != condition and c in df.columns]
                    if other_conditions:
                        specific_mask = (df[condition] == 1)
                        for other in other_conditions:
                            specific_mask &= (df[other] == 0)
                        specific_interactions = df[specific_mask]
                        condition_specific[condition] = specific_interactions
                        if not specific_interactions.empty:
                            print(f"  {condition}-specific interactions: {len(specific_interactions):,}")
                            network_props = self._analyze_network_properties(specific_interactions)
                            condition_specific[f"{condition}_properties"] = network_props
            networks[threshold] = condition_specific
        return networks

    def _analyze_network_properties(self, interactions_df):
        if interactions_df.empty:
            return {}
        all_genes = set(interactions_df['RefSeqName'].dropna()) | set(interactions_df['InteractorName'].dropna())
        gene_counts = interactions_df['RefSeqName'].value_counts()
        hub_genes = gene_counts[gene_counts > gene_counts.quantile(0.9)].to_dict()
        distance_stats = {
            'median_distance': interactions_df['distance'].median(),
            'mean_distance': interactions_df['distance'].mean(),
            'long_range_interactions': (interactions_df['distance'] > 1_000_000).sum()
        }
        type_dist = interactions_df['IntGroup'].value_counts(normalize=True).to_dict()
        return {
            'unique_genes': len(all_genes),
            'hub_genes': hub_genes,
            'distance_stats': distance_stats,
            'interaction_types': type_dist,
            'total_interactions': len(interactions_df)
        }

    def find_threshold_transitions(self):
        print("\n=== ANALYZING THRESHOLD TRANSITIONS ===")
        all_interactions = []
        for threshold, df in self.datasets.items():
            df_copy = df.copy()
            df_copy['threshold'] = threshold
            df_copy['interaction_id'] = (
                df_copy['Feature_Chr'].astype(str) + '_' +
                df_copy['Feature_Start'].astype(str) + '_' +
                df_copy['Interactor_Chr'].astype(str) + '_' +
                df_copy['Interactor_Start'].astype(str)
            )
            all_interactions.append(df_copy)
        combined_df = pd.concat(all_interactions, ignore_index=True)
        interaction_counts = combined_df['interaction_id'].value_counts()
        stable = interaction_counts[interaction_counts == len(self.datasets)].index
        specific = interaction_counts[interaction_counts == 1].index
        partial = interaction_counts[(interaction_counts > 1) & (interaction_counts < len(self.datasets))].index
        results = {
            'stable_interactions': len(stable),
            'threshold_specific': len(specific),
            'partial_interactions': len(partial),
            'total_unique_interactions': len(interaction_counts)
        }
        print(f"  Stable interactions (all thresholds): {len(stable):,}")
        print(f"  Threshold-specific interactions: {len(specific):,}")
        print(f"  Partial interactions: {len(partial):,}")
        stable_df = combined_df[combined_df['interaction_id'].isin(stable)]
        specific_df = combined_df[combined_df['interaction_id'].isin(specific)]
        if not stable_df.empty and not specific_df.empty:
            print("\n  Stable vs Specific Interaction Characteristics:")
            print(f"    Stable - Median distance: {stable_df['distance'].median():,.0f} bp")
            print(f"    Specific - Median distance: {specific_df['distance'].median():,.0f} bp")
            supp_cols = [col for col in stable_df.columns if 'SuppPairs' in col]
            if supp_cols:
                col = supp_cols[0]
                print(f"    Stable - Median supporting pairs: {stable_df[col].median():.1f}")
                print(f"    Specific - Median supporting pairs: {specific_df[col].median():.1f}")
        return results, stable, specific

    def cluster_interaction_patterns(self):
        print("\n=== CLUSTERING INTERACTION PATTERNS ===")
        clustering_results = {}
        for threshold, df in self.datasets.items():
            print(f"\nClustering {threshold} interactions...")
            feature_cols = []
            # Filter out rows with non-positive or missing values before log transform
            df = df.copy()
            df = df[df['distance'] > 0].dropna(subset=['distance'])

            # Safe log10 transform
            df['log_distance'] = np.log10(df['distance'])

            feature_cols = ['log_distance']

            # Supporting pairs (log-transformed)
            supp_cols = [col for col in df.columns if 'SuppPairs' in col]
            for col in supp_cols[:3]:
                df = df[df[col] > 0].dropna(subset=[col])  # <-- filter + dropna
                df[f'log_{col}'] = np.log10(df[col])
                feature_cols.append(f'log_{col}')

            # P-values (-log10 transformed)
            p_cols = [col for col in df.columns if 'p_value' in col]
            for col in p_cols[:3]:
                df = df[df[col] > 0].dropna(subset=[col])  # <-- filter + dropna
                df[f'neglog_{col}'] = -np.log10(df[col])
                feature_cols.append(f'neglog_{col}')

            feature_matrix = df[feature_cols].dropna()
            if len(feature_matrix) > 1000:
                feature_matrix = feature_matrix.sample(n=1000, random_state=42)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
            clustering = DBSCAN(eps=0.5, min_samples=10)
            cluster_labels = clustering.fit_predict(features_scaled)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            print(f"  Found {n_clusters} clusters with {n_noise} noise points")
            cluster_characteristics = {}
            feature_matrix['cluster'] = cluster_labels
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    continue
                cluster_data = feature_matrix[feature_matrix['cluster'] == cluster_id]
                cluster_characteristics[cluster_id] = {
                    'size': len(cluster_data),
                    'avg_distance': np.power(10, cluster_data['log_distance'].mean()) - 1,
                    'avg_supporting_pairs': np.power(10, cluster_data[[col for col in cluster_data.columns if 'log_' in col and 'SuppPairs' in col][0]].mean()) - 1 if any('SuppPairs' in col for col in cluster_data.columns) else 0
                }
            clustering_results[threshold] = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'cluster_characteristics': cluster_characteristics
            }
        return clustering_results


def main():
    data_files = {
        'FDR_0.1': r'data/raw/k562_FDR1.xls',
        'FDR_0.01': r'data/raw/k562_FDR01.xls',
        'FDR_0.001': r'data/raw/k562_FDR001.xls'
    }
    datasets = {}
    for threshold, filepath in data_files.items():
        datasets[threshold] = pd.read_csv(filepath, sep='\t' if filepath.endswith('.xls') else ',')
    discovery = HiddenPatternDiscovery(datasets)
    hotspots = discovery.discover_interaction_hotspots()
    anomalies = discovery.find_distance_anomalies()
    networks = discovery.discover_drug_specific_networks()
    transitions, stable, specific = discovery.find_threshold_transitions()
    clusters = discovery.cluster_interaction_patterns()
    print("\n" + "=" * 50)
    print("=" * 50)
    return discovery, hotspots, anomalies, networks, transitions, clusters


if __name__ == "__main__":
    discovery, hotspots, anomalies, networks, transitions, clusters = main()
