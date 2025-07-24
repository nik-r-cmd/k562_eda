"""
Create interactive visualizations for Hi-C insights
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class InteractiveVisualizer:
    def __init__(self, datasets, analysis_results):
        self.datasets = datasets
        self.results = analysis_results
    
    def create_threshold_comparison_dashboard(self):
        """Create interactive dashboard comparing thresholds"""
        
        # Prepare data for visualization
        comparison_data = []
        for threshold, df in self.datasets.items():
            total_interactions = len(df)
            unique_genes = df['RefSeqName'].nunique()
            pp_count = (df['IntGroup'] == 'PP').sum()
            pd_count = (df['IntGroup'] == 'PD').sum()
            avg_distance = df['distance'].median()
            
            comparison_data.append({
                'Threshold': threshold,
                'Total_Interactions': total_interactions,
                'Unique_Genes': unique_genes,
                'PP_Interactions': pp_count,
                'PD_Interactions': pd_count,
                'Median_Distance': avg_distance,
                'PP_Percentage': pp_count / (pp_count + pd_count) * 100 if (pp_count + pd_count) > 0 else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Total Interactions', 'Unique Genes', 'PP vs PD Distribution',
                          'Median Distance', 'Interaction Density', 'PP/PD Ratio'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Total Interactions
        fig.add_trace(
            go.Bar(x=comparison_df['Threshold'], y=comparison_df['Total_Interactions'],
                   name='Total Interactions', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Unique Genes
        fig.add_trace(
            go.Bar(x=comparison_df['Threshold'], y=comparison_df['Unique_Genes'],
                   name='Unique Genes', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # PP vs PD for first threshold (example)
        fig.add_trace(
            go.Pie(labels=['PP', 'PD'], 
                   values=[comparison_df.iloc[0]['PP_Interactions'], 
                          comparison_df.iloc[0]['PD_Interactions']],
                   name=f"{comparison_df.iloc[0]['Threshold']} PP/PD"),
            row=1, col=3
        )
        
        # Median Distance
        fig.add_trace(
            go.Bar(x=comparison_df['Threshold'], y=comparison_df['Median_Distance'],
                   name='Median Distance', marker_color='salmon'),
            row=2, col=1
        )
        
        # Interaction Density (interactions per gene)
        comparison_df['Interaction_Density'] = comparison_df['Total_Interactions'] / comparison_df['Unique_Genes']
        fig.add_trace(
            go.Bar(x=comparison_df['Threshold'], y=comparison_df['Interaction_Density'],
                   name='Interactions per Gene', marker_color='gold'),
            row=2, col=2
        )
        
        # PP Percentage
        fig.add_trace(
            go.Bar(x=comparison_df['Threshold'], y=comparison_df['PP_Percentage'],
                   name='PP Percentage', marker_color='purple'),
            row=2, col=3
        )
        
        fig.update_layout(height=800, showlegend=False,
                         title_text="Hi-C Data Threshold Comparison Dashboard")
        
        return fig
    
    def create_distance_interaction_heatmap(self):
        """Create interactive heatmap of distance vs interaction strength"""
        
        # Combine all datasets
        all_data = []
        for threshold, df in self.datasets.items():
            df_sample = df.sample(n=min(5000, len(df)), random_state=42)  # Sample for performance
            df_sample['Threshold'] = threshold
            all_data.append(df_sample)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create distance bins
        combined_df['distance_bin'] = pd.cut(
            combined_df['distance'], 
            bins=[0, 50000, 100000, 500000, 1000000, 5000000, np.inf],
            labels=['<50kb', '50-100kb', '100-500kb', '500kb-1Mb', '1-5Mb', '>5Mb']
        )
        
        # Get supporting pairs data
        supp_cols = [col for col in combined_df.columns if 'SuppPairs' in col]
        if supp_cols:
            # Create heatmap data
            heatmap_data = combined_df.groupby(['Threshold', 'distance_bin'])[supp_cols[0]].median().reset_index()
            heatmap_pivot = heatmap_data.pivot(index='distance_bin', columns='Threshold', values=supp_cols[0])
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='Viridis',
                text=np.round(heatmap_pivot.values, 1),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Median Supporting Pairs by Distance and Threshold',
                xaxis_title='Threshold',
                yaxis_title='Distance Category',
                width=800,
                height=500
            )
            
            return fig
        
        return None
    
    def create_drug_response_network(self):
        """Create network visualization of drug-specific interactions"""
        
        # Analyze drug effects across thresholds
        drug_effects = []
        
        for threshold, df in self.datasets.items():
            conditions = ['Normal', 'CarboplatinTreated', 'GemcitabineTreated']
            
            for condition in conditions:
                if condition in df.columns:
                    condition_interactions = df[df[condition] == 1]
                    
                    # Top interacting genes
                    top_genes = condition_interactions['RefSeqName'].value_counts().head(10)
                    
                    for gene, count in top_genes.items():
                        drug_effects.append({
                            'Threshold': threshold,
                            'Condition': condition,
                            'Gene': gene,
                            'Interaction_Count': count
                        })
        
        drug_df = pd.DataFrame(drug_effects)
        
        if len(drug_df) > 0:
            # Create interactive scatter plot
            fig = px.scatter(
                drug_df, 
                x='Condition', 
                y='Interaction_Count',
                color='Threshold',
                size='Interaction_Count',
                hover_data=['Gene'],
                title='Drug Response: Top Interacting Genes by Condition',
                labels={'Interaction_Count': 'Number of Interactions'}
            )
            
            fig.update_layout(
                width=900,
                height=600,
                xaxis_title='Treatment Condition',
                yaxis_title='Number of Interactions'
            )
            
            return fig
        
        return None
    
    def create_gene_interaction_network_plot(self):
        """Create network plot of top interacting genes"""
        
        # Use the most stringent threshold for cleaner visualization 
        threshold = list(self.datasets.keys())[0]  # Assuming first is most stringent
        df = self.datasets[threshold]
        
        # Get top 20 most connected genes
        gene_counts = df['RefSeqName'].value_counts().head(20)
        top_genes = set(gene_counts.index)
        
        # Filter interactions between top genes
        network_interactions = df[
            (df['RefSeqName'].isin(top_genes)) & 
            (df['InteractorName'].isin(top_genes))
        ]
        
        if len(network_interactions) > 0:
            # Create network graph data
            nodes = []
            edges = []
            
            # Add nodes
            for gene in top_genes:
                nodes.append({
                    'id': gene,
                    'label': gene,
                    'size': gene_counts.get(gene, 0),
                    'color': 'lightblue'
                })
            
            # Add edges
            for _, row in network_interactions.iterrows():
                edges.append({
                    'source': row['RefSeqName'],
                    'target': row['InteractorName'],
                    'weight': row.get('KN1_SuppPairs', 1)  # Use normal condition supporting pairs
                })
            
            # Create plotly network visualization
            # This is a simplified version - for full network viz, consider using cytoscape or networkx
            
            gene_list = list(top_genes)
            x_pos = np.cos(np.linspace(0, 2*np.pi, len(gene_list)))
            y_pos = np.sin(np.linspace(0, 2*np.pi, len(gene_list)))
            
            fig = go.Figure()
            
            # Add edges
            for edge in edges:
                source_idx = gene_list.index(edge['source'])
                target_idx = gene_list.index(edge['target'])
                
                fig.add_trace(go.Scatter(
                    x=[x_pos[source_idx], x_pos[target_idx], None],
                    y=[y_pos[source_idx], y_pos[target_idx], None],
                    mode='lines',
                    line=dict(width=0.5, color='gray'),
                    hoverinfo='none',
                    showlegend=False
                ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+text',
                marker=dict(
                    size=[gene_counts.get(gene, 0)/5 for gene in gene_list],
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                ),
                text=gene_list,
                textposition="middle center",
                hovertemplate='<b>%{text}</b><br>Interactions: %{marker.size}<extra></extra>',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f'Gene Interaction Network ({threshold})',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Node size represents interaction frequency",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(color="gray", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,
                height=800
            )
            
            return fig
        
        return None

def main():
    """Create all interactive visualizations"""
    # You'll need to load your data and analysis results
    # This is a template showing how to use the visualizer
    
    print("Creating interactive visualizations...")
    print("Save the returned figures using:")
    print("fig.write_html('results/interactive_dashboard.html')")
    print("fig.show()  # To display in browser")

if __name__ == "__main__":
    main()
