"""
Utility functions for visualization, export, and analysis.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path


class Visualization:
    """Visualization utilities."""
    
    @staticmethod
    def plot_distributions(measurements: List[Dict], 
                          output_dir: Optional[str] = None):
        """
        Plot feature distributions.
        
        Args:
            measurements: List of measurement dictionaries
            output_dir: Optional directory to save plots
        """
        df = pd.DataFrame(measurements)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Fish Measurement Distributions", fontsize=16)
        
        # Length distribution
        axes[0, 0].hist(df['length_cm'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Length (cm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Length Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height distribution
        axes[0, 1].hist(df['height_cm'], bins=20, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Height (cm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Area distribution
        axes[1, 0].hist(df['area_cm2'], bins=20, color='salmon', edgecolor='black')
        axes[1, 0].set_xlabel('Area (cm²)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Area Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Length vs Height
        axes[1, 1].scatter(df['length_cm'], df['height_cm'], alpha=0.6)
        axes[1, 1].set_xlabel('Length (cm)')
        axes[1, 1].set_ylabel('Height (cm)')
        axes[1, 1].set_title('Length vs Height')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "feature_distributions.png"
            plt.savefig(output_path, dpi=100)
            print(f"✅ Plot saved to {output_path}")
        
        plt.show()
    
    @staticmethod
    def plot_weight_analysis(measurements: List[Dict], 
                            output_dir: Optional[str] = None):
        """
        Plot weight-related analysis.
        
        Args:
            measurements: List of measurement dictionaries with predicted_weight_kg
            output_dir: Optional directory to save plots
        """
        df = pd.DataFrame(measurements)
        
        if 'predicted_weight_kg' not in df.columns:
            print("⚠️  No predicted_weight_kg column found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Weight Prediction Analysis", fontsize=16)
        
        # Weight distribution
        axes[0, 0].hist(df['predicted_weight_kg'], bins=20, color='orange', edgecolor='black')
        axes[0, 0].set_xlabel('Weight (kg)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Predicted Weight Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Length vs Weight
        axes[0, 1].scatter(df['length_cm'], df['predicted_weight_kg'], alpha=0.6, color='blue')
        axes[0, 1].set_xlabel('Length (cm)')
        axes[0, 1].set_ylabel('Weight (kg)')
        axes[0, 1].set_title('Length vs Predicted Weight')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Area vs Weight
        axes[1, 0].scatter(df['area_cm2'], df['predicted_weight_kg'], alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Area (cm²)')
        axes[1, 0].set_ylabel('Weight (kg)')
        axes[1, 0].set_title('Area vs Predicted Weight')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative weight
        cumsum = df['predicted_weight_kg'].cumsum()
        axes[1, 1].plot(cumsum.values, marker='o', linestyle='-', color='red')
        axes[1, 1].set_xlabel('Fish #')
        axes[1, 1].set_ylabel('Cumulative Weight (kg)')
        axes[1, 1].set_title('Cumulative Weight')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "weight_analysis.png"
            plt.savefig(output_path, dpi=100)
            print(f"✅ Plot saved to {output_path}")
        
        plt.show()
    
    @staticmethod
    def plot_summary(measurements: List[Dict], 
                    output_dir: Optional[str] = None):
        """
        Create summary statistics plot.
        
        Args:
            measurements: List of measurement dictionaries
            output_dir: Optional directory to save plots
        """
        df = pd.DataFrame(measurements)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Summary Statistics", fontsize=16)
        
        # Box plots
        data_to_plot = [df['length_cm'], df['height_cm'], df['area_cm2']]
        axes[0].boxplot(data_to_plot, labels=['Length (cm)', 'Height (cm)', 'Area (cm²)'])
        axes[0].set_title('Measurement Box Plots')
        axes[0].grid(True, alpha=0.3)
        
        # Summary stats table
        stats = {
            'Metric': ['Length (cm)', 'Height (cm)', 'Area (cm²)'],
            'Mean': [
                f"{df['length_cm'].mean():.2f}",
                f"{df['height_cm'].mean():.2f}",
                f"{df['area_cm2'].mean():.2f}"
            ],
            'Std': [
                f"{df['length_cm'].std():.2f}",
                f"{df['height_cm'].std():.2f}",
                f"{df['area_cm2'].std():.2f}"
            ],
            'Min': [
                f"{df['length_cm'].min():.2f}",
                f"{df['height_cm'].min():.2f}",
                f"{df['area_cm2'].min():.2f}"
            ],
            'Max': [
                f"{df['length_cm'].max():.2f}",
                f"{df['height_cm'].max():.2f}",
                f"{df['area_cm2'].max():.2f}"
            ]
        }
        
        axes[1].axis('tight')
        axes[1].axis('off')
        table = axes[1].table(cellText=[[stats['Metric'][i], stats['Mean'][i], 
                                        stats['Std'][i], stats['Min'][i], 
                                        stats['Max'][i]] 
                                       for i in range(3)],
                             colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                             cellLoc='center',
                             loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "summary_statistics.png"
            plt.savefig(output_path, dpi=100)
            print(f"✅ Plot saved to {output_path}")
        
        plt.show()


class DataExport:
    """Data export utilities."""
    
    @staticmethod
    def export_csv(measurements: List[Dict], filepath: str) -> bool:
        """
        Export measurements to CSV.
        
        Args:
            measurements: List of measurement dictionaries
            filepath: Output CSV file path
            
        Returns:
            True if successful
        """
        try:
            df = pd.DataFrame(measurements)
            df.to_csv(filepath, index=False)
            print(f"✅ Exported {len(df)} records to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")
            return False
    
    @staticmethod
    def export_json(measurements: List[Dict], filepath: str) -> bool:
        """
        Export measurements to JSON.
        
        Args:
            measurements: List of measurement dictionaries
            filepath: Output JSON file path
            
        Returns:
            True if successful
        """
        try:
            df = pd.DataFrame(measurements)
            df.to_json(filepath, orient='records', indent=2)
            print(f"✅ Exported to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error exporting JSON: {e}")
            return False
    
    @staticmethod
    def generate_html_report(measurements: List[Dict], 
                            filepath: str,
                            title: str = "Fish Analysis Report") -> bool:
        """
        Generate HTML report with statistics.
        
        Args:
            measurements: List of measurement dictionaries
            filepath: Output HTML file path
            title: Report title
            
        Returns:
            True if successful
        """
        try:
            df = pd.DataFrame(measurements)
            
            stats_dict = {
                'Total Fish': len(df),
                'Avg Length (cm)': f"{df['length_cm'].mean():.2f}",
                'Avg Height (cm)': f"{df['height_cm'].mean():.2f}",
                'Avg Area (cm²)': f"{df['area_cm2'].mean():.2f}",
            }
            
            if 'predicted_weight_kg' in df.columns:
                stats_dict['Avg Weight (kg)'] = f"{df['predicted_weight_kg'].mean():.2f}"
                stats_dict['Total Weight (kg)'] = f"{df['predicted_weight_kg'].sum():.2f}"
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    h1 {{ color: #333; }}
                    .stats {{ background-color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .stats h2 {{ margin-top: 0; }}
                    .stat-item {{ display: inline-block; margin: 10px 20px; }}
                    .stat-label {{ font-weight: bold; }}
                    table {{ border-collapse: collapse; width: 100%; background-color: white; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="stats">
                    <h2>Summary Statistics</h2>
                    {''.join([f'<div class="stat-item"><span class="stat-label">{k}:</span> {v}</div>' 
                             for k, v in stats_dict.items()])}
                </div>
                <h2>Detailed Measurements</h2>
                {df.to_html(index=False)}
            </body>
            </html>
            """
            
            with open(filepath, 'w') as f:
                f.write(html)
            
            print(f"✅ HTML report generated: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
            return False
