"""
Visualization utilities for creating plots and charts
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import seaborn as sns


def create_evaluation_plots(results: Dict[str, Any], 
                           severities: List[float],
                           save_dir: Path):
    """
    Create comprehensive evaluation plots
    
    Args:
        results: Evaluation results dictionary
        severities: List of severity levels
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip if no results
    if not results:
        print("No results available for visualization")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create main comparison plot
    create_accuracy_vs_severity_plot(results, severities, save_dir)
    
    # Create 3FC vs original comparison if applicable
    create_3fc_comparison_plot(results, severities, save_dir)
    
    # Create robustness heatmap
    create_robustness_heatmap(results, severities, save_dir)
    
    # Create performance summary
    create_performance_summary(results, save_dir)


def create_accuracy_vs_severity_plot(results: Dict[str, Any], 
                                    severities: List[float],
                                    save_dir: Path):
    """Create accuracy vs severity plot for all models"""
    plt.figure(figsize=(14, 8))
    
    # Plot each model
    for name, data in results.items():
        model_results = data['results']
        sev_list = []
        acc_list = []
        
        for sev in sorted(severities):
            if sev in model_results:
                sev_list.append(sev)
                acc_list.append(model_results[sev])
        
        if len(sev_list) > 1:
            # Different line styles for different model types
            if '3fc' in name.lower():
                plt.plot(sev_list, acc_list, 'o-', linewidth=2.5, markersize=8, 
                        label=name, linestyle='--', alpha=0.9)
            elif 'robust' in name.lower():
                plt.plot(sev_list, acc_list, 'o-', linewidth=2.5, markersize=8, 
                        label=name, linestyle=':', alpha=0.9)
            else:
                plt.plot(sev_list, acc_list, 'o-', linewidth=2, markersize=6, 
                        label=name, alpha=0.8)
    
    plt.xlabel('Transformation Severity', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Performance vs Transformation Severity', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add annotations for best and worst performers if results exist
    clean_performances = [(name, data['results'].get(0.0, 0)) for name, data in results.items()]
    if clean_performances:
        best_clean = max(clean_performances, key=lambda x: x[1])
        plt.text(0.02, 0.95, f"Best Clean: {best_clean[0]} ({best_clean[1]:.3f})", 
                 transform=plt.gca().transAxes, fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    save_path = save_dir / "accuracy_vs_severity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy vs severity plot to {save_path}")


def create_3fc_comparison_plot(results: Dict[str, Any], 
                              severities: List[float],
                              save_dir: Path):
    """Create comparison plot between 3FC models and their originals"""
    # Check if we have 3FC models
    has_3fc = any('3fc' in name.lower() for name in results.keys())
    if not has_3fc:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Define comparison pairs
    comparisons = [
        ('TTT', 'TTT3fc', 'blue'),
        ('TTT_Robust', 'TTT3fc_Robust', 'darkblue'),
        ('BlendedTTT', 'BlendedTTT3fc', 'red'),
        ('BlendedTTT_Robust', 'BlendedTTT3fc_Robust', 'darkred')
    ]
    
    for orig_name, fc3_name, color in comparisons:
        # Find matching models in results
        orig_data = None
        fc3_data = None
        
        for name, data in results.items():
            if name == orig_name:
                orig_data = data
            elif name == fc3_name:
                fc3_data = data
        
        if orig_data and fc3_data:
            # Plot original
            orig_results = orig_data['results']
            sev_list = sorted([s for s in severities if s in orig_results])
            acc_list = [orig_results[s] for s in sev_list]
            
            plt.plot(sev_list, acc_list, 'o-', linewidth=2, markersize=6,
                    label=f"{orig_name} (Original)", color=color, alpha=0.6, linestyle=':')
            
            # Plot 3FC version
            fc3_results = fc3_data['results']
            sev_list = sorted([s for s in severities if s in fc3_results])
            acc_list = [fc3_results[s] for s in sev_list]
            
            plt.plot(sev_list, acc_list, 'o-', linewidth=2.5, markersize=8,
                    label=f"{fc3_name} (3FC)", color=color, alpha=0.9)
    
    plt.xlabel('Transformation Severity', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('3FC Models vs Original Models Comparison', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    save_path = save_dir / "3fc_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3FC comparison plot to {save_path}")


def create_robustness_heatmap(results: Dict[str, Any], 
                             severities: List[float],
                             save_dir: Path):
    """Create heatmap showing model robustness across severities"""
    # Prepare data for heatmap
    model_names = []
    accuracy_matrix = []
    
    # Sort models by clean accuracy
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['results'].get(0.0, 0),
        reverse=True
    )
    
    for name, data in sorted_models:
        model_names.append(name)
        model_results = data['results']
        
        acc_row = []
        for sev in severities:
            acc_row.append(model_results.get(sev, 0))
        accuracy_matrix.append(acc_row)
    
    # Create heatmap
    plt.figure(figsize=(10, max(8, len(model_names) * 0.4)))
    
    # Convert to numpy array
    accuracy_matrix = np.array(accuracy_matrix)
    
    # Create heatmap
    ax = sns.heatmap(accuracy_matrix, 
                     xticklabels=[f"S={s}" if s > 0 else "Clean" for s in severities],
                     yticklabels=model_names,
                     cmap='RdYlGn',
                     vmin=0, vmax=1,
                     annot=True,
                     fmt='.3f',
                     cbar_kws={'label': 'Accuracy'})
    
    plt.title('Model Robustness Heatmap', fontsize=16, pad=20)
    plt.xlabel('Transformation Severity', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    save_path = save_dir / "robustness_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness heatmap to {save_path}")


def create_performance_summary(results: Dict[str, Any], save_dir: Path):
    """Create performance summary bar chart"""
    # Calculate metrics
    metrics = {}
    
    for name, data in results.items():
        model_results = data['results']
        
        # Clean accuracy
        clean_acc = model_results.get(0.0, 0)
        
        # Average accuracy across all severities
        all_accs = list(model_results.values())
        avg_acc = np.mean(all_accs) if all_accs else 0
        
        # Robustness (smallest drop from clean to worst)
        if len(model_results) > 1 and 0.0 in model_results:
            worst_acc = min(v for k, v in model_results.items() if k > 0)
            robustness = worst_acc / clean_acc if clean_acc > 0 else 0
        else:
            robustness = 0
        
        metrics[name] = {
            'clean': clean_acc,
            'average': avg_acc,
            'robustness': robustness
        }
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sort by clean accuracy
    sorted_names = sorted(metrics.keys(), key=lambda x: metrics[x]['clean'], reverse=True)
    
    # Plot clean accuracy
    ax = axes[0]
    values = [metrics[name]['clean'] for name in sorted_names]
    colors = ['green' if '3fc' in name else 'blue' for name in sorted_names]
    bars = ax.bar(range(len(sorted_names)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Clean Data Performance')
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot average accuracy
    ax = axes[1]
    values = [metrics[name]['average'] for name in sorted_names]
    bars = ax.bar(range(len(sorted_names)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Average Performance Across All Severities')
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot robustness
    ax = axes[2]
    values = [metrics[name]['robustness'] for name in sorted_names]
    bars = ax.bar(range(len(sorted_names)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('Robustness Score')
    ax.set_title('Robustness (Worst/Clean Ratio)')
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Model Performance Summary', fontsize=16)
    plt.tight_layout()
    
    save_path = save_dir / "performance_summary.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance summary to {save_path}")