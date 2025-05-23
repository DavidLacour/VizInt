#!/usr/bin/env python3
"""
Training monitor and progress tracker for the automatic training system.
Provides real-time monitoring, model comparison, and training analytics.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from collections import defaultdict
import os

class TrainingMonitor:
    """Monitor and analyze training progress"""
    
    def __init__(self, results_dir: str = "training_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Model directories pattern
        self.model_patterns = {
            'classification': 'bestmodel_*_classification',
            'healer': 'bestmodel_*_healer',
            'ttt': 'bestmodel_*_ttt',
            'blended_ttt': 'bestmodel_*_blended_ttt'
        }
    
    def scan_existing_models(self) -> Dict[str, Dict[str, Any]]:
        """Scan for existing trained models"""
        models = {}
        
        for model_type, pattern in self.model_patterns.items():
            for model_dir in Path('.').glob(pattern):
                best_model_path = model_dir / 'best_model.pt'
                if best_model_path.exists():
                    # Extract backbone name from directory name
                    dir_name = model_dir.name
                    backbone_name = dir_name.replace(f'bestmodel_', '').replace(f'_{model_type}', '')
                    
                    # Get model info
                    try:
                        checkpoint = torch.load(best_model_path, map_location='cpu')
                        
                        model_info = {
                            'model_type': model_type,
                            'backbone': backbone_name,
                            'path': str(best_model_path),
                            'size_mb': best_model_path.stat().st_size / (1024 * 1024),
                            'modified_time': datetime.fromtimestamp(best_model_path.stat().st_mtime),
                            'epoch': checkpoint.get('epoch', 'Unknown'),
                            'val_acc': checkpoint.get('val_acc', None),
                            'val_loss': checkpoint.get('val_loss', None)
                        }
                        
                        model_key = f"{model_type}_{backbone_name}"
                        models[model_key] = model_info
                    except Exception as e:
                        print(f"Error loading {best_model_path}: {e}")
        
        return models
    
    def get_training_logs(self) -> List[Dict[str, Any]]:
        """Get all training log files"""
        log_files = list(self.results_dir.glob("training_log_*.json"))
        logs = []
        
        for log_file in sorted(log_files):
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    log_data['log_file'] = str(log_file)
                    logs.append(log_data)
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        return logs
    
    def print_model_summary(self):
        """Print summary of all existing models"""
        models = self.scan_existing_models()
        
        if not models:
            print("📭 No trained models found.")
            return
        
        print(f"📊 Model Summary ({len(models)} models found)")
        print("=" * 80)
        
        # Group by model type
        by_type = defaultdict(list)
        for model_key, info in models.items():
            by_type[info['model_type']].append(info)
        
        for model_type, model_list in by_type.items():
            print(f"\n🤖 {model_type.upper()} Models ({len(model_list)}):")
            print("   " + "-" * 65)
            print("   Backbone        | Epoch | Val Acc | Val Loss | Size (MB) | Modified")
            print("   " + "-" * 65)
            
            for info in sorted(model_list, key=lambda x: x['backbone']):
                backbone = info['backbone'][:14]
                epoch = str(info['epoch'])[:5]
                val_acc = f"{info['val_acc']:.4f}" if info['val_acc'] else "N/A"
                val_loss = f"{info['val_loss']:.4f}" if info['val_loss'] else "N/A"
                size_mb = f"{info['size_mb']:.1f}"
                modified = info['modified_time'].strftime("%m/%d %H:%M")
                
                print(f"   {backbone:<15} | {epoch:<5} | {val_acc:<7} | {val_loss:<8} | {size_mb:<8} | {modified}")
    
    def print_training_progress(self):
        """Print recent training progress"""
        logs = self.get_training_logs()
        
        if not logs:
            print("📭 No training logs found.")
            return
        
        print(f"📈 Recent Training Progress")
        print("=" * 60)
        
        # Get the most recent log
        recent_log = logs[-1]
        
        print(f"Latest training session: {recent_log.get('timestamp', 'Unknown')}")
        print(f"Dataset: {recent_log.get('dataset_path', 'Unknown')}")
        print(f"Force mode: {recent_log.get('force_mode', False)}")
        
        # Training progress
        training_log = recent_log.get('training_log', [])
        if training_log:
            completed = len(training_log)
            total = training_log[-1].get('total_steps', completed) if training_log else 0
            success_count = sum(1 for entry in training_log if entry.get('success', False))
            
            print(f"\nProgress: {completed}/{total} models")
            print(f"Success rate: {success_count}/{completed} ({success_count/completed*100:.1f}%)")
            
            # Recent completions
            print(f"\n📋 Recent Completions:")
            for entry in training_log[-5:]:
                status = "✅" if entry.get('success', False) else "❌"
                model_type = entry.get('model_type', 'Unknown')
                backbone = entry.get('backbone', 'Unknown')
                timestamp = entry.get('timestamp', '')[:16].replace('T', ' ')
                print(f"  {status} {model_type:<12} + {backbone:<12} ({timestamp})")
        
        # Failed models
        failed_models = recent_log.get('failed_models', {})
        if failed_models:
            print(f"\n❌ Failed Models ({len(failed_models)}):")
            for model_key, info in failed_models.items():
                error = info.get('error', 'Unknown error')[:50]
                print(f"  • {info['model_type']:<12} + {info['backbone']:<12} - {error}...")
    
    def generate_comparison_report(self, output_file: str = None):
        """Generate a comprehensive comparison report"""
        models = self.scan_existing_models()
        
        if not models:
            print("No models to compare.")
            return
        
        # Create comparison data
        comparison_data = []
        for model_key, info in models.items():
            if info['val_acc'] is not None:
                comparison_data.append({
                    'Model': info['model_type'],
                    'Backbone': info['backbone'],
                    'Validation Accuracy': info['val_acc'],
                    'Validation Loss': info['val_loss'] if info['val_loss'] else 0,
                    'Model Size (MB)': info['size_mb'],
                    'Training Epoch': info['epoch']
                })
        
        if not comparison_data:
            print("No models with validation metrics found.")
            return
        
        df = pd.DataFrame(comparison_data)
        
        # Generate report
        report = []
        report.append("# Model Comparison Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total models: {len(comparison_data)}")
        report.append("")
        
        # Best models by type
        report.append("## Best Models by Type")
        for model_type in df['Model'].unique():
            model_df = df[df['Model'] == model_type]
            best_model = model_df.loc[model_df['Validation Accuracy'].idxmax()]
            
            report.append(f"### {model_type.upper()}")
            report.append(f"- **Best**: {best_model['Backbone']} (Acc: {best_model['Validation Accuracy']:.4f})")
            report.append(f"- **Size**: {best_model['Model Size (MB)']:.1f} MB")
            report.append(f"- **Epochs**: {best_model['Training Epoch']}")
            report.append("")
        
        # Backbone comparison
        report.append("## Backbone Performance")
        backbone_stats = df.groupby('Backbone').agg({
            'Validation Accuracy': ['mean', 'std', 'max'],
            'Model Size (MB)': 'mean'
        }).round(4)
        
        report.append("| Backbone | Avg Acc | Std Acc | Max Acc | Avg Size (MB) |")
        report.append("|----------|---------|---------|---------|---------------|")
        
        for backbone in backbone_stats.index:
            stats = backbone_stats.loc[backbone]
            avg_acc = stats[('Validation Accuracy', 'mean')]
            std_acc = stats[('Validation Accuracy', 'std')]
            max_acc = stats[('Validation Accuracy', 'max')]
            avg_size = stats[('Model Size (MB)', 'mean')]
            
            report.append(f"| {backbone} | {avg_acc:.4f} | {std_acc:.4f} | {max_acc:.4f} | {avg_size:.1f} |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("| Model | Backbone | Val Acc | Val Loss | Size (MB) | Epochs |")
        report.append("|-------|----------|---------|----------|-----------|--------|")
        
        for _, row in df.sort_values('Validation Accuracy', ascending=False).iterrows():
            report.append(f"| {row['Model']} | {row['Backbone']} | {row['Validation Accuracy']:.4f} | {row['Validation Loss']:.4f} | {row['Model Size (MB)']:.1f} | {row['Training Epoch']} |")
        
        # Save report
        if output_file is None:
            output_file = self.results_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"📊 Comparison report saved to: {output_file}")
        
        # Print summary to console
        print("\n📊 Model Performance Summary:")
        print("=" * 50)
        best_overall = df.loc[df['Validation Accuracy'].idxmax()]
        print(f"🏆 Best Overall: {best_overall['Model']} + {best_overall['Backbone']}")
        print(f"   Accuracy: {best_overall['Validation Accuracy']:.4f}")
        print(f"   Size: {best_overall['Model Size (MB)']:.1f} MB")
        
        return df
    
    def plot_performance_comparison(self, output_dir: str = None):
        """Create performance comparison plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting. Install with: pip install matplotlib seaborn")
            return
        
        models = self.scan_existing_models()
        
        # Prepare data
        plot_data = []
        for model_key, info in models.items():
            if info['val_acc'] is not None:
                plot_data.append({
                    'Model Type': info['model_type'],
                    'Backbone': info['backbone'],
                    'Validation Accuracy': info['val_acc'],
                    'Model Size (MB)': info['size_mb']
                })
        
        if not plot_data:
            print("No data available for plotting.")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # 1. Accuracy by Model Type
        sns.boxplot(data=df, x='Model Type', y='Validation Accuracy', ax=ax1)
        ax1.set_title('Accuracy Distribution by Model Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Accuracy by Backbone
        sns.boxplot(data=df, x='Backbone', y='Validation Accuracy', ax=ax2)
        ax2.set_title('Accuracy Distribution by Backbone')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Accuracy vs Model Size
        sns.scatterplot(data=df, x='Model Size (MB)', y='Validation Accuracy', 
                       hue='Model Type', style='Backbone', ax=ax3)
        ax3.set_title('Accuracy vs Model Size Trade-off')
        
        # 4. Performance Heatmap
        pivot_df = df.pivot_table(values='Validation Accuracy', 
                                 index='Model Type', 
                                 columns='Backbone', 
                                 aggfunc='mean')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
        ax4.set_title('Performance Heatmap (Model Type vs Backbone)')
        
        plt.tight_layout()
        
        # Save plot
        if output_dir is None:
            output_dir = self.results_dir
        
        plot_file = Path(output_dir) / f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance plots saved to: {plot_file}")
    
    def watch_training(self, interval: int = 30):
        """Watch training progress in real-time"""
        print("👀 Watching training progress... (Ctrl+C to stop)")
        print(f"Refresh interval: {interval} seconds")
        print("=" * 60)
        
        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print(f"🕒 Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                
                # Show current progress
                self.print_training_progress()
                print("\n" + "=" * 60)
                print(f"Next update in {interval} seconds... (Ctrl+C to stop)")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Monitor and analyze training progress',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show model summary
  python training_monitor.py --summary
  
  # Show training progress
  python training_monitor.py --progress
  
  # Generate comparison report
  python training_monitor.py --compare
  
  # Create performance plots
  python training_monitor.py --plot
  
  # Watch training in real-time
  python training_monitor.py --watch
  
  # Full analysis
  python training_monitor.py --all
        """
    )
    
    parser.add_argument('--summary', action='store_true',
                        help='Show summary of existing models')
    
    parser.add_argument('--progress', action='store_true',
                        help='Show recent training progress')
    
    parser.add_argument('--compare', action='store_true',
                        help='Generate model comparison report')
    
    parser.add_argument('--plot', action='store_true',
                        help='Create performance comparison plots')
    
    parser.add_argument('--watch', action='store_true',
                        help='Watch training progress in real-time')
    
    parser.add_argument('--all', action='store_true',
                        help='Run all analysis (summary, progress, compare, plot)')
    
    parser.add_argument('--interval', type=int, default=30,
                        help='Watch mode refresh interval in seconds')
    
    parser.add_argument('--output_dir', type=str, default="training_results",
                        help='Output directory for reports and plots')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = TrainingMonitor(args.output_dir)
    
    # Execute requested operations
    if args.all:
        print("🔍 Running complete analysis...\n")
        monitor.print_model_summary()
        print("\n")
        monitor.print_training_progress()
        print("\n")
        monitor.generate_comparison_report()
        print("\n")
        monitor.plot_performance_comparison(args.output_dir)
    
    elif args.summary:
        monitor.print_model_summary()
    
    elif args.progress:
        monitor.print_training_progress()
    
    elif args.compare:
        monitor.generate_comparison_report()
    
    elif args.plot:
        monitor.plot_performance_comparison(args.output_dir)
    
    elif args.watch:
        monitor.watch_training(args.interval)
    
    else:
        # Default: show summary and progress
        monitor.print_model_summary()
        print("\n")
        monitor.print_training_progress()

if __name__ == "__main__":
    main()