"""
Anti-Spoofing Test Report Generator
====================================

Generates comprehensive HTML and text reports from test sessions.
Includes metrics, visualizations, and detailed analysis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np


class TestReportGenerator:
    """Generate comprehensive test reports with visualizations."""
    
    def __init__(self, output_dir: str = "test_reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Session data
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_frames': 0,
            'real_count': 0,
            'spoof_count': 0,
            'attack_types': {},
            'metrics_history': [],
            'warnings': [],
            'recognition_results': [],
            'fps_history': [],
            'processing_times': []
        }
    
    def record_frame(
        self,
        is_real: bool,
        confidence: float,
        attack_type: str,
        metrics: Dict,
        warnings: List[str] = None,
        recognition_name: str = None,
        recognition_confidence: float = None,
        fps: float = None,
        processing_time: float = None
    ):
        """Record data from a single frame."""
        self.session_data['total_frames'] += 1
        
        if is_real:
            self.session_data['real_count'] += 1
        else:
            self.session_data['spoof_count'] += 1
            
            # Track attack types
            if attack_type not in self.session_data['attack_types']:
                self.session_data['attack_types'][attack_type] = 0
            self.session_data['attack_types'][attack_type] += 1
        
        # Store metrics
        self.session_data['metrics_history'].append({
            'frame': self.session_data['total_frames'],
            'is_real': is_real,
            'confidence': confidence,
            'attack_type': attack_type,
            **metrics
        })
        
        # Store warnings
        if warnings:
            for warning in warnings:
                if warning not in self.session_data['warnings']:
                    self.session_data['warnings'].append(warning)
        
        # Store recognition results
        if recognition_name:
            self.session_data['recognition_results'].append({
                'frame': self.session_data['total_frames'],
                'name': recognition_name,
                'confidence': recognition_confidence
            })
        
        # Store performance metrics
        if fps:
            self.session_data['fps_history'].append(fps)
        if processing_time:
            self.session_data['processing_times'].append(processing_time)
    
    def generate_report(
        self,
        report_name: Optional[str] = None,
        include_visualizations: bool = True
    ) -> Dict[str, str]:
        """
        Generate comprehensive test report.
        
        Args:
            report_name: Custom report name (default: timestamp)
            include_visualizations: Generate charts and graphs
            
        Returns:
            Dict with paths to generated files
        """
        self.session_data['end_time'] = datetime.now().isoformat()
        
        # Generate report name
        if not report_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"test_report_{timestamp}"
        
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)
        
        # Generate different report formats
        files = {}
        
        # 1. JSON report (raw data)
        json_path = report_dir / "data.json"
        with open(json_path, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        files['json'] = str(json_path)
        
        # 2. Text report (human readable)
        text_path = report_dir / "report.txt"
        self._generate_text_report(text_path)
        files['text'] = str(text_path)
        
        # 3. HTML report (interactive)
        html_path = report_dir / "report.html"
        self._generate_html_report(html_path, report_dir if include_visualizations else None)
        files['html'] = str(html_path)
        
        # 4. Visualizations
        if include_visualizations and len(self.session_data['metrics_history']) > 0:
            viz_dir = report_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            self._generate_visualizations(viz_dir)
            files['visualizations'] = str(viz_dir)
        
        print(f"\n{'='*70}")
        print(f"REPORT GENERATED: {report_name}")
        print(f"{'='*70}")
        print(f"Location: {report_dir}")
        print(f"Files:")
        for file_type, path in files.items():
            print(f"  - {file_type}: {path}")
        print(f"{'='*70}\n")
        
        return files
    
    def _generate_text_report(self, path: Path):
        """Generate text format report."""
        with open(path, 'w') as f:
            # Header
            f.write("="*70 + "\n")
            f.write("ANTI-SPOOFING TEST REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Session info
            start = datetime.fromisoformat(self.session_data['start_time'])
            end = datetime.fromisoformat(self.session_data['end_time'])
            duration = (end - start).total_seconds()
            
            f.write("SESSION INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Start Time:     {start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time:       {end.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration:       {duration:.1f} seconds ({duration/60:.1f} minutes)\n")
            f.write(f"Total Frames:   {self.session_data['total_frames']}\n\n")
            
            # Detection results
            f.write("DETECTION RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Real Detections:  {self.session_data['real_count']} ({self.session_data['real_count']/max(self.session_data['total_frames'],1)*100:.1f}%)\n")
            f.write(f"Spoof Detections: {self.session_data['spoof_count']} ({self.session_data['spoof_count']/max(self.session_data['total_frames'],1)*100:.1f}%)\n\n")
            
            # Attack types
            if self.session_data['attack_types']:
                f.write("ATTACK TYPES DETECTED\n")
                f.write("-" * 70 + "\n")
                for attack, count in sorted(self.session_data['attack_types'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {attack:30s} {count:5d} frames ({count/self.session_data['spoof_count']*100:.1f}%)\n")
                f.write("\n")
            
            # Metrics summary
            if self.session_data['metrics_history']:
                f.write("METRICS SUMMARY\n")
                f.write("-" * 70 + "\n")
                
                metrics_keys = ['texture_score', 'motion_score', 'color_score', 'depth_score', 
                               'frequency_score', 'blink_score', 'pulse_score', 
                               'color_temp_score', 'refresh_score', 'rppg_score']
                
                for key in metrics_keys:
                    values = [m.get(key, 0) for m in self.session_data['metrics_history']]
                    if values:
                        f.write(f"  {key:20s} Avg: {np.mean(values):.3f}  Min: {np.min(values):.3f}  Max: {np.max(values):.3f}\n")
                f.write("\n")
            
            # Performance
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n")
            if self.session_data['fps_history']:
                f.write(f"Average FPS:        {np.mean(self.session_data['fps_history']):.2f}\n")
            if self.session_data['processing_times']:
                f.write(f"Avg Processing:     {np.mean(self.session_data['processing_times']):.2f} ms\n")
                f.write(f"Min Processing:     {np.min(self.session_data['processing_times']):.2f} ms\n")
                f.write(f"Max Processing:     {np.max(self.session_data['processing_times']):.2f} ms\n")
            f.write("\n")
            
            # Warnings
            if self.session_data['warnings']:
                f.write("WARNINGS/ALERTS\n")
                f.write("-" * 70 + "\n")
                for warning in self.session_data['warnings']:
                    f.write(f"  - {warning}\n")
                f.write("\n")
            
            # Recognition results
            if self.session_data['recognition_results']:
                f.write("RECOGNITION RESULTS\n")
                f.write("-" * 70 + "\n")
                names = {}
                for result in self.session_data['recognition_results']:
                    name = result['name']
                    if name not in names:
                        names[name] = []
                    names[name].append(result['confidence'])
                
                for name, confidences in sorted(names.items()):
                    f.write(f"  {name:20s} {len(confidences):5d} frames  Avg Conf: {np.mean(confidences):.2f}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
    
    def _generate_html_report(self, path: Path, viz_dir: Optional[Path] = None):
        """Generate HTML format report with embedded visualizations."""
        start = datetime.fromisoformat(self.session_data['start_time'])
        end = datetime.fromisoformat(self.session_data['end_time'])
        duration = (end - start).total_seconds()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anti-Spoofing Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .metric-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .metric-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        .metric-table tr:hover {{
            background: #f5f5f5;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }}
        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .success-box {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .error-box {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Anti-Spoofing Test Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
        
        <div class="content">
            <!-- Session Info -->
            <div class="section">
                <h2>📊 Session Information</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Duration</h3>
                        <div class="value">{duration:.1f}s</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Frames</h3>
                        <div class="value">{self.session_data['total_frames']}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Real Detections</h3>
                        <div class="value">{self.session_data['real_count']}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Spoof Detections</h3>
                        <div class="value">{self.session_data['spoof_count']}</div>
                    </div>
                </div>
            </div>
            
            <!-- Detection Results -->
            <div class="section">
                <h2>🎯 Detection Results</h2>
                <div class="success-box">
                    <strong>Real Face Detection Rate:</strong> {self.session_data['real_count']/max(self.session_data['total_frames'],1)*100:.1f}%
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {self.session_data['real_count']/max(self.session_data['total_frames'],1)*100}%;"></div>
                    </div>
                </div>
                <div class="error-box">
                    <strong>Spoof Detection Rate:</strong> {self.session_data['spoof_count']/max(self.session_data['total_frames'],1)*100:.1f}%
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {self.session_data['spoof_count']/max(self.session_data['total_frames'],1)*100}%; background: linear-gradient(90deg, #dc3545 0%, #c82333 100%);"></div>
                    </div>
                </div>
            </div>
"""
        
        # Attack types
        if self.session_data['attack_types']:
            html += """
            <div class="section">
                <h2>⚠️ Attack Types Detected</h2>
                <table class="metric-table">
                    <tr>
                        <th>Attack Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
"""
            for attack, count in sorted(self.session_data['attack_types'].items(), key=lambda x: x[1], reverse=True):
                percentage = count/self.session_data['spoof_count']*100 if self.session_data['spoof_count'] > 0 else 0
                html += f"""
                    <tr>
                        <td>{attack}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
"""
            html += """
                </table>
            </div>
"""
        
        # Metrics summary
        if self.session_data['metrics_history']:
            html += """
            <div class="section">
                <h2>📈 Metrics Summary</h2>
                <table class="metric-table">
                    <tr>
                        <th>Metric</th>
                        <th>Average</th>
                        <th>Minimum</th>
                        <th>Maximum</th>
                        <th>Std Dev</th>
                    </tr>
"""
            metrics_keys = ['texture_score', 'motion_score', 'color_score', 'depth_score', 
                           'frequency_score', 'blink_score', 'pulse_score', 
                           'color_temp_score', 'refresh_score', 'rppg_score']
            
            for key in metrics_keys:
                values = [m.get(key, 0) for m in self.session_data['metrics_history']]
                if values:
                    html += f"""
                    <tr>
                        <td>{key.replace('_', ' ').title()}</td>
                        <td>{np.mean(values):.3f}</td>
                        <td>{np.min(values):.3f}</td>
                        <td>{np.max(values):.3f}</td>
                        <td>{np.std(values):.3f}</td>
                    </tr>
"""
            html += """
                </table>
            </div>
"""
        
        # Performance
        html += """
            <div class="section">
                <h2>⚡ Performance Metrics</h2>
                <div class="stats-grid">
"""
        if self.session_data['fps_history']:
            avg_fps = np.mean(self.session_data['fps_history'])
            html += f"""
                    <div class="stat-card">
                        <h3>Average FPS</h3>
                        <div class="value">{avg_fps:.2f}</div>
                    </div>
"""
        if self.session_data['processing_times']:
            avg_time = np.mean(self.session_data['processing_times'])
            html += f"""
                    <div class="stat-card">
                        <h3>Avg Processing Time</h3>
                        <div class="value">{avg_time:.2f}ms</div>
                    </div>
"""
        html += """
                </div>
            </div>
"""
        
        # Visualizations
        if viz_dir:
            html += """
            <div class="section">
                <h2>📊 Visualizations</h2>
                <div class="visualization">
                    <img src="visualizations/metrics_over_time.png" alt="Metrics Over Time">
                </div>
                <div class="visualization">
                    <img src="visualizations/detection_distribution.png" alt="Detection Distribution">
                </div>
                <div class="visualization">
                    <img src="visualizations/metrics_heatmap.png" alt="Metrics Heatmap">
                </div>
            </div>
"""
        
        # Warnings
        if self.session_data['warnings']:
            html += """
            <div class="section">
                <h2>⚠️ Warnings & Alerts</h2>
"""
            for warning in self.session_data['warnings']:
                html += f"""
                <div class="warning-box">
                    {warning}
                </div>
"""
            html += """
            </div>
"""
        
        html += """
        </div>
        
        <div class="footer">
            <p>Generated by Anti-Spoofing Test Report Generator</p>
            <p>Ultimate Face Liveness Detection System © 2025</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(path, 'w') as f:
            f.write(html)
    
    def _generate_visualizations(self, viz_dir: Path):
        """Generate visualization charts."""
        if not self.session_data['metrics_history']:
            return
        
        # 1. Metrics over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Anti-Spoofing Metrics Over Time', fontsize=16, fontweight='bold')
        
        frames = [m['frame'] for m in self.session_data['metrics_history']]
        
        # Plot primary metrics
        metrics_to_plot = [
            (['texture_score', 'motion_score', 'color_score'], 'Primary Detection Metrics'),
            (['depth_score', 'frequency_score', 'blink_score'], 'Secondary Detection Metrics'),
            (['color_temp_score', 'refresh_score', 'rppg_score'], 'Advanced Detection Metrics'),
            (['confidence'], 'Overall Confidence')
        ]
        
        for idx, (metrics, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            for metric in metrics:
                values = [m.get(metric, 0) for m in self.session_data['metrics_history']]
                ax.plot(frames, values, label=metric.replace('_', ' ').title(), linewidth=2)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'metrics_over_time.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Detection distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Detection Distribution', fontsize=16, fontweight='bold')
        
        # Pie chart
        labels = ['Real', 'Spoof']
        sizes = [self.session_data['real_count'], self.session_data['spoof_count']]
        colors = ['#28a745', '#dc3545']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Real vs Spoof Detection')
        
        # Bar chart for attack types
        if self.session_data['attack_types']:
            attacks = list(self.session_data['attack_types'].keys())
            counts = list(self.session_data['attack_types'].values())
            ax2.barh(attacks, counts, color='#dc3545')
            ax2.set_xlabel('Count')
            ax2.set_title('Attack Types Detected')
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'detection_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Metrics heatmap
        metrics_keys = ['texture_score', 'motion_score', 'color_score', 'depth_score', 
                       'frequency_score', 'color_temp_score', 'refresh_score', 'rppg_score']
        
        # Sample every 10 frames if too many
        step = max(1, len(frames) // 50)
        sampled_frames = frames[::step]
        
        heatmap_data = []
        for metric in metrics_keys:
            values = [m.get(metric, 0) for m in self.session_data['metrics_history']][::step]
            heatmap_data.append(values)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        
        ax.set_yticks(range(len(metrics_keys)))
        ax.set_yticklabels([k.replace('_', ' ').title() for k in metrics_keys])
        ax.set_xlabel('Frame')
        ax.set_title('Metrics Heatmap Over Time', fontsize=16, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Score')
        plt.tight_layout()
        plt.savefig(viz_dir / 'metrics_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Report Generator...")
    
    # Create generator
    generator = TestReportGenerator()
    
    # Simulate some test data
    for i in range(100):
        is_real = i % 3 != 0  # 2/3 real, 1/3 spoof
        generator.record_frame(
            is_real=is_real,
            confidence=0.8 if is_real else 0.3,
            attack_type="None" if is_real else "Printed Photo",
            metrics={
                'texture_score': 0.7 + np.random.random() * 0.2 if is_real else 0.2 + np.random.random() * 0.1,
                'motion_score': 0.6 + np.random.random() * 0.3,
                'color_score': 0.75 if is_real else 0.1,
                'depth_score': 0.8 if is_real else 0.4,
                'frequency_score': 0.7 if is_real else 0.3,
                'blink_score': 0.6 if is_real else 0.3,
                'pulse_score': 0.5 if is_real else 0.2,
                'color_temp_score': 0.7 if is_real else 0.3,
                'refresh_score': 0.9 if is_real else 0.2,
                'rppg_score': 0.8 if is_real else 0.1,
            },
            fps=25.0,
            processing_time=35.5
        )
    
    # Generate report
    files = generator.generate_report(report_name="demo_test")
    print(f"\nDemo report generated successfully!")
