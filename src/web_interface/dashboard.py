"""
LunarVision AI - Advanced Dashboard
================================

This module provides an advanced dashboard with visualization capabilities.
"""

from flask import Flask, render_template_string
import os
import base64
import io

# Create Flask app
app = Flask(__name__)

# Sample data for visualization
sample_data = {
    'ice_probability': 78,
    'confidence_score': 85,
    'regions': 3,
    'coordinates': [(120, 240), (300, 400), (450, 180)]
}

@app.route('/')
def dashboard():
    """
    Main dashboard page with visualization
    """
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LunarVision AI - Advanced Dashboard</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --primary: #4682b4;
                --secondary: #5a9bd4;
                --dark: #1a3c5e;
                --light: #f0f8ff;
                --success: #4caf50;
                --warning: #ff9800;
                --danger: #f44336;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #0c1a33, #1a3c5e, #2a4b8d);
                color: #fff;
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            header {
                text-align: center;
                padding: 20px 0;
                margin-bottom: 30px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            
            h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #4682b4, #ffffff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                max-width: 800px;
                margin: 0 auto;
            }
            
            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .card {
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
                background: rgba(255, 255, 255, 0.12);
            }
            
            .metric-card {
                text-align: center;
            }
            
            .metric-value {
                font-size: 3rem;
                font-weight: 700;
                margin: 15px 0;
            }
            
            .high-confidence {
                color: var(--success);
                text-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
            }
            
            .medium-confidence {
                color: var(--warning);
                text-shadow: 0 0 10px rgba(255, 152, 0, 0.5);
            }
            
            .low-confidence {
                color: var(--danger);
                text-shadow: 0 0 10px rgba(244, 67, 54, 0.5);
            }
            
            .chart-container {
                height: 300px;
                margin: 20px 0;
            }
            
            .visualization {
                text-align: center;
                margin: 20px 0;
            }
            
            .visualization img {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }
            
            .btn {
                background: linear-gradient(45deg, var(--primary), var(--secondary));
                color: white;
                border: none;
                padding: 12px 25px;
                font-size: 1rem;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 600;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                text-decoration: none;
                display: inline-block;
                margin: 10px 5px;
            }
            
            .btn:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            }
            
            .btn-secondary {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .controls {
                text-align: center;
                margin: 30px 0;
            }
            
            footer {
                text-align: center;
                padding: 20px;
                margin-top: 30px;
                opacity: 0.7;
                font-size: 0.9rem;
            }
            
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            
            .feature-item {
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            
            .feature-icon {
                font-size: 2rem;
                margin-bottom: 10px;
                color: var(--secondary);
            }
            
            @media (max-width: 768px) {
                .dashboard-grid {
                    grid-template-columns: 1fr;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .metric-value {
                    font-size: 2.5rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1><i class="fas fa-satellite"></i> LunarVision AI Dashboard</h1>
                <p class="subtitle">Advanced AI System for Detecting Water Ice on Lunar and Martian Surfaces</p>
            </header>
            
            <div class="dashboard-grid">
                <div class="card metric-card">
                    <h3><i class="fas fa-icicles"></i> Ice Probability</h3>
                    <div class="metric-value high-confidence">{{ ice_probability }}%</div>
                    <p>Likelihood of water ice presence</p>
                </div>
                
                <div class="card metric-card">
                    <h3><i class="fas fa-shield-alt"></i> Confidence Score</h3>
                    <div class="metric-value high-confidence">{{ confidence_score }}%</div>
                    <p>Model prediction confidence</p>
                </div>
                
                <div class="card metric-card">
                    <h3><i class="fas fa-map-marker-alt"></i> Ice Regions</h3>
                    <div class="metric-value medium-confidence">{{ regions }}</div>
                    <p>Potential ice deposits detected</p>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="card">
                    <h2><i class="fas fa-chart-pie"></i> Confidence Distribution</h2>
                    <div class="chart-container">
                        <canvas id="confidenceChart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h2><i class="fas fa-wave-square"></i> Detection Accuracy</h2>
                    <div class="chart-container">
                        <canvas id="accuracyChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2><i class="fas fa-eye"></i> Ice Detection Visualization</h2>
                <div class="visualization">
                    <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMGEyMzQ0Ii8+PHJlY3QgeD0iMTAwIiB5PSIxMDAiIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiBmaWxsPSIjNGNhZjUwIiBvcGFjaXR5PSIwLjgiLz48cmVjdCB4PSIzMDAiIHk9IjE1MCIgd2lkdGg9IjE1MCIgaGVpZ2h0PSIxNTAiIGZpbGw9IiM0Y2FmNTAiIG9wYWNpdHk9IjAuNiIvPjxyZWN0IHg9IjUwMCIgeT0iODAiIHdpZHRoPSIxMjAiIGhlaWdodD0iMTIwIiBmaWxsPSIjNGNhZjUwIiBvcGFjaXR5PSIwLjciLz48dGV4dCB4PSI0MDAiIHk9IjIwMCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE4IiBmaWxsPSIjZmZmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+U2F0ZWxsaXRlIEltYWdlIFZpc3VhbGl6YXRpb248L3RleHQ+PC9zdmc+" alt="Ice Detection Heatmap">
                    <p style="margin-top: 15px; opacity: 0.8;">Heatmap showing ice detection confidence levels across the surface</p>
                </div>
            </div>
            
            <div class="card">
                <h2><i class="fas fa-cogs"></i> System Features</h2>
                <div class="feature-grid">
                    <div class="feature-item">
                        <div class="feature-icon">
                            <i class="fas fa-microscope"></i>
                        </div>
                        <h3>Advanced Analysis</h3>
                        <p>Computer vision and machine learning algorithms</p>
                    </div>
                    
                    <div class="feature-item">
                        <div class="feature-icon">
                            <i class="fas fa-brain"></i>
                        </div>
                        <h3>AI-Powered</h3>
                        <p>Deep learning models trained on planetary data</p>
                    </div>
                    
                    <div class="feature-item">
                        <div class="feature-icon">
                            <i class="fas fa-bolt"></i>
                        </div>
                        <h3>Real-time Results</h3>
                        <p>Instant detection and confidence scoring</p>
                    </div>
                    
                    <div class="feature-item">
                        <div class="feature-icon">
                            <i class="fas fa-file-alt"></i>
                        </div>
                        <h3>Report Generation</h3>
                        <p>Automated PDF reports with analysis</p>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <a href="#" class="btn">
                    <i class="fas fa-cloud-upload-alt"></i> Upload New Image
                </a>
                <a href="#" class="btn btn-secondary">
                    <i class="fas fa-file-pdf"></i> Generate Report
                </a>
                <a href="#" class="btn btn-secondary">
                    <i class="fas fa-sync-alt"></i> Refresh Data
                </a>
            </div>
            
            <footer>
                <p>LunarVision AI &copy; 2025 | Advanced Planetary Ice Detection System</p>
                <p>NASA Space Apps Challenge 2025</p>
            </footer>
        </div>
        
        <script>
            // Confidence distribution chart
            const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
            const confidenceChart = new Chart(confidenceCtx, {
                type: 'doughnut',
                data: {
                    labels: ['High Confidence', 'Medium Confidence', 'Low Confidence'],
                    datasets: [{
                        data: [65, 25, 10],
                        backgroundColor: [
                            'rgba(76, 175, 80, 0.8)',
                            'rgba(255, 152, 0, 0.8)',
                            'rgba(244, 67, 54, 0.8)'
                        ],
                        borderColor: [
                            'rgba(76, 175, 80, 1)',
                            'rgba(255, 152, 0, 1)',
                            'rgba(244, 67, 54, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#fff',
                                font: {
                                    size: 12
                                }
                            }
                        }
                    }
                }
            });
            
            // Accuracy chart
            const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
            const accuracyChart = new Chart(accuracyCtx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                    datasets: [{
                        label: 'Detection Accuracy',
                        data: [72, 75, 78, 82, 85, 85],
                        borderColor: 'rgba(70, 130, 180, 1)',
                        backgroundColor: 'rgba(70, 130, 180, 0.2)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: 70,
                            max: 90,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#fff'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#fff'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#fff'
                            }
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    ''', **sample_data)

def main():
    """
    Main function to run the dashboard
    """
    print("LunarVision AI - Advanced Dashboard")
    print("=" * 35)
    print("Starting dashboard server...")
    print("Visit http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()