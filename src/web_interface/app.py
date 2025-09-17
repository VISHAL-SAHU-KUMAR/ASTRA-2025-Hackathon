"""
LunarVision AI - Web Interface
============================

This module provides a web interface for the ice detection system.
"""

from flask import Flask

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """
    Main page with image upload functionality
    """
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LunarVision AI - Ice Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #1a2a6c, #2a4b8d, #4682b4); color: white; }
            .container { max-width: 800px; margin: 0 auto; background: rgba(255, 255, 255, 0.1); padding: 30px; border-radius: 10px; }
            h1 { color: #fff; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LunarVision AI - Lunar and Martian Ice Detection</h1>
            <p style="text-align: center;">Advanced AI System for Detecting Water Ice on Planetary Surfaces</p>
        </div>
    </body>
    </html>
    '''

def main():
    """
    Main function to run the web interface
    """
    print("LunarVision AI - Web Interface Module")
    print("=" * 35)
    print("Starting web server...")
    print("Visit http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()