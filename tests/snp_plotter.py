import sys
import argparse
import skrf as rf
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots
from flask import Flask, render_template_string, request, jsonify

class SNPPlotter:
    def __init__(self, snp_file_path):
        """Initialize SNP plotter with S-parameter file path."""
        self.snp_file_path = Path(snp_file_path)
        self.network = None
        self.load_network()
    
    def load_network(self):
        """Load S-parameter network using skrf."""
        try:
            self.network = rf.Network(str(self.snp_file_path))
            print(f"Loaded network: {self.snp_file_path}")
            
            # Debug frequency data
            if len(self.network.f) > 0:
                print(f"Frequency range: {self.network.f[0]/1e9:.3f} - {self.network.f[-1]/1e9:.3f} GHz")
                print(f"Raw frequency range: {self.network.f[0]} - {self.network.f[-1]} Hz")
            else:
                print("Warning: No frequency points found in file")
                
            print(f"Number of ports: {self.network.nports}")
            print(f"Number of frequency points: {len(self.network.f)}")
            print(f"S-parameter matrix shape: {self.network.s.shape}")
            
        except Exception as e:
            print(f"Error loading network: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def get_sparam_data(self, p1, p2):
        """Get S-parameter data for specific port combination."""
        if self.network is None:
            print("Error: Network not loaded")
            return None
        
        try:
            # Convert to 0-based indexing
            i, j = p1 - 1, p2 - 1
            print(f"Converting S{p1}{p2} to indices [{i},{j}]")  # Debug
            
            if i < 0 or i >= self.network.nports or j < 0 or j >= self.network.nports:
                print(f"Invalid port indices: [{i},{j}] for {self.network.nports}-port network")
                return None
            
            # Check if frequency data exists
            if len(self.network.f) == 0:
                print("Error: No frequency data in network")
                return None
            
            s_param = self.network.s[:, i, j]
            print(f"S-parameter shape: {s_param.shape}")  # Debug
            
            freq_ghz = self.network.f / 1e9
            
            # Check for valid S-parameter data
            if np.all(s_param == 0):
                print(f"Warning: S{p1}{p2} appears to be all zeros")
            
            s_magnitude = np.abs(s_param)
            # Avoid log(0) by setting minimum value
            s_magnitude = np.maximum(s_magnitude, 1e-20)
            s_db = 20 * np.log10(s_magnitude)
            s_phase = np.angle(s_param, deg=True)
            
            print(f"Magnitude range: {s_db.min():.2f} to {s_db.max():.2f} dB")  # Debug
            print(f"Phase range: {s_phase.min():.2f} to {s_phase.max():.2f} degrees")  # Debug
            
            return {
                'frequency': freq_ghz.tolist(),
                'magnitude_db': s_db.tolist(),
                'phase_deg': s_phase.tolist()
            }
        except Exception as e:
            print(f"Error getting S{p1}{p2} data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_interactive_plot(self, output_file=None):
        """Create interactive plotly plot with scrollable S-parameter selection."""
        if self.network is None:
            print("No network loaded")
            return
        
        nports = self.network.nports
        
        # For large files, recommend using server mode
        if nports > 20:
            print(f"Warning: {nports}-port file detected. Consider using --server mode for better performance.")
            response = input("Continue with static plot? (y/N): ")
            if response.lower() != 'y':
                print("Use --server flag for interactive server mode")
                return
        
        freq_ghz = self.network.f / 1e9
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]],
            subplot_titles=[f"S-Parameter Frequency Response - {self.snp_file_path.name}"]
        )
        
        # Create traces for all S-parameter combinations
        traces = []
        for i in range(nports):
            for j in range(nports):
                s_param = self.network.s[:, i, j]
                s_db = 20 * np.log10(np.abs(s_param))
                s_phase = np.angle(s_param, deg=True)
                
                # Magnitude trace
                mag_trace = go.Scatter(
                    x=freq_ghz,
                    y=s_db,
                    mode='lines',
                    name=f'|S{i+1}{j+1}| (dB)',
                    visible=False,
                    line=dict(width=2)
                )
                
                # Phase trace
                phase_trace = go.Scatter(
                    x=freq_ghz,
                    y=s_phase,
                    mode='lines',
                    name=f'∠S{i+1}{j+1} (°)',
                    visible=False,
                    line=dict(width=2, dash='dash'),
                    yaxis='y2'
                )
                
                traces.append((mag_trace, phase_trace, i+1, j+1))
        
        # Make first S-parameter visible by default
        if traces:
            traces[0][0].visible = True
            traces[0][1].visible = True
        
        # Add all traces to figure
        for mag_trace, phase_trace, _, _ in traces:
            fig.add_trace(mag_trace)
            fig.add_trace(phase_trace, secondary_y=True)
        
        # Create dropdown menu for S-parameter selection
        buttons = []
        for idx, (mag_trace, phase_trace, p1, p2) in enumerate(traces):
            visibility = [False] * len(traces) * 2
            visibility[idx * 2] = True  # magnitude
            visibility[idx * 2 + 1] = True  # phase
            
            buttons.append(dict(
                label=f"S{p1}{p2}",
                method="update",
                args=[{"visible": visibility},
                      {"title": f"S{p1}{p2} Frequency Response - {self.snp_file_path.name}"}]
            ))
        
        # Update layout
        fig.update_layout(
            title=f"S-Parameter Frequency Response - {self.snp_file_path.name}",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Magnitude (dB)",
            width=1000,
            height=600,
            hovermode='x unified',
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top",
                    buttons=buttons
                )
            ],
            annotations=[
                dict(text="Select S-parameter:", showarrow=False, x=0, y=1.08, 
                     xref="paper", yref="paper", align="left")
            ]
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Magnitude (dB)", secondary_y=False)
        fig.update_yaxes(title_text="Phase (°)", secondary_y=True)
        
        # Configure grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', secondary_y=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', secondary_y=True)
        
        # Save or show plot
        if output_file:
            output_path = Path(output_file)
            fig.write_html(str(output_path))
            print(f"Interactive plot saved to: {output_path}")
        else:
            output_path = self.snp_file_path.with_suffix('.html')
            fig.write_html(str(output_path))
            print(f"Interactive plot saved to: {output_path}")
        
        return fig
    
    def create_slider_plot(self, output_file=None):
        """Create plotly plot with sliders for port selection."""
        if self.network is None:
            print("No network loaded")
            return
        
        nports = self.network.nports
        freq_ghz = self.network.f / 1e9
        
        fig = go.Figure()
        
        # Create traces for all combinations, initially hidden
        for i in range(nports):
            for j in range(nports):
                s_param = self.network.s[:, i, j]
                s_db = 20 * np.log10(np.abs(s_param))
                
                fig.add_trace(
                    go.Scatter(
                        x=freq_ghz,
                        y=s_db,
                        mode='lines',
                        name=f'S{i+1}{j+1}',
                        visible=(i == 0 and j == 0),
                        line=dict(width=2)
                    )
                )
        
        # Create sliders for port selection
        steps_p1 = []
        for p1 in range(nports):
            step = dict(
                method="update",
                args=[{"visible": [False] * (nports * nports)},
                      {"title": f"S{p1+1}x Frequency Response"}],
            )
            # Show all S-parameters with first port = p1
            for p2 in range(nports):
                idx = p1 * nports + p2
                step["args"][0]["visible"][idx] = True
            steps_p1.append(step)
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Port 1: "},
            pad={"t": 50},
            steps=steps_p1
        )]
        
        fig.update_layout(
            sliders=sliders,
            title="S-Parameter Frequency Response with Port Selection",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Magnitude (dB)",
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        # Save plot
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = self.snp_file_path.with_suffix('_slider.html')
        
        fig.write_html(str(output_path))
        print(f"Slider plot saved to: {output_path}")
        
        return fig

def create_server_app(snp_plotter):
    """Create Flask server app for on-demand S-parameter plotting."""
    
    app = Flask(__name__)
    
    # HTML template for the interactive interface
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>S-Parameter Plotter - {{ filename }}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .controls { margin-bottom: 20px; }
            .controls label { margin-right: 10px; }
            .controls input, .controls select { margin-right: 20px; padding: 5px; }
            .controls button { padding: 8px 16px; background-color: #007bff; color: white; border: none; cursor: pointer; }
            .controls button:hover { background-color: #0056b3; }
            #plot { width: 100%; height: 600px; }
            .info { background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="info">
            <h2>S-Parameter Plotter: {{ filename }}</h2>
            <p>Ports: {{ nports }} | Frequency Points: {{ nfreq }} | Range: {{ freq_range }}</p>
        </div>
        
        <div class="controls">
            <label>Port 1:</label>
            <input type="number" id="p1" min="1" max="{{ nports }}" value="1">
            
            <label>Port 2:</label>
            <input type="number" id="p2" min="1" max="{{ nports }}" value="1">
            
            <label>Show:</label>
            <select id="show_type">
                <option value="both">Magnitude & Phase</option>
                <option value="magnitude">Magnitude Only</option>
                <option value="phase">Phase Only</option>
            </select>
            
            <button onclick="updatePlot()">Plot S-Parameter</button>
        </div>
        
        <div id="plot"></div>
        
        <script>
            function updatePlot() {
                const p1 = document.getElementById('p1').value;
                const p2 = document.getElementById('p2').value;
                const showType = document.getElementById('show_type').value;
                
                fetch('/plot_sparam', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({p1: parseInt(p1), p2: parseInt(p2), show_type: showType})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    
                    const plotDiv = document.getElementById('plot');
                    Plotly.newPlot(plotDiv, data.traces, data.layout);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error fetching data');
                });
            }
            
            // Initial plot
            updatePlot();
        </script>
    </body>
    </html>
    """
    
    @app.route('/')
    def index():
        return render_template_string(html_template,
            filename=snp_plotter.snp_file_path.name,
            nports=snp_plotter.network.nports,
            nfreq=len(snp_plotter.network.f),
            freq_range=f"{snp_plotter.network.f[0]/1e9:.3f} - {snp_plotter.network.f[-1]/1e9:.3f} GHz"
        )
    
    @app.route('/plot_sparam', methods=['POST'])
    def plot_sparam():
        try:
            print(f"Received plot request")  # Debug
            data = request.json
            p1 = data.get('p1', 1)
            p2 = data.get('p2', 1)
            show_type = data.get('show_type', 'both')
            print(f"Plotting S{p1}{p2}, show_type: {show_type}")  # Debug
            
            sparam_data = snp_plotter.get_sparam_data(p1, p2)
            if sparam_data is None:
                print(f"Failed to get data for S{p1}{p2}")  # Debug
                return jsonify({'error': f'Invalid port combination S{p1}{p2}'})
            
            print(f"Got data: {len(sparam_data['frequency'])} frequency points")  # Debug
            
            traces = []
            
            if show_type in ['both', 'magnitude']:
                traces.append({
                    'x': sparam_data['frequency'],
                    'y': sparam_data['magnitude_db'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': f'|S{p1}{p2}| (dB)',
                    'line': {'width': 2}
                })
            
            if show_type in ['both', 'phase']:
                traces.append({
                    'x': sparam_data['frequency'],
                    'y': sparam_data['phase_deg'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': f'∠S{p1}{p2} (°)',
                    'line': {'width': 2, 'dash': 'dash'},
                    'yaxis': 'y2' if show_type == 'both' else 'y'
                })
            
            layout = {
                'title': f'S{p1}{p2} Frequency Response - {snp_plotter.snp_file_path.name}',
                'xaxis': {'title': 'Frequency (GHz)', 'showgrid': True},
                'yaxis': {'title': 'Magnitude (dB)', 'showgrid': True},
                'hovermode': 'x unified',
                'width': 1000,
                'height': 600
            }
            
            if show_type == 'both':
                layout['yaxis2'] = {
                    'title': 'Phase (°)',
                    'overlaying': 'y',
                    'side': 'right',
                    'showgrid': True
                }
            
            print(f"Returning {len(traces)} traces")  # Debug
            return jsonify({'traces': traces, 'layout': layout})
            
        except Exception as e:
            print(f"Server error: {e}")  # Debug
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)})
    
    return app

def main():
    parser = argparse.ArgumentParser(description="Plot S-parameter data from SNP files")
    parser.add_argument("snp_file", help="Path to S-parameter file (.snp, .s2p, etc.)")
    parser.add_argument("-o", "--output", help="Output HTML file path")
    parser.add_argument("--slider", action="store_true", help="Use slider interface instead of dropdown")
    parser.add_argument("--server", action="store_true", help="Run as interactive server (recommended for large files)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--bind_all", action="store_true", help="Bind to all interfaces (0.0.0.0) for remote access")
    
    args = parser.parse_args()
    
    # Override host if bind_all is specified
    if args.bind_all:
        args.host = "0.0.0.0"
    
    # Validate input file
    snp_path = Path(args.snp_file)
    if not snp_path.exists():
        print(f"Error: File {snp_path} does not exist")
        sys.exit(1)
    
    # Create plotter
    plotter = SNPPlotter(args.snp_file)
    
    # Run in server mode
    if args.server:
        print(f"Starting server for {plotter.network.nports}-port S-parameter file...")
        print(f"Open http://{args.host}:{args.port} in your browser")
        
        app = create_server_app(plotter)
        app.run(host=args.host, port=args.port, debug=False)
        return
    
    # Create static plot
    if args.slider:
        plotter.create_slider_plot(args.output)
    else:
        plotter.create_interactive_plot(args.output)

if __name__ == "__main__":
    main() 