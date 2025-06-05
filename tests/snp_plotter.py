import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import skrf as rf
from pathlib import Path
import sys

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
            print(f"Frequency range: {self.network.f[0]/1e9:.3f} - {self.network.f[-1]/1e9:.3f} GHz")
            print(f"Number of ports: {self.network.nports}")
            print(f"Number of frequency points: {len(self.network.f)}")
        except Exception as e:
            print(f"Error loading network: {e}")
            sys.exit(1)
    
    def create_interactive_plot(self, output_file=None):
        """Create interactive plotly plot with scrollable S-parameter selection."""
        if self.network is None:
            print("No network loaded")
            return
        
        nports = self.network.nports
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

def main():
    parser = argparse.ArgumentParser(description="Plot S-parameter data from SNP files")
    parser.add_argument("snp_file", help="Path to S-parameter file (.snp, .s2p, etc.)")
    parser.add_argument("-o", "--output", help="Output HTML file path")
    parser.add_argument("--slider", action="store_true", help="Use slider interface instead of dropdown")
    
    args = parser.parse_args()
    
    # Validate input file
    snp_path = Path(args.snp_file)
    if not snp_path.exists():
        print(f"Error: File {snp_path} does not exist")
        sys.exit(1)
    
    # Create plotter
    plotter = SNPPlotter(args.snp_file)
    
    # Create plot
    if args.slider:
        plotter.create_slider_plot(args.output)
    else:
        plotter.create_interactive_plot(args.output)

if __name__ == "__main__":
    main() 