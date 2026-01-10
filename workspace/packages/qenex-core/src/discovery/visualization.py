"""
QENEX Visualization Module
==========================
Generates visualizations for scientific discovery results including:
- Hypothesis networks (domain connections)
- Cross-domain pattern maps
- Score distributions
- Parameter exploration landscapes
- Time series plots for simulations

Uses matplotlib for static plots and can export to various formats.

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Matplotlib configuration for headless environments
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


# =============================================================================
# QENEX COLOR SCHEME
# =============================================================================

QENEX_COLORS = {
    "primary": "#2E86AB",       # Blue
    "secondary": "#A23B72",     # Magenta
    "accent": "#F18F01",        # Orange
    "success": "#C73E1D",       # Red (for emphasis)
    "background": "#1a1a2e",    # Dark blue-black
    "text": "#e6e6e6",          # Light gray
    "grid": "#404040",          # Dark gray
    
    # Domain colors
    "quantum_chemistry": "#3498db",
    "climate_science": "#2ecc71",
    "neuroscience": "#9b59b6",
    "astrophysics": "#e74c3c",
    "biology": "#1abc9c",
    "physics": "#f39c12",
    "mathematics": "#95a5a6",
}

PATTERN_COLORS = {
    "SCALING_LAW": "#3498db",
    "PHASE_TRANSITION": "#e74c3c",
    "SYMMETRY": "#9b59b6",
    "FEEDBACK_LOOP": "#2ecc71",
    "OPTIMIZATION": "#f39c12",
    "EMERGENCE": "#1abc9c",
    "NETWORK_EFFECT": "#e91e63",
    "DIFFUSION": "#00bcd4",
    "OSCILLATION": "#ff9800",
    "RENORMALIZATION": "#795548",
}


def setup_qenex_style():
    """Configure matplotlib for QENEX dark theme."""
    plt.rcParams.update({
        'figure.facecolor': QENEX_COLORS["background"],
        'axes.facecolor': QENEX_COLORS["background"],
        'axes.edgecolor': QENEX_COLORS["grid"],
        'axes.labelcolor': QENEX_COLORS["text"],
        'text.color': QENEX_COLORS["text"],
        'xtick.color': QENEX_COLORS["text"],
        'ytick.color': QENEX_COLORS["text"],
        'grid.color': QENEX_COLORS["grid"],
        'legend.facecolor': QENEX_COLORS["background"],
        'legend.edgecolor': QENEX_COLORS["grid"],
        'font.family': 'sans-serif',
        'font.size': 10,
    })


# =============================================================================
# HYPOTHESIS VISUALIZATION
# =============================================================================

class HypothesisVisualizer:
    """Visualize hypothesis generation results."""
    
    def __init__(self, output_dir: str = "/opt/qenex_lab/workspace/reports/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        setup_qenex_style()
    
    def plot_hypothesis_scores(self, hypotheses: List[Dict], 
                                filename: str = "hypothesis_scores.png") -> str:
        """
        Bar chart of hypothesis scores by dimension.
        
        Args:
            hypotheses: List of hypothesis dictionaries with score fields
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract scores
        ids = [h.get("id", f"H{i}")[:8] for i, h in enumerate(hypotheses)]
        novelty = [h.get("novelty_score", 0) for h in hypotheses]
        testability = [h.get("testability_score", 0) for h in hypotheses]
        impact = [h.get("impact_score", 0) for h in hypotheses]
        plausibility = [h.get("plausibility_score", 0) for h in hypotheses]
        
        x = np.arange(len(ids))
        width = 0.6
        
        # Novelty
        axes[0, 0].bar(x, novelty, width, color=QENEX_COLORS["primary"])
        axes[0, 0].set_title("Novelty Scores", fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(ids, rotation=45, ha='right')
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].axhline(y=0.5, color=QENEX_COLORS["accent"], linestyle='--', alpha=0.5)
        
        # Testability
        axes[0, 1].bar(x, testability, width, color=QENEX_COLORS["secondary"])
        axes[0, 1].set_title("Testability Scores", fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(ids, rotation=45, ha='right')
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].axhline(y=0.5, color=QENEX_COLORS["accent"], linestyle='--', alpha=0.5)
        
        # Impact
        axes[1, 0].bar(x, impact, width, color=QENEX_COLORS["accent"])
        axes[1, 0].set_title("Impact Scores", fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(ids, rotation=45, ha='right')
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].axhline(y=0.5, color=QENEX_COLORS["primary"], linestyle='--', alpha=0.5)
        
        # Plausibility
        axes[1, 1].bar(x, plausibility, width, color=QENEX_COLORS["success"])
        axes[1, 1].set_title("Plausibility Scores", fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(ids, rotation=45, ha='right')
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].axhline(y=0.5, color=QENEX_COLORS["primary"], linestyle='--', alpha=0.5)
        
        fig.suptitle("QENEX Hypothesis Evaluation", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        return filepath
    
    def plot_pattern_distribution(self, hypotheses: List[Dict],
                                   filename: str = "pattern_distribution.png") -> str:
        """
        Pie chart showing distribution of patterns used.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        # Count patterns
        pattern_counts = {}
        for h in hypotheses:
            pattern = h.get("pattern", "UNKNOWN")
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        labels = list(pattern_counts.keys())
        sizes = list(pattern_counts.values())
        colors = [PATTERN_COLORS.get(p, "#808080") for p in labels]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.85,
            textprops={'color': QENEX_COLORS["text"]}
        )
        
        # Draw center circle for donut chart
        centre_circle = plt.Circle((0, 0), 0.60, fc=QENEX_COLORS["background"])
        ax.add_patch(centre_circle)
        
        ax.set_title("Pattern Distribution in Generated Hypotheses", 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add total count in center
        ax.text(0, 0, f"Total\n{sum(sizes)}", ha='center', va='center',
                fontsize=16, fontweight='bold', color=QENEX_COLORS["text"])
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        return filepath
    
    def plot_domain_network(self, hypotheses: List[Dict],
                            filename: str = "domain_network.png") -> str:
        """
        Network graph showing connections between domains via hypotheses.
        
        Args:
            hypotheses: List of hypothesis dictionaries with source_analogy and target_domain
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        # Build adjacency
        domains = set()
        edges = []
        
        for h in hypotheses:
            domain = h.get("domain", "unknown")
            domains.add(domain)
            
            source = h.get("source_analogy", "")
            if source:
                # Extract source domain from "domain: phenomenon" format
                source_domain = source.split(":")[0].strip() if ":" in source else source
                if source_domain:
                    domains.add(source_domain)
                    edges.append((source_domain, domain))
        
        domains = list(domains)
        n_domains = len(domains)
        
        if n_domains == 0:
            # No domains to plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No domain connections found", 
                    ha='center', va='center', fontsize=14)
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
            plt.close(fig)
            return filepath
        
        # Position domains in a circle
        angles = np.linspace(0, 2*np.pi, n_domains, endpoint=False)
        radius = 2.5
        positions = {
            domain: (radius * np.cos(angle), radius * np.sin(angle))
            for domain, angle in zip(domains, angles)
        }
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Draw edges
        for source, target in edges:
            if source in positions and target in positions:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle="->", color=QENEX_COLORS["primary"],
                                          alpha=0.6, connectionstyle="arc3,rad=0.1"))
        
        # Draw nodes
        for domain, (x, y) in positions.items():
            color = QENEX_COLORS.get(domain.lower().replace(" ", "_"), QENEX_COLORS["accent"])
            circle = plt.Circle((x, y), 0.4, color=color, ec=QENEX_COLORS["text"], linewidth=2)
            ax.add_patch(circle)
            
            # Label
            ax.text(x, y - 0.7, domain.replace("_", "\n"), ha='center', va='top',
                   fontsize=9, fontweight='bold', color=QENEX_COLORS["text"])
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Cross-Domain Hypothesis Network", fontsize=14, fontweight='bold', pad=20)
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        return filepath
    
    def plot_composite_scores(self, hypotheses: List[Dict],
                               filename: str = "composite_scores.png") -> str:
        """
        Horizontal bar chart of composite scores ranked.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        # Sort by composite score
        sorted_hyps = sorted(hypotheses, key=lambda h: h.get("composite_score", 0), reverse=True)[:15]
        
        labels = [h.get("statement", "")[:50] + "..." for h in sorted_hyps]
        scores = [h.get("composite_score", 0) for h in sorted_hyps]
        patterns = [h.get("pattern", "UNKNOWN") for h in sorted_hyps]
        colors = [PATTERN_COLORS.get(p, "#808080") for p in patterns]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, scores, color=colors, height=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Composite Score")
        ax.set_xlim(0, 1.1)
        ax.axvline(x=0.5, color=QENEX_COLORS["accent"], linestyle='--', alpha=0.5, label='Threshold')
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, i, f"{score:.2f}", va='center', fontsize=8)
        
        # Legend for patterns
        unique_patterns = list(set(patterns))
        legend_handles = [mpatches.Patch(color=PATTERN_COLORS.get(p, "#808080"), label=p) 
                         for p in unique_patterns]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=8)
        
        ax.set_title("Top Hypotheses by Composite Score", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        return filepath


# =============================================================================
# SIMULATION VISUALIZATION
# =============================================================================

class SimulationVisualizer:
    """Visualize simulation results from domain simulators."""
    
    def __init__(self, output_dir: str = "/opt/qenex_lab/workspace/reports/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        setup_qenex_style()
    
    def plot_parameter_exploration(self, results: List[Dict],
                                    x_param: str, y_param: str, color_param: str,
                                    filename: str = "param_exploration.png") -> str:
        """
        Scatter plot of parameter exploration results.
        
        Args:
            results: List of simulation result dictionaries
            x_param: Parameter name for x-axis
            y_param: Parameter name for y-axis
            color_param: Parameter name for coloring (or 'objective_value')
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        x_vals = []
        y_vals = []
        c_vals = []
        
        for r in results:
            params = r.get("parameters", {})
            outputs = r.get("outputs", {})
            
            x = params.get(x_param) or outputs.get(x_param)
            y = params.get(y_param) or outputs.get(y_param)
            c = params.get(color_param) or outputs.get(color_param) or r.get(color_param)
            
            if x is not None and y is not None and c is not None:
                x_vals.append(x)
                y_vals.append(y)
                c_vals.append(c)
        
        if not x_vals:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f"No data for {x_param} vs {y_param}", 
                    ha='center', va='center', fontsize=14)
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
            plt.close(fig)
            return filepath
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(x_vals, y_vals, c=c_vals, cmap='viridis', 
                            s=100, alpha=0.7, edgecolors=QENEX_COLORS["text"])
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_param.replace("_", " ").title())
        
        ax.set_xlabel(x_param.replace("_", " ").title())
        ax.set_ylabel(y_param.replace("_", " ").title())
        ax.set_title(f"Parameter Exploration: {y_param} vs {x_param}", 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        return filepath
    
    def plot_objective_history(self, objectives: List[float],
                                filename: str = "objective_history.png") -> str:
        """
        Line plot of objective value over optimization iterations.
        
        Args:
            objectives: List of objective values
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = range(1, len(objectives) + 1)
        
        ax.plot(iterations, objectives, '-o', color=QENEX_COLORS["primary"],
               markersize=4, linewidth=1.5, label='Current')
        
        # Running best
        running_best = np.maximum.accumulate(objectives)
        ax.plot(iterations, running_best, '--', color=QENEX_COLORS["accent"],
               linewidth=2, label='Best so far')
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value")
        ax.set_title("Optimization History", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        return filepath
    
    def plot_climate_projection(self, years: List[float], temps: List[float],
                                 co2: List[float], sea_level: List[float],
                                 filename: str = "climate_projection.png") -> str:
        """
        Multi-panel climate projection plot.
        
        Args:
            years: List of years
            temps: Temperature anomaly (K)
            co2: CO2 concentration (ppm)
            sea_level: Sea level rise (m)
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Temperature
        axes[0].plot(years, temps, color=QENEX_COLORS["success"], linewidth=2)
        axes[0].fill_between(years, temps, alpha=0.3, color=QENEX_COLORS["success"])
        axes[0].set_ylabel("Temperature\nAnomaly (K)")
        axes[0].axhline(y=1.5, color=QENEX_COLORS["accent"], linestyle='--', 
                       label='1.5°C target')
        axes[0].axhline(y=2.0, color=QENEX_COLORS["secondary"], linestyle='--',
                       label='2.0°C limit')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # CO2
        axes[1].plot(years, co2, color=QENEX_COLORS["primary"], linewidth=2)
        axes[1].fill_between(years, co2, alpha=0.3, color=QENEX_COLORS["primary"])
        axes[1].set_ylabel("CO₂ (ppm)")
        axes[1].axhline(y=450, color=QENEX_COLORS["accent"], linestyle='--',
                       label='450 ppm threshold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # Sea level
        axes[2].plot(years, sea_level, color=QENEX_COLORS["secondary"], linewidth=2)
        axes[2].fill_between(years, sea_level, alpha=0.3, color=QENEX_COLORS["secondary"])
        axes[2].set_ylabel("Sea Level\nRise (m)")
        axes[2].set_xlabel("Year")
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle("Climate Projection", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        return filepath
    
    def plot_neural_spikes(self, spike_times: Dict[int, List[float]],
                            filename: str = "neural_spikes.png") -> str:
        """
        Raster plot of neural spike times.
        
        Args:
            spike_times: Dictionary {neuron_id: [spike_times]}
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for neuron_id, times in spike_times.items():
            ax.scatter(times, [neuron_id] * len(times), s=2, 
                      color=QENEX_COLORS["primary"], alpha=0.7)
        
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron ID")
        ax.set_title("Neural Spike Raster", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        return filepath


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class DiscoveryReportGenerator:
    """Generate comprehensive visual reports from discovery campaigns."""
    
    def __init__(self, output_dir: str = "/opt/qenex_lab/workspace/reports"):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.hyp_viz = HypothesisVisualizer(self.figures_dir)
        self.sim_viz = SimulationVisualizer(self.figures_dir)
    
    def generate_hypothesis_report(self, hypotheses: List[Dict],
                                    title: str = "Hypothesis Generation Report") -> Dict[str, str]:
        """
        Generate all hypothesis visualizations.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            title: Report title
        
        Returns:
            Dictionary mapping figure names to file paths
        """
        figures = {}
        
        if not hypotheses:
            print("No hypotheses to visualize")
            return figures
        
        print(f"Generating hypothesis visualizations for {len(hypotheses)} hypotheses...")
        
        figures["scores"] = self.hyp_viz.plot_hypothesis_scores(hypotheses)
        figures["patterns"] = self.hyp_viz.plot_pattern_distribution(hypotheses)
        figures["network"] = self.hyp_viz.plot_domain_network(hypotheses)
        figures["composite"] = self.hyp_viz.plot_composite_scores(hypotheses)
        
        print(f"Generated {len(figures)} figures in {self.figures_dir}")
        
        return figures
    
    def generate_summary_dashboard(self, hypotheses: List[Dict],
                                    simulation_results: Optional[Dict[str, List[Dict]]] = None,
                                    filename: str = "discovery_dashboard.png") -> str:
        """
        Generate a single-page dashboard summarizing all results.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            simulation_results: Optional dict of {domain: [results]}
            filename: Output filename
        
        Returns:
            Path to saved figure
        """
        setup_qenex_style()
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # === Panel 1: Summary Stats ===
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        n_hyp = len(hypotheses)
        n_patterns = len(set(h.get("pattern", "") for h in hypotheses))
        n_domains = len(set(h.get("domain", "") for h in hypotheses))
        avg_score = np.mean([h.get("composite_score", 0) for h in hypotheses]) if hypotheses else 0
        
        stats_text = f"""
QENEX Discovery Summary
━━━━━━━━━━━━━━━━━━━━━
Hypotheses Generated: {n_hyp}
Unique Patterns Used: {n_patterns}
Domains Explored: {n_domains}
Average Score: {avg_score:.2f}
━━━━━━━━━━━━━━━━━━━━━
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        ax1.text(0.5, 0.5, stats_text, transform=ax1.transAxes,
                fontsize=12, fontfamily='monospace', va='center', ha='center',
                bbox=dict(boxstyle='round', facecolor=QENEX_COLORS["grid"], alpha=0.5))
        
        # === Panel 2: Pattern Distribution ===
        ax2 = fig.add_subplot(gs[0, 1])
        pattern_counts = {}
        for h in hypotheses:
            p = h.get("pattern", "UNKNOWN")
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
        
        if pattern_counts:
            labels = list(pattern_counts.keys())
            sizes = list(pattern_counts.values())
            colors = [PATTERN_COLORS.get(p, "#808080") for p in labels]
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                   textprops={'fontsize': 8, 'color': QENEX_COLORS["text"]})
        ax2.set_title("Pattern Distribution", fontweight='bold')
        
        # === Panel 3: Score Distributions ===
        ax3 = fig.add_subplot(gs[0, 2])
        score_types = ['novelty_score', 'testability_score', 'impact_score', 'plausibility_score']
        score_data = [[h.get(s, 0) for h in hypotheses] for s in score_types]
        
        bp = ax3.boxplot(score_data, labels=['Novelty', 'Testability', 'Impact', 'Plausibility'],
                        patch_artist=True)
        colors = [QENEX_COLORS["primary"], QENEX_COLORS["secondary"], 
                 QENEX_COLORS["accent"], QENEX_COLORS["success"]]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_title("Score Distributions", fontweight='bold')
        ax3.set_ylim(0, 1.1)
        
        # === Panel 4-6: Top Hypotheses ===
        ax_top = fig.add_subplot(gs[1, :])
        ax_top.axis('off')
        
        sorted_hyps = sorted(hypotheses, key=lambda h: h.get("composite_score", 0), reverse=True)[:5]
        
        top_text = "TOP 5 HYPOTHESES\n" + "=" * 80 + "\n"
        for i, h in enumerate(sorted_hyps, 1):
            top_text += f"\n{i}. [{h.get('pattern', 'N/A')}] Score: {h.get('composite_score', 0):.2f}\n"
            top_text += f"   Domain: {h.get('domain', 'N/A')}\n"
            statement = h.get('statement', 'N/A')[:120]
            top_text += f"   {statement}...\n"
        
        ax_top.text(0.02, 0.98, top_text, transform=ax_top.transAxes,
                   fontsize=9, fontfamily='monospace', va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor=QENEX_COLORS["grid"], alpha=0.3))
        
        # === Panel 7: Domain Counts ===
        ax7 = fig.add_subplot(gs[2, 0])
        domain_counts = {}
        for h in hypotheses:
            d = h.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        
        if domain_counts:
            domains = list(domain_counts.keys())
            counts = list(domain_counts.values())
            colors = [QENEX_COLORS.get(d.lower().replace(" ", "_"), QENEX_COLORS["accent"]) 
                     for d in domains]
            ax7.barh(domains, counts, color=colors)
            ax7.set_xlabel("Count")
        ax7.set_title("Hypotheses by Domain", fontweight='bold')
        
        # === Panel 8: Composite Score Histogram ===
        ax8 = fig.add_subplot(gs[2, 1])
        composite_scores = [h.get("composite_score", 0) for h in hypotheses]
        ax8.hist(composite_scores, bins=10, color=QENEX_COLORS["primary"], 
                edgecolor=QENEX_COLORS["text"], alpha=0.7)
        ax8.axvline(x=0.5, color=QENEX_COLORS["accent"], linestyle='--', label='Threshold')
        ax8.set_xlabel("Composite Score")
        ax8.set_ylabel("Frequency")
        ax8.set_title("Score Distribution", fontweight='bold')
        ax8.legend()
        
        # === Panel 9: Quality Metrics ===
        ax9 = fig.add_subplot(gs[2, 2])
        
        # Radar chart data
        metrics = ['Avg Novelty', 'Avg Testability', 'Avg Impact', 'Avg Plausibility']
        values = [
            np.mean([h.get("novelty_score", 0) for h in hypotheses]),
            np.mean([h.get("testability_score", 0) for h in hypotheses]),
            np.mean([h.get("impact_score", 0) for h in hypotheses]),
            np.mean([h.get("plausibility_score", 0) for h in hypotheses]),
        ]
        
        # Simple bar chart instead of radar
        x = np.arange(len(metrics))
        bars = ax9.bar(x, values, color=colors[:4])
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics, rotation=45, ha='right', fontsize=8)
        ax9.set_ylim(0, 1)
        ax9.set_title("Average Quality Metrics", fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', fontsize=8)
        
        fig.suptitle("QENEX Discovery Campaign Dashboard", fontsize=16, fontweight='bold', y=0.98)
        
        filepath = os.path.join(self.figures_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=QENEX_COLORS["background"])
        plt.close(fig)
        
        print(f"Dashboard saved to {filepath}")
        return filepath


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Demonstrate visualization capabilities."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     QENEX VISUALIZATION MODULE                               ║
    ║     Scientific Discovery Visualization System                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create sample hypotheses for demonstration
    sample_hypotheses = [
        {
            "id": "abc123",
            "statement": "The scaling law pattern governs neural firing rates in cortical networks",
            "domain": "neuroscience",
            "pattern": "SCALING_LAW",
            "novelty_score": 0.85,
            "testability_score": 0.7,
            "impact_score": 0.6,
            "plausibility_score": 0.8,
            "composite_score": 0.74,
        },
        {
            "id": "def456",
            "statement": "Phase transition behavior in climate tipping points follows critical exponents",
            "domain": "climate_science",
            "pattern": "PHASE_TRANSITION",
            "novelty_score": 0.9,
            "testability_score": 0.5,
            "impact_score": 0.95,
            "plausibility_score": 0.75,
            "composite_score": 0.77,
            "source_analogy": "physics: Ising model",
        },
        {
            "id": "ghi789",
            "statement": "Stellar evolution exhibits emergence patterns similar to neural development",
            "domain": "astrophysics",
            "pattern": "EMERGENCE",
            "novelty_score": 0.95,
            "testability_score": 0.4,
            "impact_score": 0.7,
            "plausibility_score": 0.6,
            "composite_score": 0.66,
            "source_analogy": "neuroscience: development",
        },
    ]
    
    # Generate visualizations
    generator = DiscoveryReportGenerator()
    
    figures = generator.generate_hypothesis_report(sample_hypotheses)
    print(f"\nGenerated figures: {list(figures.keys())}")
    
    dashboard = generator.generate_summary_dashboard(sample_hypotheses)
    print(f"Dashboard: {dashboard}")
    
    return generator


if __name__ == "__main__":
    main()
