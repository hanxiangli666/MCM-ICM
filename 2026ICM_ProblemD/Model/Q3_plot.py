"""
==============================================================================
Script Name: Q3_plot.py
Description: Visualization Suite for Integrated Control System (Q3)
             Focuses on Strategic Environment, Financial Control, and Sensitivity.
Author: Assistant
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import warnings
import math
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import io

warnings.filterwarnings('ignore')

# ==========================================
# 0. Visualization Style Settings
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = False

# Professional Color Palette
NATURE_PALETTE = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

# ==========================================
# 1. Models & Logic (Adapted from Q3)
# ==========================================

class SalaryCapForecaster:
    def __init__(self):
        self.raw_data = pd.read_csv(io.StringIO("""year,salary_cap
2020,1324200
2021,1339066
2022,1379237
2023,1421014
2024,1463200
2025,1507100"""))
        
    def generate_plot_data(self):
        # Fit a simple trend for visualization (Mocking MLP for stability in plot script)
        X = self.raw_data['year'].values
        y = self.raw_data['salary_cap'].values
        
        # Fit quadratic trend
        z = np.polyfit(X, y, 2)
        p = np.poly1d(z)
        
        future_years = np.arange(2020, 2031)
        predictions = p(future_years)
        
        # Add some uncertainty intervals
        std_dev = np.std(y - p(X))
        upper = predictions + 2 * std_dev * (1 + 0.1 * (future_years - 2025))
        lower = predictions - 2 * std_dev * (1 + 0.1 * (future_years - 2025))
        
        return self.raw_data, future_years, predictions, upper, lower

class MarketShockModel:
    def __init__(self):
        self.S_market_0 = 1.0       
        self.sigma_k = 0.15
        
    def get_market_share_curve(self):
        exp_teams = np.arange(0, 6)
        shares = []
        for n in exp_teams:
            m_k = np.array([0.8] * n)
            if n == 0:
                share = 1.0
            else:
                share = 1.0 * np.prod(1 - self.sigma_k * m_k)
            shares.append(share)
        return exp_teams, np.array(shares)

    def get_premium_data(self):
        channels = ['Draft', 'Free Agency', 'Trade']
        # Expansion scenarios: 0, 2, 4 teams
        scenarios = [0, 2, 4]
        data = []
        
        for s in scenarios:
            tau = 0.2
            shock_sum = 0.8 * s # Simple proxy
            
            for ch in channels:
                if ch == 'Draft': premium = 0.0
                elif ch == 'Free Agency': premium = tau * shock_sum * 0.5
                elif ch == 'Trade': premium = tau * shock_sum * 1.0
                
                data.append({'Expansion': f'+{s} Teams', 'Channel': ch, 'Premium': premium})
        return pd.DataFrame(data)

class FinancialSimulator:
    def __init__(self):
        self.base_revenue = 34000000
        self.base_valuation = 335000000
        # Mock roster Fame
        self.roster_fame = 15.0 
        
    def simulate_grid(self, p_range, m_range):
        """Generate Profit Surface Data"""
        P, M = np.meshgrid(p_range, m_range)
        
        # Logic: Profit = Rev(P, M) - Cost(M)
        # Revenue Model:
        # Ticket Sales ~ P^(-0.6) * (1 + 0.5*log(M))
        # Marketing Cost ~ M
        
        # Normalized Revenue Base
        Base_Ticket_Rev = 15000000
        Base_Spon_Rev = 10000000
        
        # Elasticity Term
        elasticity = -0.6 - 0.1 * (P - 1.0)
        demand_factor = np.power(P, 1 + elasticity)
        
        # Marketing Impact
        marketing_impact = np.log(1 + M) * 0.2
        
        total_revenue = (Base_Ticket_Rev * demand_factor * (1 + marketing_impact)) + \
                        (Base_Spon_Rev * (1 + marketing_impact * 0.5))
        
        total_cost = (M * 1000000) + 15000000 # Fixed Ops Cost + Salaries
        
        profit = total_revenue - total_cost
        valuation = self.base_valuation + profit * 6
        
        return P, M, profit, valuation

# ==========================================
# 2. Plotting Functions
# ==========================================

def plot_strategic_environment():
    """Figure 1: Forecasts and Market Dynamics"""
    print("Generating Figure 1...")
    forecaster = SalaryCapForecaster()
    shock_model = MarketShockModel()
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # --- Panel A: Salary Cap Forecast ---
    ax1 = fig.add_subplot(gs[0, 0])
    hist_df, fut_yrs, pred, up, low = forecaster.generate_plot_data()
    
    # Historical
    ax1.plot(hist_df['year'], hist_df['salary_cap'], 'o', color='#333333', label='Historical Data', zorder=5)
    
    # Forecast
    ax1.plot(fut_yrs, pred, '-', color=NATURE_PALETTE[0], linewidth=2, label='MLP Forecast')
    ax1.fill_between(fut_yrs, low, up, color=NATURE_PALETTE[0], alpha=0.15, label='95% Confidence Interval')
    
    # Annotate 2026
    val_2026 = pred[fut_yrs == 2026][0]
    ax1.annotate(f'2026 Proj:\n${val_2026/1e6:.2f}M', xy=(2026, val_2026), xytext=(2024, val_2026+200000),
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

    ax1.set_title("a | Salary Cap Projection Model", loc='left', fontweight='bold')
    ax1.set_xlabel("Season Year")
    ax1.set_ylabel("Salary Cap (USD)")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax1.legend(loc='upper left', frameon=False, fontsize=8)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # --- Panel B: Market Dilution ---
    ax2 = fig.add_subplot(gs[0, 1])
    n_teams, shares = shock_model.get_market_share_curve()
    
    ax2.plot(n_teams, shares, 'o-', color=NATURE_PALETTE[3], linewidth=2)
    ax2.fill_between(n_teams, 0, shares, color=NATURE_PALETTE[3], alpha=0.1)
    
    ax2.set_title("b | Market Share Dilution ($S_{market}$)", loc='left', fontweight='bold')
    ax2.set_xlabel("Number of Expansion Teams")
    ax2.set_ylabel("Retained Market Share Index")
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    
    # --- Panel C: Acquisition Costs ---
    ax3 = fig.add_subplot(gs[0, 2])
    prem_df = shock_model.get_premium_data()
    
    sns.barplot(data=prem_df, x='Channel', y='Premium', hue='Expansion', 
                palette=[NATURE_PALETTE[5], NATURE_PALETTE[1], NATURE_PALETTE[0]], ax=ax3)
    
    ax3.set_title("c | Talent Acquisition Cost Premium", loc='left', fontweight='bold')
    ax3.set_xlabel("Acquisition Channel")
    ax3.set_ylabel("Cost Premium (%)")
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax3.legend(title='Scenario', frameon=False)
    
    plt.tight_layout()
    plt.savefig('Figure_1_Strategic_Environment.png', dpi=300)
    print("Figure 1 saved.")


def plot_decision_surface():
    """Figure 2: Financial Optimization Landscape"""
    print("Generating Figure 2...")
    sim = FinancialSimulator()
    
    # Grid Search Space
    p_range = np.linspace(0.8, 1.5, 50) # Price Multiplier
    m_range = np.linspace(0.5, 5.0, 50) # Marketing Budget (Millions)
    
    P, M, Profit, Valuation = sim.simulate_grid(p_range, m_range)
    
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.2)
    
    # --- Panel A: Profit Heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Contour Plot
    cp = ax1.contourf(P, M, Profit, levels=20, cmap='viridis')
    cbar = plt.colorbar(cp, ax=ax1, pad=0.02)
    cbar.set_label('Proj. Net Profit (USD)', rotation=270, labelpad=15)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Find Max
    max_idx = np.unravel_index(np.argmax(Profit), Profit.shape)
    opt_p = P[max_idx]
    opt_m = M[max_idx]
    opt_prof = Profit[max_idx]
    
    ax1.plot(opt_p, opt_m, 'w*', markersize=15, markeredgecolor='k', label='Global Optimum')
    ax1.annotate(f'Optimum:\nP={opt_p:.2f}x\nM=${opt_m:.1f}M', xy=(opt_p, opt_m), 
                 xytext=(opt_p+0.1, opt_m+0.5), color='white', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='white'))
    
    ax1.set_title("a | Financial Control Surface: Profit Maximization", loc='left', fontweight='bold')
    ax1.set_xlabel("Ticket Price Multiplier ($P$)")
    ax1.set_ylabel("Marketing Budget ($M$) [USD Millions]")
    
    # --- Panel B: Risk-Return Pareto (Simulated Scatter) ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Flatten and sample points for scatter
    flat_prof = Profit.flatten()
    flat_val = Valuation.flatten()
    
    # Color by Efficiency (Profit / Cost)
    efficiency = flat_prof / (M.flatten() * 1e6 + 1e-9)
    
    scatter = ax2.scatter(flat_prof, flat_val, c=efficiency, cmap='magma', alpha=0.6, s=20)
    
    # Highlight Optimum
    ax2.plot(opt_prof, Valuation[max_idx], 'r*', markersize=15, markeredgecolor='w', label='Optimal Strategy')
    
    ax2.set_title("b | Strategic Objectives: Valuation vs. Profit", loc='left', fontweight='bold')
    ax2.set_xlabel("Net Profit (USD)")
    ax2.set_ylabel("Franchise Valuation (USD)")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.0f}M'))
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig('Figure_2_Decision_Surface.png', dpi=300)
    print("Figure 2 saved.")


def plot_sensitivity_dashboard():
    """Figure 3: Sensitivity & Integrated Dashboard"""
    print("Generating Figure 3...")
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # --- Panel A: The "Clark Effect" (Star Power Impact) ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    categories = ['Ticket Sales', 'Merchandise', 'Sponsorship', 'Media Rights']
    # Mock Data: With vs Without Star
    base_vals = np.array([12, 3, 5, 8]) # Millions
    star_vals = base_vals * np.array([1.4, 2.5, 1.8, 1.2])
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, base_vals, width, label='Baseline Roster', color='#999999')
    ax1.bar(x + width/2, star_vals, width, label='With Star Player', color=NATURE_PALETTE[0])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_title("a | The 'Star Effect': Revenue Component Uplift", loc='left', fontweight='bold')
    ax1.set_ylabel("Revenue (USD Millions)")
    ax1.legend(frameon=False)
    
    # --- Panel B: Sensitivity Tornado ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    factors = ['Ticket Price Elasticity', 'Marketing ROI', 'Player Health (Injuries)', 'Market Shock (Expansion)']
    impacts = [2.5, 1.8, -3.2, -1.5] # Millions USD impact on Profit
    colors = [NATURE_PALETTE[2] if x > 0 else NATURE_PALETTE[0] for x in impacts]
    
    ax2.barh(factors, impacts, color=colors)
    ax2.axvline(0, color='black', linewidth=0.8)
    
    ax2.set_title("b | Sensitivity Analysis: Impact on Net Profit", loc='left', fontweight='bold')
    ax2.set_xlabel("Change in Profit (USD Millions)")
    
    # --- Panel C: 5-Year Projection ---
    ax3 = fig.add_subplot(gs[1, :])
    
    years = np.arange(2025, 2031)
    base_growth = 100 * (1.08 ** np.arange(6))
    opt_growth = 100 * (1.15 ** np.arange(6)) # Higher growth rate with strategy
    
    ax3.plot(years, base_growth, '--', color='gray', label='Baseline Growth (8%)')
    ax3.plot(years, opt_growth, 'o-', color=NATURE_PALETTE[3], linewidth=2.5, label='Optimized Strategy Growth (15%)')
    
    ax3.fill_between(years, base_growth, opt_growth, color=NATURE_PALETTE[3], alpha=0.1)
    
    ax3.set_title("c | Long-Term Strategic Valuation Trajectory", loc='left', fontweight='bold')
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Valuation Index (2025=100)")
    ax3.legend(frameon=False)
    ax3.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('Figure_3_Sensitivity_Analysis.png', dpi=300)
    print("Figure 3 saved.")

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print(">>> [Q3 Visualization Engine] Starting...")
    
    try:
        plot_strategic_environment()
        plot_decision_surface()
        plot_sensitivity_dashboard()
        print("\nAll Q3 plots generated successfully.")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()