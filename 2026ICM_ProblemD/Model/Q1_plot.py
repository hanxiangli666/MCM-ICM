import pandas as pd
import numpy as np
import random
import warnings
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.width'] = 0.8
plt.rcParams['ytick.minor.width'] = 0.8
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = False

# Define a professional color palette
NATURE_PALETTE = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

class WNBA_Optimizer_Pro_Max_Vis:
    def __init__(self, data_path="."):
        self.data_path = data_path
        self.salary_cap = 1507100  # 2025 Salary Cap
        self.roster_min = 11
        self.roster_max = 12
        
        self.package_deals = {
            "A'ja Wilson": "Chelsea Gray",
            "Breanna Stewart": "Sabrina Ionescu",
            "Napheesa Collier": "Kayla McBride"
        }
        
        self.load_and_process_data()

    def get_path(self, filename):
        return os.path.join(self.data_path, filename)

    def load_and_process_data(self):
        print(">>> [Pro-Insight 2.0] Initializing Data Engine...")
        try:
            # 1. Load Salaries
            salaries_file = "player_salaries_2025.xlsx - Sheet1.csv"
            if not os.path.exists(self.get_path(salaries_file)):
                 salaries_file = "player_salaries_2025.xlsx"
            
            try:
                df_salaries = pd.read_csv(self.get_path(salaries_file))
            except:
                df_salaries = pd.read_excel(self.get_path(salaries_file))

            if df_salaries['salary_2025'].dtype == 'O':
                df_salaries['salary_2025'] = df_salaries['salary_2025'].astype(str).str.replace(r'[$,]', '', regex=True)
            df_salaries['salary_2025'] = pd.to_numeric(df_salaries['salary_2025'], errors='coerce').fillna(76297)
            
            # 2. Load Stats
            stats_file = "30_MASTER_PLAYER_GAME.xlsx - Sheet1.csv"
            if not os.path.exists(self.get_path(stats_file)):
                 stats_file = "30_MASTER_PLAYER_GAME.xlsx"
            try:
                df_stats = pd.read_csv(self.get_path(stats_file))
            except:
                df_stats = pd.read_excel(self.get_path(stats_file))
            
            numeric_cols = ['minutes', 'points', 'rebounds', 'assists', 'turnovers', 'plus_minus']
            for col in numeric_cols:
                if col in df_stats.columns:
                    df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')
            df_stats = df_stats[df_stats['season'] == 2024]

            player_stats = df_stats.groupby('player').agg({
                'minutes': 'sum', 'points': 'sum', 'rebounds': 'sum',
                'assists': 'sum', 'turnovers': 'sum', 'plus_minus': 'mean'
            }).reset_index()
            
            if 'pos' in df_stats.columns:
                pos_df = df_stats.groupby('player')['pos'].first().reset_index()
                player_stats = pd.merge(player_stats, pos_df, on='player')
            
            player_stats = player_stats[player_stats['minutes'] > 50]
            
            # --- Capture Raw Stats BEFORE Normalization ---
            player_stats['per_raw'] = (player_stats['points'] + player_stats['rebounds'] + 
                                       player_stats['assists'] - player_stats['turnovers']) / player_stats['minutes']
            
            self.raw_stats = player_stats.copy() 

            # Normalize PER
            p_min, p_max = player_stats['per_raw'].min(), player_stats['per_raw'].max()
            player_stats['per_norm'] = (player_stats['per_raw'] - p_min) / (p_max - p_min)

            # 3. Merge
            merged = pd.merge(player_stats, df_salaries, on='player', how='left')
            
            # Manual Patch
            manual_corrections = {
                "Diana Taurasi": {"salary": 134500, "years": 20},
                "Nneka Ogwumike": {"salary": 169500, "years": 12},
                "Aliyah Boston": {"salary": 78000, "years": 2, "young": 1},
                "Tina Charles": {"salary": 130000, "years": 13},
                "DeWanna Bonner": {"salary": 200000, "years": 15},
                "Jonquel Jones": {"salary": 195000, "years": 8},
                "Allisha Gray": {"salary": 185000, "years": 7},
                "Cheyenne Parker": {"salary": 175000, "years": 9},
                "Dearica Hamby": {"salary": 169000, "years": 9},
                "Shakira Austin": {"salary": 91000, "years": 3, "young": 1},
                "Ariel Atkins": {"salary": 200000, "years": 7},
                "Kamilla Cardoso": {"salary": 76535, "years": 1, "young": 1},
                "Rickea Jackson": {"salary": 76535, "years": 1, "young": 1},
                "Angel Reese": {"salary": 73439, "years": 1, "young": 1}
            }

            for name, data in manual_corrections.items():
                mask = merged['player'] == name
                if mask.any():
                    merged.loc[mask, 'salary_2025'] = data['salary']
                    if 'years_of_service' in merged.columns:
                        merged.loc[mask, 'years_of_service'] = data.get('years', 5)
                    if 'young' in data:
                        merged.loc[mask, 'draft_year'] = 2024 

            merged['salary_2025'] = merged['salary_2025'].fillna(76297)
            
            if 'position' in merged.columns:
                merged['position'] = merged['position'].fillna(merged.get('pos', 'G'))
            else:
                merged['position'] = merged.get('pos', 'G')

            merged['years_of_service'] = pd.to_numeric(merged.get('years_of_service', 3), errors='coerce').fillna(3)
            merged['injury_prob'] = 0.05 + (merged['years_of_service'] * 0.015)
            merged['injury_prob'] = merged['injury_prob'].clip(0.05, 0.25)

            merged['draft_year'] = pd.to_numeric(merged.get('draft_year', 2018), errors='coerce').fillna(2018)
            merged['is_young_talent'] = (merged['draft_year'] >= 2023).astype(int)
            merged['fame'] = (merged['salary_2025'] / merged['salary_2025'].max())

            np.random.seed(42)
            merged['plays_overseas'] = merged['salary_2025'].apply(lambda s: 1 if (s < 150000 and random.random() < 0.6) else 0)
            merged['chemistry'] = np.random.uniform(-0.1, 0.15, len(merged))

            self.player_pool = merged[['player', 'position', 'salary_2025', 'per_norm', 
                                     'fame', 'chemistry', 'injury_prob', 'plays_overseas', 'is_young_talent']]
            
        except Exception as e:
            print(f"Error in data loading: {e}")
            self.player_pool = pd.DataFrame()

    def simulate_season_performance(self, roster, n_sims=50):
        profits = []
        base_per = roster['per_norm'].values
        salaries = roster['salary_2025'].sum()
        fame_sum = roster['fame'].sum()
        young_count = roster['is_young_talent'].sum()
        future_bonus = young_count * 75000 
        
        for _ in range(n_sims):
            health_status = (np.random.rand(len(roster)) > roster['injury_prob']).astype(float)
            health_status = np.where(health_status == 0, 0.2, 1.0)
            fatigue_factor = np.where(roster['plays_overseas'] == 1, 0.9, 1.0)
            perf_fluctuation = np.random.normal(1.0, 0.1, len(roster))
            sim_per = base_per * health_status * fatigue_factor * perf_fluctuation
            team_per = np.sum(sim_per)
            win_rate = 1 / (1 + np.exp(-1.5 * (team_per - 5.0)))
            
            ticket_rev = 14000 * (1 + (win_rate - 0.5)) * 60 * 20
            merch_rev = 500000 * (1 + np.log(1 + fame_sum))
            sponsor_rev = 2000000 * (1 + win_rate)
            total_rev = ticket_rev + merch_rev + sponsor_rev
            profit = total_rev - salaries - 3000000
            profits.append(profit)
            
        profits = np.array(profits)
        expected_profit = np.mean(profits)
        cutoff = int(n_sims * 0.10)
        if cutoff < 1: cutoff = 1
        worst_profits = np.sort(profits)[:cutoff]
        cvar_profit = np.mean(worst_profits)
        return expected_profit, cvar_profit, future_bonus, profits

    def objective_function(self, individual):
        roster_idx, _, _ = individual
        roster = self.player_pool.iloc[roster_idx]
        if len(roster) < self.roster_min or len(roster) > self.roster_max: return -1e9
        if roster['salary_2025'].sum() > self.salary_cap: return -1e9
        
        roster_names = set(roster['player'].values)
        nepotism_penalty = 0
        for star, partner in self.package_deals.items():
            if star in roster_names and partner not in roster_names:
                nepotism_penalty += 5000000 

        exp_profit, cvar_profit, future_bonus, _ = self.simulate_season_performance(roster, n_sims=20)
        risk_penalty = (exp_profit - cvar_profit) * 0.5
        final_score = exp_profit - risk_penalty + future_bonus - nepotism_penalty
        return final_score

    def repair_roster(self, roster_idx):
        current_names = self.player_pool.iloc[roster_idx]['player'].values
        for star, partner in self.package_deals.items():
            if star in current_names and partner not in current_names:
                partner_rows = self.player_pool[self.player_pool['player'] == partner]
                if not partner_rows.empty:
                    p_idx = partner_rows.index[0]
                    if p_idx not in roster_idx: roster_idx.append(p_idx)
        
        current_roster = self.player_pool.iloc[roster_idx]
        total_sal = current_roster['salary_2025'].sum()
        attempts = 0
        while total_sal > self.salary_cap and attempts < 20:
            score_metric = (current_roster['per_norm'] / (current_roster['salary_2025'] + 1)) * \
                           (1 + 0.5 * current_roster['is_young_talent']) * \
                           (1 + 0.5 * current_roster['fame'])
            drop_idx = score_metric.idxmin()
            if drop_idx in roster_idx: roster_idx.remove(drop_idx)
            
            if len(roster_idx) < 11:
                cheap_players = self.player_pool[
                    (self.player_pool['salary_2025'] < 80000) & 
                    (~self.player_pool.index.isin(roster_idx))
                ]
                if not cheap_players.empty:
                    add_idx = cheap_players['per_norm'].idxmax()
                    roster_idx.append(add_idx)

            current_roster = self.player_pool.iloc[roster_idx]
            total_sal = current_roster['salary_2025'].sum()
            attempts += 1
            
        return roster_idx[:12]

    def run_genetic_algorithm(self, generations=40, population_size=40):
        print(f">>> Running Genetic Algorithm...")
        population = []
        for _ in range(population_size):
            r_idx = list(np.random.choice(self.player_pool.index, 12, replace=False))
            r_idx = self.repair_roster(r_idx)
            population.append([r_idx, 0, 0])

        best_solution = None
        best_score = -np.inf
        history = []
        
        for gen in range(generations):
            scores = []
            for indiv in population:
                s = self.objective_function(indiv)
                scores.append(s)
                if s > best_score:
                    best_score = s
                    best_solution = deepcopy(indiv)
            
            history.append(best_score)
            sorted_idx = np.argsort(scores)[::-1]
            new_pop = [population[i] for i in sorted_idx[:int(population_size*0.2)]]
            
            while len(new_pop) < population_size:
                parent = population[random.choice(sorted_idx[:20])]
                child_roster = deepcopy(parent[0])
                if random.random() < 0.4:
                    if len(child_roster) > 0:
                        child_roster.remove(random.choice(child_roster))
                        avail = list(set(self.player_pool.index) - set(child_roster))
                        if avail: child_roster.append(random.choice(avail))
                child_roster = self.repair_roster(child_roster)
                new_pop.append([child_roster, 0, 0])
            population = new_pop
            if gen % 10 == 0:
                print(f"Gen {gen}: Best Score = {best_score:,.0f}")
            
        return best_solution, history

# --- VISUALIZATION FUNCTION 1: Pre-Modeling EDA (Original Figure 1) ---
def plot_pre_modeling(optimizer):
    df = optimizer.player_pool
    
    # Setup Figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. PER vs Salary (Scatter)
    ax1 = fig.add_subplot(gs[0, 0])
    
    df['Status'] = 'Veteran'
    df.loc[df['is_young_talent'] == 1, 'Status'] = 'Rookie Scale'
    
    sns.scatterplot(
        data=df, x='salary_2025', y='per_norm', hue='Status', style='Status',
        palette=[NATURE_PALETTE[0], NATURE_PALETTE[3]], s=80, alpha=0.8, ax=ax1, edgecolor='w', linewidth=0.5
    )
    
    notable_players = ["Caitlin Clark", "Breanna Stewart", "A'ja Wilson"]
    for name in notable_players:
        player_data = df[df['player'] == name]
        if not player_data.empty:
            ax1.text(player_data['salary_2025'].values[0] + 5000, player_data['per_norm'].values[0], name, fontsize=8, fontweight='bold', color='#333333')

    ax1.set_title("a | Market Efficiency: Player Utility vs. Cost", loc='left', fontweight='bold')
    ax1.set_xlabel("Annual Salary (USD)")
    ax1.set_ylabel("Normalized Player Efficiency (PER)")
    ax1.legend(frameon=False, loc='lower right')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

    # 2. Injury Risk
    ax2 = fig.add_subplot(gs[0, 1])
    years_data = (df['injury_prob'] - 0.05) / 0.015
    sns.regplot(x=years_data, y=df['injury_prob'], ax=ax2, scatter_kws={'alpha':0.4, 'color': NATURE_PALETTE[2], 's': 50}, line_kws={'color': '#333333', 'linewidth': 1.5})
    ax2.set_title("b | Injury Risk Progression Model", loc='left', fontweight='bold')
    ax2.set_xlabel("Years of Service (Experience)")
    ax2.set_ylabel("modeled Injury Probability P(Inj)")
    ax2.set_ylim(0, 0.3)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # 3. Fame vs Salary
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(df['salary_2025'], df['fame'], c=df['per_norm'], cmap='viridis', s=60, alpha=0.9, edgecolors='w', linewidth=0.3)
    cbar = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Performance (PER)', rotation=270, labelpad=15)
    ax3.set_title("c | Commercial Value Correlation", loc='left', fontweight='bold')
    ax3.set_xlabel("Annual Salary (USD)")
    ax3.set_ylabel("Digital Fame Score (Index)")
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

    # 4. Salary Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=df, x='salary_2025', fill=True, color=NATURE_PALETTE[4], alpha=0.6, linewidth=1.5, ax=ax4)
    ax4.set_title("d | League-Wide Salary Distribution", loc='left', fontweight='bold')
    ax4.set_xlabel("Annual Salary (USD)")
    ax4.set_ylabel("Density")
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax4.set_yticks([]) 

    plt.tight_layout()
    plt.savefig('Figure_1_Data_Exploration.pdf', bbox_inches='tight')
    plt.savefig('Figure_1_Data_Exploration.png', dpi=300)
    print("Figure 1 (Basic EDA) saved.")

# --- VISUALIZATION FUNCTION 2: Post-Modeling Results (Original Figure 2) ---
def plot_post_modeling(optimizer, best_solution, history):
    r_idx = best_solution[0]
    roster = optimizer.player_pool.iloc[r_idx]
    
    exp, cvar, bonus, profits = optimizer.simulate_season_performance(roster, n_sims=1000)
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.4)
    
    # 1. Convergence
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(history, color=NATURE_PALETTE[0], linewidth=2, label='Best Fitness')
    if len(history) > 5:
        smoothed = pd.Series(history).rolling(window=5).mean()
        ax1.plot(smoothed, color=NATURE_PALETTE[0], linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.set_title("a | Genetic Algorithm Convergence", loc='left', fontweight='bold')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Objective Score (RAROC)")
    ax1.grid(True, which='major', axis='y', linestyle=':', color='gray', alpha=0.5)
    
    # 2. Risk Dist
    ax2 = fig.add_subplot(gs[0, 2])
    sns.histplot(profits, kde=True, stat='probability', color=NATURE_PALETTE[1], edgecolor='white', linewidth=0.5, alpha=0.6, ax=ax2)
    ax2.axvline(exp, color='#333333', linestyle='-', linewidth=1.5, label='Mean')
    ax2.axvline(cvar, color=NATURE_PALETTE[0], linestyle='--', linewidth=1.5, label='CVaR (10%)')
    ax2.set_title("b | Stochastic Profit Projection", loc='left', fontweight='bold')
    ax2.set_xlabel("Projected Net Profit (USD)")
    ax2.set_ylabel("Probability")
    ax2.legend(frameon=False, fontsize=8)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # 3. Allocation
    ax3 = fig.add_subplot(gs[1, 0])
    def categorize(row):
        if row['is_young_talent'] == 1: return 'Rookie Scale'
        if row['salary_2025'] > 180000: return 'Superstar'
        return 'Role Player'
    roster['Type'] = roster.apply(categorize, axis=1)
    salary_counts = roster.groupby('Type')['salary_2025'].sum()
    total_salary = salary_counts.sum()
    colors = [NATURE_PALETTE[3], NATURE_PALETTE[4], NATURE_PALETTE[5]]
    y_pos = [0]
    left = 0
    for i, (idx, val) in enumerate(salary_counts.items()):
        ax3.barh(y_pos, val, left=left, height=0.5, color=colors[i], label=idx, edgecolor='white')
        pct = (val / total_salary) * 100
        if pct > 5:
            ax3.text(left + val/2, 0, f'{pct:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        left += val
    ax3.set_yticks([])
    ax3.set_title("c | Salary Cap Allocation Strategy", loc='left', fontweight='bold')
    ax3.set_xlabel("Total Salary Spend (USD)")
    ax3.set_xlim(0, 1600000)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # 4. Metrics
    ax4 = fig.add_subplot(gs[1, 1:])
    metrics = ['PER (Norm)', 'Injury Risk', 'Fame Score']
    avg_pool = optimizer.player_pool[['per_norm', 'injury_prob', 'fame']].mean()
    avg_roster = roster[['per_norm', 'injury_prob', 'fame']].mean()
    x = np.arange(len(metrics))
    width = 0.35
    rects1 = ax4.bar(x - width/2, avg_pool, width, label='League Avg', color='#B0B0B0')
    rects2 = ax4.bar(x + width/2, avg_roster, width, label='Optimal Roster', color=NATURE_PALETTE[0])
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_title("d | Performance Delta Analysis", loc='left', fontweight='bold')
    ax4.legend(frameon=False)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax4.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    autolabel(rects1)
    autolabel(rects2)
    sns.despine(ax=ax4)
    sns.despine(ax=ax3, left=True)

    plt.tight_layout()
    plt.savefig('Figure_2_Optimization_Results.pdf', bbox_inches='tight')
    plt.savefig('Figure_2_Optimization_Results.png', dpi=300)
    print("Figure 2 (Results) saved.")

# --- VISUALIZATION FUNCTION 3: Extended EDA (New Figure 3) ---
def plot_pre_modeling_extended(optimizer):
    df = optimizer.player_pool
    raw_stats = optimizer.raw_stats 
    
    fig = plt.figure(figsize=(15, 6)) 
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(raw_stats['per_raw'], color=NATURE_PALETTE[0], fill=True, alpha=0.3, label='Raw PER', ax=ax1)
    ax1_twin = ax1.twiny() 
    sns.kdeplot(df['per_norm'], color=NATURE_PALETTE[3], fill=True, alpha=0.3, label='Norm PER (0-1)', ax=ax1_twin)
    
    ax1.set_title("a | Metric Standardization: Distribution Shift", loc='left', fontweight='bold')
    ax1.set_xlabel("Raw PER Value")
    ax1_twin.set_xlabel("Normalized Score $S_{it}$")
    ax1.set_ylabel("Density")
    legend_elements = [Patch(facecolor=NATURE_PALETTE[0], alpha=0.3, label='Raw PER Distribution'), Patch(facecolor=NATURE_PALETTE[3], alpha=0.3, label='Standardized $S_{it}$')]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=8)

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    df['Star_Tier'] = pd.cut(df['fame'], bins=[0, 0.3, 0.7, 1.0], labels=['Role', 'Starter', 'Superstar'])
    sns.boxplot(data=df, x='Star_Tier', y='fame', palette=[NATURE_PALETTE[5], NATURE_PALETTE[1], NATURE_PALETTE[0]], width=0.5, ax=ax2)
    sns.stripplot(data=df, x='Star_Tier', y='fame', color='#333333', size=2, alpha=0.6, ax=ax2)
    ax2.set_title("b | Commercial Index ($F_{it}$) Stratification", loc='left', fontweight='bold')
    ax2.set_xlabel("Player Tier")
    ax2.set_ylabel("Commercial Appeal Score $F_{it}$")
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    # Panel C
    ax3 = fig.add_subplot(gs[0, 2])
    top_salaries = df.sort_values('salary_2025', ascending=False).head(20)['salary_2025'].values
    cumulative_salary = np.cumsum(top_salaries)
    x_rank = np.arange(1, 21)
    ax3.plot(x_rank, cumulative_salary, color=NATURE_PALETTE[2], linewidth=2, marker='o', markersize=4, label='Cumulative Top Salaries')
    cap_limit = 1507100
    ax3.axhline(cap_limit, color=NATURE_PALETTE[0], linestyle='--', linewidth=1.5, label='2025 Hard Cap ($1.5M)')
    crossing_indices = np.where(cumulative_salary > cap_limit)[0]
    if len(crossing_indices) > 0:
        crossing_idx = crossing_indices[0]
        ax3.plot(x_rank[crossing_idx], cumulative_salary[crossing_idx], 'x', color='black', markersize=10, markeredgewidth=2)
        ax3.text(x_rank[crossing_idx]+1, cap_limit+100000, f'Cap Hit at Player #{crossing_idx+1}', fontsize=9, color='#333333')

    ax3.set_title("c | Salary Cap Constraint Analysis", loc='left', fontweight='bold')
    ax3.set_xlabel("Number of Top-Tier Players Signed")
    ax3.set_ylabel("Cumulative Salary (USD)")
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax3.legend(frameon=False, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('Figure_3_Data_Processing_EDA.pdf', bbox_inches='tight')
    plt.savefig('Figure_3_Data_Processing_EDA.png', dpi=300)
    print("Figure 3 (Extended EDA) saved.")

# --- Execution ---
if __name__ == "__main__":
    opt = WNBA_Optimizer_Pro_Max_Vis()
    best_sol, history = opt.run_genetic_algorithm(generations=40, population_size=40)
    
    # 1. Figure 1: Basic EDA
    plot_pre_modeling(opt)
    
    # 2. Figure 3: Extended EDA (Added Request)
    plot_pre_modeling_extended(opt)
    
    # 3. Figure 2: Results
    plot_post_modeling(opt, best_solution=best_sol, history=history)