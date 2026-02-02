"""
==============================================================================
Script Name: Q2_plot.py
Description: Visualization Suite for Mixed-Variable Roster Optimizer (Q2)
             Adapts the visualization style of Q1 to the Q2 model results.
Author: Assistant
==============================================================================
"""

import pandas as pd
import numpy as np
import random
import math
import os
import re
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# ==========================================
# 0. Visualization Style Settings (From Q1)
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
# 1. Configuration (From Q2)
# ==========================================
CONFIG = {
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    'SALARY_CAP': 1500000,
    'ROSTER_MIN': 11,
    'ROSTER_MAX': 12,
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
    'SALARY_MIN_LEAGUE': 64154,
    'SALARY_MAX_LEAGUE': 241984,
    'TOTAL_GAME_MINUTES': 200,
    'PLAYER_MAX_MINUTES': 38.0,
    'PLAYER_MIN_MINUTES': 5.0 ,
    'N_HIGHRISK_MAX': 2,
    'N_POTENTIAL_MIN': 2
}

# ==========================================
# 2. Data Processor (From Q2)
# ==========================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file
        self.salary_file = salary_file

    def normalize_name(self, series):
        return series.astype(str).str.lower().str.strip().apply(lambda x: re.sub(r"[^a-z0-9]", "", x))

    def load_and_process(self, target_team_name="Indiana Fever"):
        print(f">>> [DataEngine] Loading data for: {target_team_name}...")
        
        try:
            # Handle potential file location differences
            s_file = self.stats_file if os.path.exists(self.stats_file) else os.path.basename(self.stats_file)
            sal_file = self.salary_file if os.path.exists(self.salary_file) else os.path.basename(self.salary_file)

            read_func = pd.read_excel if s_file.endswith('.xlsx') else pd.read_csv
            stats_raw = read_func(s_file)
            
            read_func = pd.read_excel if sal_file.endswith('.xlsx') else pd.read_csv
            salary_raw = read_func(sal_file)
        except Exception as e:
            print(f"!!! File read error: {e}")
            # Mock data generation if files missing (for demonstration safety)
            return pd.DataFrame() 

        stats_raw.columns = [c.lower().strip() for c in stats_raw.columns]
        salary_raw.columns = [c.lower().strip() for c in salary_raw.columns]
        
        if 'season' in stats_raw.columns:
            target_season = stats_raw['season'].max()
            stats = stats_raw[stats_raw['season'] == target_season].copy()
        else:
            stats = stats_raw.copy()

        stat_cols = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy']
        for c in stat_cols:
            if c not in stats.columns: stats[c] = 0.0
            stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)

        metrics = stats.groupby('player').agg({
            'points': 'mean', 'rebounds': 'mean', 'assists': 'mean', 
            'plus_minus': 'mean', 'minutes': 'mean', 'attendance': 'mean',
            'clutch_proxy': 'mean', 'team': 'last' 
        }).reset_index()

        salary = salary_raw.copy()
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce')
        salary['years_of_service'] = pd.to_numeric(salary.get('years_of_service', 0), errors='coerce').fillna(1)
        
        metrics['key'] = self.normalize_name(metrics['player'])
        salary['key'] = self.normalize_name(salary['player'])
        
        df = pd.merge(metrics, salary[['key', 'salary_2025', 'years_of_service', 'position', 'team']], 
                      on='key', how='outer', suffixes=('', '_sal'))
        
        df['player'] = df['player'].fillna(salary['player'])
        df['team'] = df['team'].fillna(df['team_sal']).fillna('Free Agent')
        
        # Manual fixes (Condensed)
        manual_fixes = {
            'aliyah boston': {'salary': 99000, 'pos': 'C', 'team': target_team_name, 'pts': 14.5, 'reb': 8.4},
            'caitlin clark': {'salary': 78066, 'pos': 'G', 'team': target_team_name, 'pts': 19.2, 'ast': 8.2},
            'kelsey mitchell': {'salary': 200000, 'pos': 'G', 'team': target_team_name, 'pts': 18.0},
        }

        for name, data in manual_fixes.items():
            key = self.normalize_name(pd.Series([name]))[0]
            mask = df['key'] == key
            if mask.any():
                idx = df[mask].index
                if 'salary' in data: df.loc[idx, 'salary_2025'] = data['salary']
                if 'pos' in data: df.loc[idx, 'position'] = data['pos']
                if 'team' in data: df.loc[idx, 'team'] = data['team']
            else:
                new_row = {
                    'player': name.title(), 'key': key, 'team': data.get('team', target_team_name),
                    'salary_2025': data.get('salary', 76297), 'position': data.get('pos', 'G'),
                    'minutes': 20.0, 'points': data.get('pts', 8.0),
                    'years_of_service': 3, 'clutch_proxy': 0.5, 'plus_minus': 0
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        fill_zeros = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy', 'years_of_service']
        for c in fill_zeros:
            if c not in df.columns: df[c] = 0.0
            df[c] = df[c].fillna(0.0)
        df['salary_2025'] = df['salary_2025'].fillna(76297)
        if 'position' not in df.columns: df['position'] = 'G'
        df['position'] = df['position'].fillna('G')
        df['pos_mapped'] = df['position'].apply(self._map_pos)

        df['S_min'] = df['salary_2025'].apply(lambda x: max(CONFIG['SALARY_MIN_LEAGUE'], x * 0.8))
        df['S_max'] = df['salary_2025'].apply(lambda x: min(CONFIG['SALARY_MAX_LEAGUE'], x * 1.2))
        mask_super = df['S_max'] > 200000
        df.loc[mask_super, 'S_min'] = df.loc[mask_super, 'salary_2025'] * 0.9

        df = self._calculate_scores(df)

        target_clean = self.normalize_name(pd.Series([target_team_name]))[0]
        df['team_clean'] = self.normalize_name(df['team'])
        if 'fever' in target_clean:
            mask_home = df['team_clean'].str.contains('fever') | df['team_clean'].str.contains('ind')
            df.loc[mask_home, 'team_clean'] = target_clean
            df.loc[mask_home, 'team'] = target_team_name

        return df

    def _map_pos(self, p):
        p = str(p).upper()
        if 'C' in p: return 'C'
        if 'F' in p: return 'F'
        return 'G'

    def _calculate_scores(self, df):
        def norm(col):
            min_v, max_v = col.min(), col.max()
            if max_v == min_v: return 0.5
            return (col - min_v) / (max_v - min_v)

        df['ViA'] = (0.3 * norm(df['points']) + 0.2 * norm(df['rebounds']) + 
                     0.2 * norm(df['assists']) + 0.15 * norm(df['plus_minus']) + 
                     0.15 * norm(df['clutch_proxy']))
        
        df['ViB'] = 0.5 * norm(df['salary_2025']) + 0.5 * norm(df['attendance'])
        mask_star = (df['salary_2025'] > 200000) | (df['player'].str.contains('Clark', na=False, case=False))
        df.loc[mask_star, 'ViB'] *= 1.5
        df['ViB'] = df['ViB'].clip(0, 1)

        df['ViH_base'] = 1 - (0.5 * norm(df['minutes']) * 0.3) 
        df['ViP'] = 1 - norm(df['years_of_service'])
        
        df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                         CONFIG['W_H'] * df['ViH_base'] + CONFIG['W_P'] * df['ViP'])
        df['Vi_base'] = df['Vi_base'].fillna(df['Vi_base'].mean())
        return df

# ==========================================
# 3. Solver (From Q2 + Modifications)
# ==========================================
class Individual:
    def __init__(self, n_total):
        self.dna_decisions = np.zeros(n_total, dtype=int)
        self.dna_salary = np.random.rand(n_total)
        self.dna_time = np.random.rand(n_total)
        self.objectives = None
        self.rank = 0
        self.crowding_dist = 0
        self.violation = 0
        self.z_score = -1e9
        self.decoded_roster = None

class StrategicSolver:
    def __init__(self, pool, current_team_name):
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        if 'fever' in target:
            mask_home = pool['team_clean'].str.contains('fever') | pool['team_clean'].str.contains('ind')
        else:
            mask_home = pool['team_clean'] == target

        self.current_roster = pool[mask_home].copy().reset_index(drop=True)
        self.market_pool = pool[~mask_home].sort_values('Vi_base', ascending=False).head(80).reset_index(drop=True)
        self.full_pool = pd.concat([self.current_roster, self.market_pool]).reset_index(drop=True)
        self.n_total = len(self.full_pool)
        self.n_current = len(self.current_roster)

    def evaluate(self, ind):
        indices = np.where(ind.dna_decisions == 1)[0]
        if len(indices) == 0:
            ind.objectives = [1e9, 1e9]
            ind.violation = 1e9
            return

        roster = self.full_pool.iloc[indices].copy()
        s_factors = ind.dna_salary[indices]
        roster['opt_salary'] = roster['S_min'] + s_factors * (roster['S_max'] - roster['S_min'])
        
        t_weights = ind.dna_time[indices]
        total_weight = np.sum(t_weights)
        if total_weight == 0:
            roster['opt_minutes'] = CONFIG['TOTAL_GAME_MINUTES'] / len(roster)
        else:
            roster['opt_minutes'] = (t_weights / total_weight) * CONFIG['TOTAL_GAME_MINUTES']
        
        roster['opt_minutes'] = roster['opt_minutes'].clip(CONFIG['PLAYER_MIN_MINUTES'], CONFIG['PLAYER_MAX_MINUTES'])
        scale_factor = CONFIG['TOTAL_GAME_MINUTES'] / roster['opt_minutes'].sum()
        roster['opt_minutes'] *= scale_factor
        
        # Calculations
        TC = roster['opt_salary'].sum()
        time_ratio = roster['opt_minutes'] / 40.0 
        roster['ViH_dynamic'] = 1.0 - (time_ratio * 0.8) 
        roster['Vi_contribution'] = roster['Vi_base'] * time_ratio
        
        avg_Vi = roster['Vi_contribution'].sum()
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        Risk = (roster['opt_salary'] * (1 - roster['ViH_dynamic'])).sum() * CONFIG['RISK_FACTOR']
        
        # Violation
        violation = 0
        count = len(roster)
        if count < CONFIG['ROSTER_MIN']: violation += (CONFIG['ROSTER_MIN'] - count) * 1000
        if count > CONFIG['ROSTER_MAX']: violation += (count - CONFIG['ROSTER_MAX']) * 1000
        if TC > CONFIG['SALARY_CAP']: violation += (TC - CONFIG['SALARY_CAP']) / 100
        
        n_high_risk = len(roster[roster['ViH_base'] < 0.6])
        if n_high_risk > CONFIG['N_HIGHRISK_MAX']: violation += (n_high_risk - CONFIG['N_HIGHRISK_MAX']) * 5000
        
        n_acquired = len(roster[roster.index >= self.n_current])
        n_retained = len(roster[roster.index < self.n_current])
        n_waived = self.n_current - n_retained
        if n_acquired > n_waived: violation += (n_acquired - n_waived) * 10000
        if n_retained < 4: violation += (4 - n_retained) * 2000

        ind.violation = violation
        ind.decoded_roster = roster

        # Objectives
        pos_raw_score = 0
        pos_counts = roster['pos_mapped'].value_counts()
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: pos_raw_score += 1
            else: pos_raw_score -= 0.5 * abs(c - (min_p + max_p) / 2)
        PosFit = max(0.0, min(1.0, pos_raw_score * 0.3 + 0.5))

        avg_pm = roster['plus_minus'].mean()
        StyleFit = 1.0 / (1.0 + math.exp(-0.5 * avg_pm))
        ViF = 0.5 * PosFit + 0.5 * StyleFit
 
        term_financial = (TR - Risk) / 10000 
        performance_score = (CONFIG['MU_1'] * term_financial + 
                             CONFIG['MU_2'] * avg_Vi * 20 + 
                             CONFIG['MU_3'] * ViF * 50)
        
        ind.objectives = [TC, -performance_score]
        ind.z_score = performance_score - violation

    def fast_nondominated_sort(self, population):
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in population:
                if p.violation < q.violation: dominates = True
                elif p.violation > q.violation: dominates = False
                else:
                    dominates = ((p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and
                                 (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]))
                if dominates: p.dominated_solutions.append(q)
                elif self._check_dominated(q, p): p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def _check_dominated(self, p, q):
        if p.violation < q.violation: return True
        if p.violation > q.violation: return False
        return ((p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and
                (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]))

    def calculate_crowding_distance(self, front):
        l = len(front)
        if l == 0: return
        for p in front: p.crowding_dist = 0
        for m in range(2):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_dist = float('inf')
            front[-1].crowding_dist = float('inf')
            obj_min = front[0].objectives[m]
            obj_max = front[-1].objectives[m]
            if obj_max == obj_min: continue
            for i in range(1, l-1):
                front[i].crowding_dist += (front[i+1].objectives[m] - front[i-1].objectives[m]) / (obj_max - obj_min)

    def crowd_comparison(self, p1, p2):
        if p1.rank < p2.rank: return True
        if p1.rank == p2.rank and p1.crowding_dist > p2.crowding_dist: return True
        return False

    def mutate(self, ind, rate=0.1):
        child = deepcopy(ind)
        if random.random() < 0.3:
            ones = np.where(child.dna_decisions == 1)[0]
            zeros = np.where(child.dna_decisions == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                child.dna_decisions[np.random.choice(ones)] = 0
                child.dna_decisions[np.random.choice(zeros)] = 1
        if random.random() < rate:
            idx = random.randint(0, self.n_total - 1)
            child.dna_decisions[idx] = 1 - child.dna_decisions[idx]
        
        sigma = 0.1
        mask_s = np.random.rand(self.n_total) < rate
        noise_s = np.random.randn(self.n_total) * sigma
        child.dna_salary[mask_s] += noise_s[mask_s]
        child.dna_salary = np.clip(child.dna_salary, 0, 1)
        
        mask_t = np.random.rand(self.n_total) < rate
        noise_t = np.random.randn(self.n_total) * sigma
        child.dna_time[mask_t] += noise_t[mask_t]
        child.dna_time = np.clip(child.dna_time, 0, 1)
        return child

    def solve(self, generations=50, pop_size=50):
        # Initialize
        population = []
        for _ in range(pop_size):
            ind = Individual(self.n_total)
            target_size = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            n_keep = random.randint(4, min(self.n_current, target_size))
            n_buy = target_size - n_keep
            idxs_keep = np.random.choice(self.n_current, n_keep, replace=False)
            ind.dna_decisions[idxs_keep] = 1
            if n_buy > 0:
                market_indices = np.arange(self.n_current, self.n_total)
                market_probs = self.full_pool.iloc[market_indices]['Vi_base'].values
                market_probs = market_probs / market_probs.sum()
                idxs_buy = np.random.choice(market_indices, n_buy, replace=False, p=market_probs)
                ind.dna_decisions[idxs_buy] = 1
            self.evaluate(ind)
            population.append(ind)

        # Evolution
        history = []
        print(f">>> [Solver] Starting Evolution ({generations} gens)...")
        for gen in range(generations):
            fronts = self.fast_nondominated_sort(population)
            for front in fronts: self.calculate_crowding_distance(front)
            
            offspring = []
            while len(offspring) < pop_size:
                p1 = random.choice(population)
                p2 = random.choice(population)
                parent = p1 if self.crowd_comparison(p1, p2) else p2
                child = self.mutate(parent, rate=0.1)
                self.evaluate(child)
                offspring.append(child)
            
            combined = population + offspring
            fronts = self.fast_nondominated_sort(combined)
            new_pop = []
            for front in fronts:
                self.calculate_crowding_distance(front)
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda x: x.crowding_dist, reverse=True)
                    new_pop.extend(front[:pop_size - len(new_pop)])
                    break
            population = new_pop
            
            # Record Best Z
            best_z = max(p.z_score for p in population)
            history.append(best_z)
            if gen % 10 == 0:
                print(f"    Gen {gen}: Best Z-Score = {best_z:.4f}")

        # Best Solution
        pareto_front = [p for p in population if p.rank == 0 and p.violation == 0]
        if not pareto_front:
            best_ind = max(population, key=lambda x: x.z_score)
        else:
            best_ind = max(pareto_front, key=lambda x: x.z_score)
            
        return best_ind, history

    def simulate_profit_distribution(self, roster, n_sims=500):
        """Mock Monte Carlo simulation for visualization purposes"""
        base_TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        base_Risk = (roster['opt_salary'] * (1 - roster['ViH_dynamic'])).sum() * CONFIG['RISK_FACTOR']
        base_Cost = roster['opt_salary'].sum()
        
        profits = []
        for _ in range(n_sims):
            # Add random noise to performance and health
            perf_noise = np.random.normal(1.0, 0.1)
            health_noise = np.random.uniform(0.8, 1.1)
            
            sim_TR = base_TR * perf_noise
            sim_Risk = base_Risk / health_noise
            
            # Profit = Revenue - Cost - RiskPenalty
            profit = sim_TR - base_Cost - sim_Risk
            profits.append(profit)
            
        return np.array(profits), np.mean(profits)


# ==========================================
# 4. Visualization Functions (Based on Q1)
# ==========================================
def plot_pre_modeling(solver):
    """Figure 1: Data Exploration"""
    df = solver.full_pool
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Performance vs Salary
    ax1 = fig.add_subplot(gs[0, 0])
    df['Status'] = 'Veteran'
    df.loc[df['years_of_service'] <= 3, 'Status'] = 'Rookie Scale'
    
    sns.scatterplot(
        data=df, x='salary_2025', y='ViA', hue='Status', style='Status',
        palette=[NATURE_PALETTE[0], NATURE_PALETTE[3]], s=80, alpha=0.8, ax=ax1, edgecolor='w'
    )
    
    # Label top stars
    top_stars = df.nlargest(3, 'ViA')['player'].tolist()
    for name in top_stars:
        player_data = df[df['player'] == name]
        if not player_data.empty:
            ax1.text(player_data['salary_2025'].values[0] + 2000, player_data['ViA'].values[0], name, fontsize=8)

    ax1.set_title("a | Market Efficiency: Performance (ViA) vs. Cost", loc='left', fontweight='bold')
    ax1.set_xlabel("Annual Salary (USD)")
    ax1.set_ylabel("Normalized Performance (ViA)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

    # 2. Health Risk (ViH)
    ax2 = fig.add_subplot(gs[0, 1])
    # Proxy Risk = 1 - ViH_base
    df['Risk_Score'] = 1 - df['ViH_base']
    sns.regplot(x='years_of_service', y='Risk_Score', data=df, ax=ax2, 
                scatter_kws={'alpha':0.4, 'color': NATURE_PALETTE[2]}, line_kws={'color': '#333333'})
    ax2.set_title("b | Injury Risk Proxy vs. Experience", loc='left', fontweight='bold')
    ax2.set_xlabel("Years of Service")
    ax2.set_ylabel("modeled Risk (1 - ViH)")

    # 3. Commercial Value (ViB) vs Salary
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(df['salary_2025'], df['ViB'], c=df['ViA'], cmap='viridis', s=60, alpha=0.9, edgecolors='w')
    cbar = plt.colorbar(scatter, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Performance (ViA)', rotation=270, labelpad=15)
    ax3.set_title("c | Commercial Value (ViB) Correlation", loc='left', fontweight='bold')
    ax3.set_xlabel("Annual Salary (USD)")
    ax3.set_ylabel("Commercial Score (ViB)")
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

    # 4. Salary Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=df, x='salary_2025', fill=True, color=NATURE_PALETTE[4], alpha=0.6, ax=ax4)
    ax4.set_title("d | Salary Distribution", loc='left', fontweight='bold')
    ax4.set_xlabel("Annual Salary (USD)")
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax4.set_yticks([])

    plt.tight_layout()
    plt.savefig('Figure_1_Data_Exploration_Q2.png', dpi=300)
    print("Figure 1 saved.")


def plot_post_modeling(solver, best_solution, history):
    """Figure 2: Optimization Results"""
    roster = best_solution.decoded_roster
    profits, exp_profit = solver.simulate_profit_distribution(roster)
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.4)
    
    # 1. Convergence
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(history, color=NATURE_PALETTE[0], linewidth=2)
    ax1.set_title("a | Genetic Algorithm Convergence", loc='left', fontweight='bold')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Objective Z-Score")
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)
    
    # 2. Risk Dist
    ax2 = fig.add_subplot(gs[0, 2])
    sns.histplot(profits, kde=True, color=NATURE_PALETTE[1], alpha=0.6, ax=ax2)
    ax2.axvline(exp_profit, color='#333333', linestyle='-', label='Mean')
    ax2.set_title("b | Stochastic Value Projection", loc='left', fontweight='bold')
    ax2.set_xlabel("Net Value Index")
    ax2.set_yticks([])
    
    # 3. Allocation
    ax3 = fig.add_subplot(gs[1, 0])
    def categorize(row):
        if row['years_of_service'] <= 3: return 'Rookie/Young'
        if row['salary_2025'] > 180000: return 'Superstar'
        return 'Veteran'
    roster['Type'] = roster.apply(categorize, axis=1)
    salary_counts = roster.groupby('Type')['opt_salary'].sum()
    
    # Pie chart style bar
    left = 0
    colors = [NATURE_PALETTE[3], NATURE_PALETTE[4], NATURE_PALETTE[5]]
    for i, (idx, val) in enumerate(salary_counts.items()):
        ax3.barh([0], val, left=left, height=0.5, color=colors[i%3], label=idx, edgecolor='white')
        if (val / salary_counts.sum()) > 0.05:
            ax3.text(left + val/2, 0, f'{val/salary_counts.sum():.0%}', ha='center', color='white', fontweight='bold')
        left += val
        
    ax3.set_yticks([])
    ax3.set_title("c | Salary Cap Allocation", loc='left', fontweight='bold')
    ax3.set_xlabel("Total Spend (USD)")
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # 4. Metrics Comparison
    ax4 = fig.add_subplot(gs[1, 1:])
    metrics = ['ViA (Perf)', 'ViB (Comm)', 'ViP (Poten)']
    avg_pool = solver.full_pool[['ViA', 'ViB', 'ViP']].mean()
    avg_roster = roster[['ViA', 'ViB', 'ViP']].mean()
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, avg_pool, width, label='League Avg', color='#B0B0B0')
    rects = ax4.bar(x + width/2, avg_roster, width, label='Roster Avg', color=NATURE_PALETTE[0])
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.set_title("d | Roster Advantages", loc='left', fontweight='bold')
    ax4.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig('Figure_2_Optimization_Results_Q2.png', dpi=300)
    print("Figure 2 saved.")


def plot_pre_modeling_extended(solver):
    """Figure 3: Extended EDA"""
    df = solver.full_pool
    
    fig = plt.figure(figsize=(15, 6)) 
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # Panel A: Standardization
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(df['points'], color=NATURE_PALETTE[0], fill=True, alpha=0.3, label='Raw Points', ax=ax1)
    ax1_twin = ax1.twiny() 
    sns.kdeplot(df['ViA'], color=NATURE_PALETTE[3], fill=True, alpha=0.3, label='Norm ViA', ax=ax1_twin)
    ax1.set_title("a | Metric Standardization (Points -> ViA)", loc='left', fontweight='bold')
    ax1.set_yticks([])
    
    # Panel B: Star Tier
    ax2 = fig.add_subplot(gs[0, 1])
    df['Star_Tier'] = pd.cut(df['ViB'], bins=[-0.1, 0.3, 0.7, 1.5], labels=['Role', 'Starter', 'Star'])
    sns.boxplot(data=df, x='Star_Tier', y='ViB', palette=NATURE_PALETTE, ax=ax2)
    ax2.set_title("b | Commercial Index (ViB) Stratification", loc='left', fontweight='bold')

    # Panel C: Cap Analysis
    ax3 = fig.add_subplot(gs[0, 2])
    top_salaries = df.sort_values('salary_2025', ascending=False).head(20)['salary_2025'].values
    cumulative = np.cumsum(top_salaries)
    ax3.plot(range(1, 21), cumulative, 'o-', color=NATURE_PALETTE[2])
    ax3.axhline(CONFIG['SALARY_CAP'], linestyle='--', color='red', label='Cap')
    ax3.set_title("c | Salary Cap Constraint", loc='left', fontweight='bold')
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax3.legend()

    plt.tight_layout()
    plt.savefig('Figure_3_Data_Processing_EDA_Q2.png', dpi=300)
    print("Figure 3 saved.")


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    # File Paths
    stats_file = "30_MASTER_PLAYER_GAME.xlsx"
    salary_file = "player_salaries_2025.xlsx"
    
    # 1. Load Data
    dp = DataProcessor(stats_file, salary_file)
    pool = dp.load_and_process(target_team_name="Indiana Fever")
    
    if pool is not None and not pool.empty:
        # 2. Init Solver
        solver = StrategicSolver(pool, current_team_name="Indiana Fever")
        
        # 3. Generate EDA Plots
        plot_pre_modeling(solver)
        plot_pre_modeling_extended(solver)
        
        # 4. Run Optimization
        best_ind, history = solver.solve(generations=40, pop_size=40)
        
        # 5. Generate Result Plots
        plot_post_modeling(solver, best_ind, history)
        
        print("\nAll visualizations generated successfully.")
    else:
        print("Error: Could not load data.")