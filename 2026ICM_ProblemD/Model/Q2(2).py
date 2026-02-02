import pandas as pd
import numpy as np
import random
import math
import os
from copy import deepcopy

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    'SALARY_CAP': 1500000,
    'ROSTER_MIN': 11,
    'ROSTER_MAX': 12,
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
}

# ==========================================
# 2. 数据处理
# ==========================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file
        self.salary_file = salary_file
        
    def load_data(self):
        print(f">>> 正在读取数据...")
        try:
            reader_stats = pd.read_excel if self.stats_file.endswith('.xlsx') else pd.read_csv
            reader_salary = pd.read_excel if self.salary_file.endswith('.xlsx') else pd.read_csv
            stats = reader_stats(self.stats_file)
            salary = reader_salary(self.salary_file)
        except Exception as e:
            print(f"文件读取错误: {e}")
            return None

        # 清洗列名
        stats.columns = [c.lower().strip() for c in stats.columns]
        salary.columns = [c.lower().strip() for c in salary.columns]
        
        # 强制数值转换
        for col in ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'season']:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors='coerce').fillna(0)
        
        target_season = stats['season'].max() if 'season' in stats.columns else 2024
        print(f">>> 使用赛季数据: {target_season}")
        
        stats['clutch_proxy'] = stats['points'] * 0.5 + stats['plus_minus'] * 0.5
        
        metrics = stats[stats['season'] == target_season].groupby('player').agg({
            'points': 'mean', 'rebounds': 'mean', 'assists': 'mean', 
            'plus_minus': 'mean', 'minutes': 'mean', 'attendance': 'mean',
            'clutch_proxy': 'mean', 'pos': 'first'
        }).reset_index()

        metrics['key'] = metrics['player'].astype(str).str.lower().str.strip()
        salary['key'] = salary['player'].astype(str).str.lower().str.strip()
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce').fillna(76000)
        salary['years_of_service'] = pd.to_numeric(salary['years_of_service'], errors='coerce').fillna(2)

        # 合并
        df = pd.merge(salary, metrics, on='key', how='left', suffixes=('', '_stat'))
        
        # 缺失值填充
        df['pos'] = df['pos'].fillna('G')
        num_cols = ['points', 'rebounds', 'minutes', 'attendance', 'clutch_proxy']
        for col in num_cols:
            df[col] = df[col].fillna(0)

        # 映射位置
        def map_pos(p):
            p = str(p).upper()
            if 'C' in p: return 'C'
            if 'F' in p: return 'F'
            return 'G'
        df['pos_mapped'] = df['pos'].apply(map_pos)

        # 因为这些球员目前属于其他球队，要获得他们通常需要通过交易(Trade)
        df['channel'] = 'Trade (q=3)'
        
        return df

    def generate_real_bench(self, df):
        """
        填充低薪池，并在此处明确区分 Draft(q=1) 和 Free Agent(q=2)
        """
        real_bench_players = [
            # 姓名, 位置, 2025薪资, 类型(用于判断渠道), 备注
            ("Kate Martin", "G", 67000, "Rookie", "Fan Favorite"),
            ("Joyner Holmes", "F", 78831, "Veteran Min", "Seattle Storm"),
            ("Brianna Turner", "F", 85000, "Value Vet", "Chicago Sky"),
            ("Sydney Colson", "G", 90000, "Veteran", "Las Vegas Aces"),
            ("Katie Lou Samuelson", "G", 90000, "Shooter", "Indiana Fever"),
            ("Veronica Burton", "G", 78831, "Defender", "Connecticut Sun"),
            ("Temi Fagbenle", "C", 78831, "Veteran Min", "Indiana Fever"),
            ("Julie Vanloo", "G", 66079, "Rookie Scale", "Washington Mystics"),
            ("Stephanie Talbot", "F", 80000, "Role Player", "Los Angeles Sparks"),
            ("Victaria Saxton", "F", 66079, "Rookie Min", "Indiana Fever"),
            ("Grace Berger", "G", 70000, "Rookie Scale", "Indiana Fever"),
            ("Li Yueru", "C", 66079, "Prospect", "Los Angeles Sparks"),
            ("Crystal Dangerfield", "G", 78831, "Veteran Min", "Atlanta Dream"),
            ("Kiana Williams", "G", 66079, "Min", "Seattle Storm"),
            ("Queen Egbo", "C", 78831, "Veteran Min", "Los Angeles Sparks")
        ]

        dummies = []
        for name, pos, salary, p_type, note in real_bench_players:
            base_vi = 0.20 + random.random() * 0.15 
            
            # === 根据球员类型判断收购渠道 ===
            if any(k in p_type for k in ["Rookie", "Prospect", "Scale"]):
                acq_channel = "Draft (q=1)"
            else:
                acq_channel = "Free Agent (q=2)"
            
            dummy = {
                'player': name,
                'team': 'Free Agent Market', 
                'team_clean': 'free agent market',
                'salary_2025': salary,
                'pos_mapped': pos,
                'Vi_base': base_vi,
                'ViA': base_vi * 0.8, 
                'ViB': base_vi * 0.5, 
                'ViH': 0.9, 
                'years_of_service': 3 if salary > 70000 else 1,
                'channel': acq_channel # 写入渠道
            }
            dummies.append(dummy)
        
        dummy_df = pd.DataFrame(dummies)
        for col in df.columns:
            if col not in dummy_df.columns:
                dummy_df[col] = 0
                
        return pd.concat([df, dummy_df], ignore_index=True)

    def calculate_values(self, df):
        df = df.copy()
        def norm(col):
            min_v, max_v = col.min(), col.max()
            if max_v == min_v: return 0.5
            denom = max_v - min_v
            return (col - min_v) / (denom if denom > 0 else 1)

        if 'Vi_base' not in df.columns or df['Vi_base'].isna().any():
            df['ViA'] = (0.25 * norm(df['points']) + 0.20 * norm(df['rebounds']) + 
                         0.20 * norm(df['assists']) + 0.15 * norm(df['plus_minus']) + 
                         0.20 * norm(df['clutch_proxy']))
            df['ViB'] = 0.6 * norm(df['attendance']) + 0.4 * norm(df['salary_2025'])
            injury_prob = norm(df['minutes']) * 0.5 + norm(df['years_of_service']) * 0.5
            df['ViH'] = 1 - (injury_prob * 0.4) 
            df['ViP'] = 1 - norm(df['years_of_service'])
            
            df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                             CONFIG['W_H'] * df['ViH'] + CONFIG['W_P'] * df['ViP'])
        
        df['Vi_base'] = df['Vi_base'].fillna(0.1) 
        return df

# ==========================================
# 3. 求解器
# ==========================================
class StrategicSolver:
    def __init__(self, pool, current_team_name):
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        
        self.current_roster = pool[pool['team_clean'] == target].copy()
        self.market_pool = pool[pool['team_clean'] != target].copy()
        
        self.n_current = len(self.current_roster)
        self.n_market = len(self.market_pool)

    def get_fitness(self, dna):
        """Calculate the Z score (Objective Function)"""
        mask_keep = dna[:self.n_current] == 1
        mask_buy = dna[self.n_current:] == 1
        
        roster = pd.concat([self.current_roster.iloc[mask_keep], self.market_pool.iloc[mask_buy]])
        count = len(roster)
        if count == 0: return -1e12 
        TC = roster['salary_2025'].sum()
        pos_counts = roster['pos_mapped'].value_counts()
        fit_score = 0
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: fit_score += 1
            else: fit_score -= 0.5 * abs(c - (min_p+max_p)/2)
        ViF = max(0, min(1, fit_score * 0.3 + 0.5))
        avg_Vi = roster['Vi_base'].mean()
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        Risk = (roster['salary_2025'] * (1 - roster['ViH'])).sum()
        penalty = 0
        if count < CONFIG['ROSTER_MIN']:
            penalty += 1e7 + (CONFIG['ROSTER_MIN'] - count) * 1e6
        elif count > CONFIG['ROSTER_MAX']:
            penalty += 1e7 + (count - CONFIG['ROSTER_MAX']) * 1e6
            
        if TC > CONFIG['SALARY_CAP']:
            penalty += (TC - CONFIG['SALARY_CAP']) * 10 
            
        n_kept = mask_keep.sum()
        if n_kept < 2: 
             penalty += (2 - n_kept) * 500000

        # --- Final Z Score ---
        term1 = (TR - Risk) / (TC + 1e-5) * 50000
        Z = (CONFIG['MU_1'] * term1 + CONFIG['MU_2'] * avg_Vi * 5000 + CONFIG['MU_3'] * ViF * 5000) - penalty
        return Z

    def mutate(self, dna, rate=0.3):
        child = deepcopy(dna)
        
        # Strategy A: Swap Mutation (Maintains roster size)
        # Find indices of selected (1) and unselected (0) players
        idx_ones = np.where(child == 1)[0]
        idx_zeros = np.where(child == 0)[0]
        
        # 60% chance to perform a Swap if possible
        if len(idx_ones) > 0 and len(idx_zeros) > 0 and random.random() < 0.6:
            i_one = np.random.choice(idx_ones)
            i_zero = np.random.choice(idx_zeros)
            # Swap their status
            child[i_one] = 0
            child[i_zero] = 1
            return child

        # Strategy B: Standard Flip Mutation (Changes roster size)
        # Used to explore different roster sizes (e.g., moving from 11 to 12)
        if random.random() < rate:
            total_len = len(dna)
            m_idx = random.randint(0, total_len-1)
            child[m_idx] = 1 - child[m_idx]
            
        return child

    def solve(self, generations=150, pop_size=60):
        total_len = self.n_current + self.n_market
        pop = []
        print(f">>> Solver Environment: Current {self.n_current}, Market (Classified) {self.n_market}")
        
        for _ in range(pop_size):
            dna = np.zeros(total_len, dtype=int)
            
            if self.n_current > 0:
                n_keep = random.randint(2, min(self.n_current, 4))
                idx_keep = np.random.choice(self.n_current, n_keep, replace=False)
                dna[idx_keep] = 1
            else:
                n_keep = 0
            
            # Reach 11-12 total
            current_count = n_keep
            target = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            needed = target - current_count
            
            if needed > 0 and self.n_market >= needed:
                idx_buy = np.random.choice(self.n_market, needed, replace=False)
                dna[self.n_current + idx_buy] = 1
            pop.append(dna)
            
        best_dna, best_fitness = None, -float('inf')
        T = 200.0
        
        for gen in range(generations):
            fitnesses = [self.get_fitness(d) for d in pop]
            curr_max = max(fitnesses)
            
            if curr_max > best_fitness:
                best_fitness = curr_max
                best_dna = deepcopy(pop[fitnesses.index(curr_max)])
                
            if gen % 30 == 0:
                print(f"Iter {gen:3d} | Best Z: {best_fitness:.2f}")
                
            new_pop = []
            while len(new_pop) < pop_size:
                candidates = random.sample(pop, 3)
                candidates_fitness = [self.get_fitness(c) for c in candidates]
                parent = candidates[np.argmax(candidates_fitness)]
                child = self.mutate(parent, rate=0.2)
                f_child = self.get_fitness(child)
                f_parent = self.get_fitness(parent)
                
                if f_child >= f_parent:
                    new_pop.append(child)
                else:
                    prob = math.exp((f_child - f_parent) / T)
                    if random.random() < prob:
                        new_pop.append(child)
                    else:
                        new_pop.append(parent)
            
            pop = new_pop
            T *= 0.92 
            
        return best_dna, best_fitness

if __name__ == "__main__":
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'
    f_salary = 'player_salaries_2025.xlsx'
    
    if os.path.exists(f_stats):
        # 1. Process Data
        proc = DataProcessor(f_stats, f_salary)
        full_df = proc.load_data()
        
        if full_df is not None:
            # 2. Calculate
            valued_df = proc.calculate_values(full_df)
            
            # 3. Add Real Bench Players
            enhanced_df = proc.generate_real_bench(valued_df)
            
            # 4. Solve
            my_team = "Indiana Fever"
            solver = StrategicSolver(enhanced_df, my_team)
            
            if solver.n_current == 0:
                print(f"Warning: Team '{my_team}' not found.")
            else:
                # 5. Optimization
                best_dna, best_z = solver.solve()
                
                # 6. Results
                mask_keep = best_dna[:solver.n_current] == 1
                mask_buy = best_dna[solver.n_current:] == 1
                
                roster_keep = solver.current_roster.iloc[mask_keep]
                roster_buy = solver.market_pool.iloc[mask_buy]
                final_roster = pd.concat([roster_keep, roster_buy])
                
                # 7. Print
                print("\n" + "="*80)
                print(f"最优 Score Z: {best_z:.2f}")
                print(f"Roster Size: {len(final_roster)} (Target: 11-12)")
                print(f"Total Salary: ${final_roster['salary_2025'].sum():,.0f} (Cap: ${CONFIG['SALARY_CAP']:,.0f})")
                print("-" * 80)
                
                cols = ['player', 'pos_mapped', 'salary_2025', 'Vi_base', 'channel']
                print(f"【Retained Core ({len(roster_keep)} players)】:\n{roster_keep[cols].to_string(index=False)}")
                print("-" * 80)
                print(f"【New Acquisitions ({len(roster_buy)} players)】:\n{roster_buy[cols].to_string(index=False)}")
                print("="*80)
    else:
        print("No Data!")