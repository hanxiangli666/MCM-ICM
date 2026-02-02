import pandas as pd
import numpy as np
import random
import math
import os
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    'SALARY_CAP': 1500000,   # 2025 WNBA 软工资帽大约在 150w 左右
    'ROSTER_MIN': 11,
    'ROSTER_MAX': 12,
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
}

# ==========================================
# 2. 数据处理 (修复 NaN 和计算逻辑)
# ==========================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file
        self.salary_file = salary_file

    def normalize_name(self, series):
        """标准化名字以提高匹配率"""
        return series.astype(str).str.lower().str.strip().str.replace('.', '', regex=False)

    def load_and_process(self, target_team_name="Indiana Fever"):
        print(f">>> [DataEngine] 正在加载数据，目标主队: {target_team_name}...")
        
        # --- 1. 读取原始文件 ---
        try:
            read_func = pd.read_excel if self.stats_file.endswith('.xlsx') else pd.read_csv
            stats_raw = read_func(self.stats_file)
            
            read_func = pd.read_excel if self.salary_file.endswith('.xlsx') else pd.read_csv
            salary_raw = read_func(self.salary_file)
        except Exception as e:
            print(f"!!! 文件读取失败: {e}")
            return None

        # 标准化列名
        stats_raw.columns = [c.lower().strip() for c in stats_raw.columns]
        salary_raw.columns = [c.lower().strip() for c in salary_raw.columns]
        
        # --- 2. 提取统计数据 ---
        if 'season' in stats_raw.columns:
            target_season = stats_raw['season'].max()
            stats = stats_raw[stats_raw['season'] == target_season].copy()
        else:
            stats = stats_raw.copy()

        # 确保数值列无误
        stat_cols = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy']
        for c in stat_cols:
            if c not in stats.columns: stats[c] = 0.0
            stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)

        # 聚合球员数据
        metrics = stats.groupby('player').agg({
            'points': 'mean', 'rebounds': 'mean', 'assists': 'mean', 
            'plus_minus': 'mean', 'minutes': 'mean', 'attendance': 'mean',
            'clutch_proxy': 'mean', 'team': 'last' 
        }).reset_index()

        # --- 3. 提取薪资数据 ---
        salary = salary_raw.copy()
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce')
        salary['years_of_service'] = pd.to_numeric(salary.get('years_of_service', 0), errors='coerce').fillna(1)
        
        # --- 4. 合并数据 ---
        metrics['key'] = self.normalize_name(metrics['player'])
        salary['key'] = self.normalize_name(salary['player'])
        
        df = pd.merge(metrics, salary[['key', 'salary_2025', 'years_of_service', 'position', 'team']], 
                      on='key', how='outer', suffixes=('', '_sal'))
        
        # 字段合并与修补
        df['player'] = df['player'].fillna(salary['player'])
        df['team'] = df['team'].fillna(df['team_sal']).fillna('Free Agent')
        
        manual_fixes = {
            'aliyah boston': {'salary': 99000, 'pos': 'C', 'team': target_team_name, 'pts': 14.5, 'reb': 8.4, 'min': 30}, # 新秀合同第3年
            'caitlin clark': {'salary': 78066, 'pos': 'G', 'team': target_team_name, 'pts': 19.2, 'reb': 5.7, 'ast': 8.2, 'min': 32, 'att': 17000},
            'kelsey mitchell': {'salary': 250000, 'pos': 'G', 'team': target_team_name, 'pts': 16.0}, # 假定保留/核心
            'erica wheeler': {'salary': 140000, 'pos': 'G', 'team': target_team_name, 'pts': 9.0},
            'nalyssa smith': {'salary': 91000, 'pos': 'F', 'team': target_team_name, 'pts': 10.0, 'reb': 7.0},
            'lexie hull': {'salary': 84000, 'pos': 'G', 'team': target_team_name, 'pts': 6.0},
            'grace berger': {'salary': 73439, 'pos': 'G', 'team': target_team_name, 'pts': 3.0},
            'victaria saxton': {'salary': 69000, 'pos': 'F', 'team': target_team_name, 'pts': 2.0},
            'temi fagbenle': {'salary': 80000, 'pos': 'C', 'team': target_team_name, 'pts': 5.0}, 
            'katie lou samuelson': {'salary': 175000, 'pos': 'F', 'team': target_team_name, 'pts': 8.0},
            'breanna stewart': {'salary': 205000, 'pos': 'F', 'team': 'New York Liberty', 'pts': 23.0, 'reb': 9.3},
            'napheesa collier': {'salary': 214284, 'pos': 'F', 'team': 'Minnesota Lynx', 'pts': 21.5, 'reb': 8.5},
            'aja wilson': {'salary': 210000, 'pos': 'F', 'team': 'Las Vegas Aces', 'pts': 22.8}, 
            'kelsey plum': {'salary': 200000, 'pos': 'G', 'team': 'Las Vegas Aces', 'pts': 18.7},
            'jewell loyd': {'salary': 241984, 'pos': 'G', 'team': 'Seattle Storm', 'pts': 24.0}, 
            'arike ogunbowale': {'salary': 241984, 'pos': 'G', 'team': 'Dallas Wings', 'pts': 21.0},
            'kahleah copper': {'salary': 241984, 'pos': 'G', 'team': 'Phoenix Mercury', 'pts': 18.0},
            'brionna jones': {'salary': 212000, 'pos': 'C', 'team': 'Connecticut Sun', 'pts': 15.0},
            'dewanna bonner': {'salary': 200000, 'pos': 'F', 'team': 'Connecticut Sun', 'pts': 17.4},
            'nneka ogwumike': {'salary': 160000, 'pos': 'F', 'team': 'Seattle Storm', 'pts': 19.1},
            'skylar diggins-smith': {'salary': 214284, 'pos': 'G', 'team': 'Seattle Storm', 'pts': 16.0},
            'chelsea gray': {'salary': 196267, 'pos': 'G', 'team': 'Las Vegas Aces', 'pts': 15.0},
            'jonquel jones': {'salary': 210000, 'pos': 'C', 'team': 'New York Liberty', 'pts': 16.0},
            'alyssa thomas': {'salary': 218000, 'pos': 'F', 'team': 'Connecticut Sun', 'pts': 15.0},
            'natasha cloud': {'salary': 200000, 'pos': 'G', 'team': 'Phoenix Mercury', 'pts': 12.0},
            'diana taurasi': {'salary': 234936, 'pos': 'G', 'team': 'Phoenix Mercury', 'pts': 16.0},
            'allisha gray': {'salary': 185000, 'pos': 'G', 'team': 'Atlanta Dream', 'pts': 17.1},
            'cheyenne parker': {'salary': 190000, 'pos': 'F', 'team': 'Atlanta Dream', 'pts': 15.0},
            'courtney williams': {'salary': 175000, 'pos': 'G', 'team': 'Minnesota Lynx', 'pts': 10.0},
            'dearica hamby': {'salary': 169000, 'pos': 'F', 'team': 'Los Angeles Sparks', 'pts': 16.0},
            'marina mabrey': {'salary': 208000, 'pos': 'G', 'team': 'Chicago Sky', 'pts': 15.0},
            'satou sabally': {'salary': 210000, 'pos': 'F', 'team': 'Dallas Wings', 'pts': 18.0},
            'natasha howard': {'salary': 224675, 'pos': 'F', 'team': 'Dallas Wings', 'pts': 15.0},
            'azura stevens': {'salary': 195000, 'pos': 'F', 'team': 'Los Angeles Sparks', 'pts': 12.0},
            'betnijah laney': {'salary': 185000, 'pos': 'G', 'team': 'New York Liberty', 'pts': 14.0},
            'brittney griner': {'salary': 150000, 'pos': 'C', 'team': 'Phoenix Mercury', 'pts': 17.0},
            'sophie cunningham': {'salary': 130000, 'pos': 'G', 'team': 'Phoenix Mercury'},
            'sydney colson': {'salary': 76297, 'pos': 'G', 'team': 'Las Vegas Aces'}, 
            'kyra lambert': {'salary': 64154, 'pos': 'G', 'team': 'Free Agent'},
            'makayla timpson': {'salary': 65000, 'pos': 'F', 'team': 'Free Agent'},
        }

        for name, data in manual_fixes.items():
            key = self.normalize_name(pd.Series([name]))[0]
            mask = df['key'] == key
            
            if mask.any():
                # 更新现有行
                idx = df[mask].index
                if 'salary' in data: df.loc[idx, 'salary_2025'] = data['salary']
                if 'pos' in data: df.loc[idx, 'position'] = data['pos']
                if 'team' in data: df.loc[idx, 'team'] = data['team']
                if 'pts' in data: df.loc[idx, 'points'] = data['pts']
                if 'reb' in data: df.loc[idx, 'rebounds'] = data['reb']
            else:
                new_row = {
                    'player': name.title(), 'key': key,
                    'team': data.get('team', target_team_name),
                    'salary_2025': data.get('salary', 76297),
                    'position': data.get('pos', 'G'),
                    'minutes': data.get('min', 20.0),
                    'points': data.get('pts', 8.0),
                    'rebounds': data.get('reb', 3.0),
                    'assists': data.get('ast', 2.0),
                    'attendance': data.get('att', 5000),
                    'years_of_service': 3,
                    'clutch_proxy': 0.5, 'plus_minus': 0
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # --- 6. 全局缺失值填充 ---
        fill_zeros = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy', 'years_of_service']
        for c in fill_zeros:
            if c not in df.columns: df[c] = 0.0
            df[c] = df[c].fillna(0.0)
        df['salary_2025'] = df['salary_2025'].fillna(76297)
        if 'position' not in df.columns: df['position'] = 'G'
        df['position'] = df['position'].fillna('G')
        df['pos_mapped'] = df['position'].apply(self._map_pos)

        # --- 7. 计算 Vi  ---
        df = self._calculate_scores(df)

        # 标记渠道
        target_clean = self.normalize_name(pd.Series([target_team_name]))[0]
        df['team_clean'] = self.normalize_name(df['team'])
        if 'fever' in target_clean:
            mask_home = df['team_clean'].str.contains('fever') | df['team_clean'].str.contains('ind')
            df.loc[mask_home, 'team_clean'] = target_clean
            df.loc[mask_home, 'team'] = target_team_name

        print(f"    - 数据加载完成，全联盟共 {len(df)} 名球员。")
        print(f"    - 目标球队 ({target_team_name}) 现有 {len(df[df['team_clean'] == target_clean])} 人。")
        return df

    def _map_pos(self, p):
        p = str(p).upper()
        if 'C' in p: return 'C'
        if 'F' in p: return 'F'
        return 'G'

    def _calculate_scores(self, df):
        """计算综合评分 (Vi)，此时 df 中不应有 NaN"""
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

        injury_risk = 0.5 * norm(df['minutes'])
        df['ViH'] = 1 - (injury_risk * 0.3) 

        df['ViP'] = 1 - norm(df['years_of_service'])

        df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                         CONFIG['W_H'] * df['ViH'] + CONFIG['W_P'] * df['ViP'])
        
        # 再次检查 Vi_base 是否有 NaN，如果有，填补均值
        if df['Vi_base'].isna().any():
            print("    [Warning] Vi_base calculated NaN values, filling with mean.")
            df['Vi_base'] = df['Vi_base'].fillna(df['Vi_base'].mean())
            
        return df

# ==========================================
# 3. 求解器
# ==========================================
class StrategicSolver:
    def __init__(self, pool, current_team_name):
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        
        if 'fever' in target:
            mask_home = pool['team_clean'].str.contains('fever') | pool['team_clean'].str.contains('ind')
        else:
            mask_home = pool['team_clean'] == target

        self.current_roster = pool[mask_home].copy().reset_index(drop=True)
        # 市场池：非本队球员，且去除 Vi 极低的球员以减少搜索空间
        self.market_pool = pool[~mask_home].sort_values('Vi_base', ascending=False).head(60).reset_index(drop=True)
        
        self.n_current = len(self.current_roster)
        self.n_market = len(self.market_pool)
        
        print(f">>> [SolverInit] 现有阵容: {self.n_current} 人 | 优质自由市场: {self.n_market} 人")

    def get_fitness(self, dna):
        """计算适应度函数 (Z-Score)"""
        mask_keep = dna[:self.n_current] == 1
        mask_buy = dna[self.n_current:] == 1
        
        roster = pd.concat([self.current_roster.iloc[mask_keep], self.market_pool.iloc[mask_buy]])
        count = len(roster)
        
        if count == 0: return -1e9
        
        TC = roster['salary_2025'].sum()
        
        # 1. 阵容完整度
        pos_counts = roster['pos_mapped'].value_counts()
        fit_score = 0
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: fit_score += 1
            else: fit_score -= 0.5 * abs(c - (min_p+max_p)/2)
        ViF = max(0, min(1, fit_score * 0.3 + 0.5))

        # 2. 球队实力与商业
        avg_Vi = roster['Vi_base'].mean()
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        Risk = (roster['salary_2025'] * (1 - roster['ViH'])).sum() * CONFIG['RISK_FACTOR']

        # 3. 硬约束惩罚 (加大权重)
        penalty = 0
        # 人数约束 (强惩罚)
        if count < CONFIG['ROSTER_MIN']: penalty += (CONFIG['ROSTER_MIN'] - count) * 50
        if count > CONFIG['ROSTER_MAX']: penalty += (count - CONFIG['ROSTER_MAX']) * 50
        
        # 工资帽约束 (如果不满足，适应度急剧下降)
        if TC > CONFIG['SALARY_CAP']: 
            penalty += (TC - CONFIG['SALARY_CAP']) / 1000 # 每超 $1000 扣 1 分
        
        # 必须保留至少 4 名老队员
        n_kept = mask_keep.sum()
        if n_kept < 4: penalty += (4 - n_kept) * 20

        # Z-Score
        term_financial = (TR - Risk) / 10000 
        term_performance = avg_Vi * 100      
        
        Z = (CONFIG['MU_1'] * term_financial + 
             CONFIG['MU_2'] * term_performance + 
             CONFIG['MU_3'] * ViF * 50) - penalty
        return Z

    def mutate(self, dna, rate=0.2):
        child = deepcopy(dna)
        # 1. 普通翻转变异
        if random.random() < rate:
            idx = random.randint(0, len(dna)-1)
            child[idx] = 1 - child[idx]
            
        # 2. 交换变异 (Swap Mutation) - 专门用于保持人数平衡
        # 随机“卖”一个保留的，并“买”一个市场的，或者反之
        if random.random() < 0.3: # 30% 概率发生交换
            # 找当前的 1 和 0
            ones = np.where(child == 1)[0]
            zeros = np.where(child == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                idx_1 = np.random.choice(ones)
                idx_0 = np.random.choice(zeros)
                child[idx_1] = 0
                child[idx_0] = 1
                
        return child

    def solve(self, generations=150, pop_size=100):
        total_len = self.n_current + self.n_market
        pop = []
        
        # --- 初始化种群 (Smart Initialization) ---
        for _ in range(pop_size):
            dna = np.zeros(total_len, dtype=int)
            
            # 目标人数 11 或 12
            target_size = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            
            # 随机决定保留多少人 (比如 4 到 target_size)
            n_keep = random.randint(4, min(self.n_current, target_size))
            n_buy = target_size - n_keep
            
            # 填充 DNA
            if self.n_current > 0:
                idx_keep = np.random.choice(self.n_current, n_keep, replace=False)
                dna[idx_keep] = 1
            
            if self.n_market > 0 and n_buy > 0:
                # 优先买 Vi 高的人 (market_pool 已经排过序了)
                # 使用加权随机，倾向于选前排的
                weights = np.linspace(1.0, 0.1, self.n_market)
                weights = weights / weights.sum()
                idx_buy = np.random.choice(self.n_market, min(self.n_market, n_buy), replace=False, p=weights)
                dna[self.n_current + idx_buy] = 1
                
            pop.append(dna)
            
        best_dna, best_fitness = None, -float('inf')
        
        # --- 进化循环 ---
        for gen in range(generations):
            fitnesses = np.array([self.get_fitness(d) for d in pop])
            
            # 记录最佳
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_dna = deepcopy(pop[max_idx])
                
            if gen % 20 == 0:
                print(f"    Iter {gen:3d} | Best Z: {best_fitness:.4f}")
            
            # 选择 (锦标赛)
            new_pop = []
            # 保留精英 (Elitism)
            new_pop.append(best_dna)
            
            while len(new_pop) < pop_size:
                # 锦标赛大小 = 3
                indices = np.random.randint(0, pop_size, 3)
                candidates = [pop[i] for i in indices]
                cand_fits = [fitnesses[i] for i in indices]
                parent = candidates[np.argmax(cand_fits)]
                
                child = self.mutate(parent, rate=0.15)
                new_pop.append(child)
            
            pop = new_pop
            
        return best_dna, best_fitness

# ==========================================
# 4. 主程序与可视化
# ==========================================
if __name__ == "__main__":
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'
    f_salary = 'player_salaries_2025.xlsx'
    
    if os.path.exists(f_stats):
        # 1. 加载数据
        proc = DataProcessor(f_stats, f_salary)
        full_df = proc.load_and_process("Indiana Fever")
        
        if full_df is not None:
            # 2. 初始化求解器
            solver = StrategicSolver(full_df, "Indiana Fever")
            
            # --- 输出原始阵容 ---
            print("\n" + "="*80)
            print("【初始状态】 Indiana Fever 2025 赛季阵容")
            print("="*80)
            original_roster = solver.current_roster
            cols = ['player', 'pos_mapped', 'salary_2025', 'Vi_base']
            print(original_roster[cols].sort_values('salary_2025', ascending=False).to_string(index=False, formatters={'salary_2025': '${:,.0f}'.format, 'Vi_base': '{:.3f}'.format}))
            print(f"\n初始总薪资: ${original_roster['salary_2025'].sum():,.0f}")
            print(f"初始人数: {len(original_roster)}")
            
            # 3. 求解
            if solver.n_current == 0:
                print("Error: 现有阵容为空，请检查手动补丁或数据源！")
            else:
                best_dna, best_z = solver.solve(generations=120, pop_size=80)
                
                # 4. 构建结果
                mask_keep = best_dna[:solver.n_current] == 1
                mask_buy = best_dna[solver.n_current:] == 1
                
                roster_keep = solver.current_roster.iloc[mask_keep]
                roster_trade_out = solver.current_roster.iloc[~mask_keep] # 被交易走的
                roster_buy = solver.market_pool.iloc[mask_buy]
                final_roster = pd.concat([roster_keep, roster_buy])
                
                # --- 最终输出 ---
                print("\n" + "="*80)
                print(f"PRO-INSIGHT 2026 阵容优化结果 (Task 3: Trade & Roster)")
                print("="*80)
                print(f"Optimization Score (Z): {best_z:.4f}")
                print(f"Roster Size:     {len(final_roster)} (Target: 11-12)")
                print(f"Total Cap Used:  ${final_roster['salary_2025'].sum():,.0f} / ${CONFIG['SALARY_CAP']:,.0f}")
                print(f"Cap Space Left:  ${CONFIG['SALARY_CAP'] - final_roster['salary_2025'].sum():,.0f}")
                print(f"Avg Vi Score:    {final_roster['Vi_base'].mean():.4f}")
                print("-" * 80)
                
                fmt = {'salary_2025': '${:,.0f}'.format, 'Vi_base': '{:.3f}'.format}
                
                print(f"【保留核心 (Retained, N={len(roster_keep)})】")
                if not roster_keep.empty:
                    print(roster_keep[cols].sort_values('salary_2025', ascending=False).to_string(index=False, formatters=fmt))
                else:
                    print(" (None)")
                
                print("-" * 80)
                print(f"【新援引入 (Acquired, N={len(roster_buy)})】")
                if not roster_buy.empty:
                    print(roster_buy[cols + ['team']].sort_values('salary_2025', ascending=False).to_string(index=False, formatters=fmt))
                else:
                    print(" (None)")

                print("-" * 80)
                print(f"【离队/交易 (Traded Out, N={len(roster_trade_out)})】")
                if not roster_trade_out.empty:
                    print(roster_trade_out[cols].sort_values('salary_2025', ascending=False).to_string(index=False, formatters=fmt))
                
                print("="*80)
    else:
        print("未找到数据文件，请检查路径。")