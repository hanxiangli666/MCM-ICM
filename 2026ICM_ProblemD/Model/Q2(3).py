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
    'SALARY_CAP': 1500000,  
    'ROSTER_MIN': 11,
    'ROSTER_MAX': 12,
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
    'SALARY_MIN_LEAGUE': 64154,   # 联盟底薪
    'SALARY_MAX_LEAGUE': 241984,  # 顶薪
    'TOTAL_GAME_MINUTES': 200,    # 全队总时间 (40mins * 5人)
    'PLAYER_MAX_MINUTES': 38.0,   # 单人最大体能极限
    'PLAYER_MIN_MINUTES': 5.0 ,    # 只要上场至少打5分钟
    'N_HIGHRISK_MAX': 2,   # 高风险球员(ViH < 0.6) 允许的最大数量
    'N_POTENTIAL_MIN': 2   # 潜力球员(ViP > 0.8) 要求的最小数量
}

# ==========================================
# 2. 数据处理
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

        # 预计算：为了支持连续变量 S_j，我们需要为每个球员确定薪资谈判区间
        df['S_min'] = df['salary_2025'].apply(lambda x: max(CONFIG['SALARY_MIN_LEAGUE'], x * 0.8))
        df['S_max'] = df['salary_2025'].apply(lambda x: min(CONFIG['SALARY_MAX_LEAGUE'], x * 1.2))

        # 对于超级巨星，下限较高
        mask_super = df['S_max'] > 200000
        df.loc[mask_super, 'S_min'] = df.loc[mask_super, 'salary_2025'] * 0.9

        df = self._calculate_scores(df)

        target_clean = self.normalize_name(pd.Series([target_team_name]))[0]
        df['team_clean'] = self.normalize_name(df['team'])
        if 'fever' in target_clean:
            mask_home = df['team_clean'].str.contains('fever') | df['team_clean'].str.contains('ind')
            df.loc[mask_home, 'team_clean'] = target_clean
            df.loc[mask_home, 'team'] = target_team_name

        print(f"    - 数据加载完成，全联盟共 {len(df)} 名球员。")
        return df

    def _map_pos(self, p):
        p = str(p).upper()
        if 'C' in p: return 'C'
        if 'F' in p: return 'F'
        return 'G'


    def _calculate_scores(self, df):
        # 静态能力评分 Vi
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

        # ViH 健康分现在作为静态基准，具体风险在 evaluate 中结合时间 tj 计算
        df['ViH_base'] = 1 - (0.5 * norm(df['minutes']) * 0.3) 
        df['ViP'] = 1 - norm(df['years_of_service'])
        
        # 基础综合分
        df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                         CONFIG['W_H'] * df['ViH_base'] + CONFIG['W_P'] * df['ViP'])
        df['Vi_base'] = df['Vi_base'].fillna(df['Vi_base'].mean())
        return df


# ==========================================
# 3. 求解器
# ==========================================
class Individual:
    """个体类，存储DNA、目标函数值、Rank和拥挤度"""
    def __init__(self, n_total):
        # Part 1: Binary (0/1)
        self.dna_decisions = np.zeros(n_total, dtype=int)
        
        # Part 2: Continuous (0.0 - 1.0) 
        # Actual Salary = S_min + S_factor * (S_max - S_min)
        self.dna_salary = np.random.rand(n_total)
        
        # Part 3: Continuous (0.0 - 1.0)
        # Real Time = (Weight_i / Sum(Weights)) * 200
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
        print(f">>> [SolverInit] 混合变量优化引擎就绪。变量维度: 3 * {self.n_total}")



    def evaluate(self, ind):
        # 1. 解码二值变量：确定名单
        indices = np.where(ind.dna_decisions == 1)[0]
        
        if len(indices) == 0:
            ind.objectives = [1e9, 1e9]
            ind.violation = 1e9
            return

        roster = self.full_pool.iloc[indices].copy()
        
        # 2. 解码连续变量 S (薪资)
        s_factors = ind.dna_salary[indices]
        roster['opt_salary'] = roster['S_min'] + s_factors * (roster['S_max'] - roster['S_min'])
        
        # 3. 解码连续变量 t (时间)
        t_weights = ind.dna_time[indices]
        total_weight = np.sum(t_weights)
        if total_weight == 0:
            roster['opt_minutes'] = CONFIG['TOTAL_GAME_MINUTES'] / len(roster)
        else:
            roster['opt_minutes'] = (t_weights / total_weight) * CONFIG['TOTAL_GAME_MINUTES']
        
        # 软约束：单人时间限制
        roster['opt_minutes'] = roster['opt_minutes'].clip(CONFIG['PLAYER_MIN_MINUTES'], CONFIG['PLAYER_MAX_MINUTES'])
        
        # 重新归一化以确保总和接近 200
        scale_factor = CONFIG['TOTAL_GAME_MINUTES'] / roster['opt_minutes'].sum()
        roster['opt_minutes'] *= scale_factor
        ind.decoded_roster = roster

        # --- 计算目标 ---
        TC = roster['opt_salary'].sum()
        
        # 表现分计算 (结合时间 t)
        time_ratio = roster['opt_minutes'] / 40.0 # 标准化时间
        
        # 动态健康分 ViH: 时间越长，风险越大
        roster['ViH_dynamic'] = 1.0 - (time_ratio * 0.8) 
        
        # 综合贡献值
        roster['Vi_contribution'] = roster['Vi_base'] * time_ratio
        
        avg_Vi = roster['Vi_contribution'].sum()
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        
        # 风险计算
        Risk = (roster['opt_salary'] * (1 - roster['ViH_dynamic'])).sum() * CONFIG['RISK_FACTOR']
        violation = 0
        count = len(roster)
        
        # 1. 基础人数约束
        if count < CONFIG['ROSTER_MIN']: violation += (CONFIG['ROSTER_MIN'] - count) * 1000
        if count > CONFIG['ROSTER_MAX']: violation += (count - CONFIG['ROSTER_MAX']) * 1000
        
        # 2. 工资帽
        if TC > CONFIG['SALARY_CAP']: violation += (TC - CONFIG['SALARY_CAP']) / 100
        
        # 3. 高风险球员限制 (Constraint: Sum(ViH < 0.6) <= N_highrisk)
        # 根据图片公式：Vi^H < 0.6 的人数必须小于等于阈值
        n_high_risk = len(roster[roster['ViH_base'] < 0.6])
        if n_high_risk > CONFIG['N_HIGHRISK_MAX']:
             violation += (n_high_risk - CONFIG['N_HIGHRISK_MAX']) * 5000  # 强惩罚

        # 4. 潜力球员下限 (Constraint: Sum(ViP > 0.8) >= N_potential)
        # 根据图片公式：Vi^P > 0.8 的人数必须大于等于阈值
        n_potential = len(roster[roster['ViP'] > 0.8])
        if n_potential < CONFIG['N_POTENTIAL_MIN']:
             violation += (CONFIG['N_POTENTIAL_MIN'] - n_potential) * 5000 # 强惩罚

        # 5. 交易人数平衡
        # Acquired (交易获得): 索引 >= n_current 的球员
        n_acquired = len(roster[roster.index >= self.n_current])
        
        # 初始阵容中被保留的人数
        n_retained = len(roster[roster.index < self.n_current])
        # 被放弃的人数 = 初始总人数 - 保留人数
        n_waived = self.n_current - n_retained
        
        # 交易获得人数 不得超过 放弃人数
        if n_acquired > n_waived:
            violation += (n_acquired - n_waived) * 10000

        # 保留原有核心班底约束
        if n_retained < 4: violation += (4 - n_retained) * 2000

        ind.violation = violation

        # 构造目标函数
        obj_salary = TC
        
        # 表现分计算
        fit_score = 0
        pos_raw_score = 0
        pos_counts = roster['pos_mapped'].value_counts()
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: 
                pos_raw_score += 1
            else: 
                pos_raw_score -= 0.5 * abs(c - (min_p + max_p) / 2)
        
        # 归一化 PosFit (映射到 0.0 ~ 1.0)
        PosFit = max(0.0, min(1.0, pos_raw_score * 0.3 + 0.5))

        # --- Part 2: StyleFit ---
        # 阵容平均正负值(avg_pm)越高，协同得分越高。
        # 使用 Logistic Sigmoid 函数平滑映射，中心点为 0
        # avg_pm = 0 -> 0.5分; avg_pm = +4 -> ~0.88分; avg_pm = -4 -> ~0.12分
        avg_pm = roster['plus_minus'].mean()
        # 系数 0.5 控制曲线陡峭程度，可根据需要调整
        StyleFit = 1.0 / (1.0 + math.exp(-0.5 * avg_pm))

        # --- Part 3: 最终 V^F 合成 ---
        ViF = 0.5 * PosFit + 0.5 * StyleFit
 

        term_financial = (TR - Risk) / 10000 
        performance_score = (CONFIG['MU_1'] * term_financial + 
                             CONFIG['MU_2'] * avg_Vi * 20 + 
                             CONFIG['MU_3'] * ViF * 50)
        
        obj_perf = -performance_score # Minimize negative performance

        ind.objectives = [obj_salary, obj_perf]
        ind.z_score = performance_score - violation



    def fast_nondominated_sort(self, population):
        """标准 NSGA-II 非支配排序"""
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
                if dominates:
                    p.dominated_solutions.append(q)
                elif self._check_dominated(q, p):
                    p.domination_count += 1
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
        """混合变量变异算子"""
        child = deepcopy(ind)
        
        # 1. Binary Mutation (Flip)
        # 保持一定的稀疏性，避免全部置1
        if random.random() < 0.3:
            # Swap: 将一个1变成0，一个0变成1，保持总人数稳定
            ones = np.where(child.dna_decisions == 1)[0]
            zeros = np.where(child.dna_decisions == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                child.dna_decisions[np.random.choice(ones)] = 0
                child.dna_decisions[np.random.choice(zeros)] = 1
        
        if random.random() < rate:
            idx = random.randint(0, self.n_total - 1)
            child.dna_decisions[idx] = 1 - child.dna_decisions[idx]

        # 2. Continuous Mutation (Salary & Time) - Gaussian Noise
        # 对 S 和 t 进行微扰
        sigma = 0.1 # 标准差
        
        # Salary Mutation
        mask_s = np.random.rand(self.n_total) < rate
        noise_s = np.random.randn(self.n_total) * sigma
        child.dna_salary[mask_s] += noise_s[mask_s]
        child.dna_salary = np.clip(child.dna_salary, 0, 1) # 必须在 [0,1]
        
        # Time Mutation
        mask_t = np.random.rand(self.n_total) < rate
        noise_t = np.random.randn(self.n_total) * sigma
        child.dna_time[mask_t] += noise_t[mask_t]
        child.dna_time = np.clip(child.dna_time, 0, 1) # 必须在 [0,1]

        return child

    def solve(self, generations=100, pop_size=100):
        # 1. 初始化
        population = []
        for _ in range(pop_size):
            ind = Individual(self.n_total)
            
            # 初始化 Decisions (保证初始可行性)
            target_size = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            
            # 必须从当前阵容选一部分
            n_keep = random.randint(4, min(self.n_current, target_size))
            n_buy = target_size - n_keep
            
            idxs_keep = np.random.choice(self.n_current, n_keep, replace=False)
            ind.dna_decisions[idxs_keep] = 1
            
            # 从市场选一部分 (根据 Vi 轮盘赌)
            if n_buy > 0:
                market_indices = np.arange(self.n_current, self.n_total)
                market_probs = self.full_pool.iloc[market_indices]['Vi_base'].values
                market_probs = market_probs / market_probs.sum()
                idxs_buy = np.random.choice(market_indices, n_buy, replace=False, p=market_probs)
                ind.dna_decisions[idxs_buy] = 1
            
            # Salary/Time 随机初始化 (已经在 init 中完成)
            self.evaluate(ind)
            population.append(ind)

        # 2. 进化
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

            if gen % 10 == 0:
                best_z = max(p.z_score for p in population)
                print(f"    Iter {gen:3d} | Best Z: {best_z:.4f}")

        # 3. 结果
        pareto_front = [p for p in population if p.rank == 0 and p.violation == 0]
        if not pareto_front:
            print("Warning: No fully feasible solution found. Using best approximate.")
            best_ind = max(population, key=lambda x: x.z_score)
        else:
            best_ind = max(pareto_front, key=lambda x: x.z_score)
            
        return best_ind
    


# ==========================================
# 4. 主程序与可视化
# ==========================================


if __name__ == "__main__":
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'
    f_salary = 'player_salaries_2025.xlsx'
    
    if os.path.exists(f_stats):
        proc = DataProcessor(f_stats, f_salary)
        full_df = proc.load_and_process("Indiana Fever")
        
        if full_df is not None:
            solver = StrategicSolver(full_df, "Indiana Fever")  
            print("\n" + "="*80)
            print("【初始状态】 Indiana Fever 2025 赛季阵容")
            print("="*80)
            cols = ['player', 'pos_mapped', 'salary_2025', 'Vi_base']
            print(solver.current_roster[cols].to_string(index=False))
            print("-" * 80)
            best_ind = solver.solve(generations=120, pop_size=100)
            final_roster = best_ind.decoded_roster.copy()
            
            # 区分来源：index < n_current 为保留，否则为新援
            final_roster['Source'] = final_roster.index.map(
                lambda x: 'Retained' if x < solver.n_current else 'Acquired')
            # 计算被交易走的 (Original - Retained)
            retained_indices = final_roster[final_roster['Source'] == 'Retained'].index
            traded_df = solver.current_roster[~solver.current_roster.index.isin(retained_indices)].copy()
            print("\n" + "="*80)
            print(f"PRO-INSIGHT 2026 混合变量优化报告 (Result)")
            print("="*80)
            print(f"Optimization Score (Z): {best_ind.z_score:.4f}")
            print(f"Total Cost (S_opt):      ${final_roster['opt_salary'].sum():,.0f} / ${CONFIG['SALARY_CAP']:,.0f}")
            print(f"Total Minutes (t_opt):   {final_roster['opt_minutes'].sum():.1f} / 200.0")
            print(f"Roster Size:             {len(final_roster)}")
            print("-" * 80)
            
            fmt_dict = {
                'opt_salary': '${:,.0f}'.format,
                'salary_2025': '${:,.0f}'.format,
                'opt_minutes': '{:.1f}'.format,
                'Vi_base': '{:.3f}'.format
            }
            cols_show = ['player', 'pos_mapped', 'salary_2025', 'opt_salary', 'opt_minutes', 'Vi_base']
            
            print(f"【 1. 留队核心 (Retained) 】")
            df_ret = final_roster[final_roster['Source'] == 'Retained']
            if not df_ret.empty:
                print(df_ret[cols_show].sort_values('opt_minutes', ascending=False).to_string(index=False, formatters=fmt_dict))
            else:
                print(" (None)")
                
            print("-" * 80)
            print(f"【 2. 强援引进 (Acquired) 】")
            df_acq = final_roster[final_roster['Source'] == 'Acquired']
            if not df_acq.empty:
                print(df_acq[cols_show + ['team']].sort_values('opt_minutes', ascending=False).to_string(index=False, formatters=fmt_dict))
            else:
                print(" (None)")
                
            print("-" * 80)
            print(f"【 3. 离队/交易 (Traded/Waived) 】")
            if not traded_df.empty:
                print(traded_df[['player', 'pos_mapped', 'salary_2025', 'Vi_base']].sort_values('salary_2025', ascending=False).to_string(index=False, formatters=fmt_dict))
            else:
                print(" (None)")
            print("="*80)
    else:
        print(f"Error: File {f_stats} not found.")