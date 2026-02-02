"""
==============================================================================
项目名称: 混合变量动态阵容优化系统 Pro (Mixed-Variable Roster Optimizer Pro)
版本: Q2(4) - Robust Data Edition
注释制作: Gemini

【版本升级重点】
1. 稳健的数据清洗 (Robust Data Cleaning):
   - 引入了正则表达式 (Regex) 库 `re`。
   - `normalize_name` 函数升级：不仅仅是去空格转小写，还移除了所有非字母数字的符号。
     例如："A.J. Wilson" 和 "aj wilson" 以及 "A.J.-Wilson" 都会被清洗成 "ajwilson"，
     极大提高了数据合并时的匹配成功率。

2. 混合变量决策 (Mixed-Variable Decision Making):
   - 这是一个高难度的优化问题，因为它同时包含：
     (1) 离散变量 (Binary): 选谁？(0/1)
     (2) 连续变量 (Continuous): 
         - 薪资 (Salary): 可以在 [S_min, S_max] 区间内谈判。
         - 时间 (Minutes): 可以在 [5, 38] 分钟内分配。

3. 多目标与约束处理 (Multi-objective & Constraints):
   - 目标：最小化薪资总额 (Cost) vs 最大化球队表现 (Performance)。
   - 动态约束：
     - 硬约束：工资帽、人数限制。
     - 软约束：高风险球员数量限制 (N_HIGHRISK_MAX)、潜力新星数量限制 (N_POTENTIAL_MIN)。
     - 交易逻辑：限制新交易进来的人数不能远超裁掉的人数。

【适合场景】
这是数学建模竞赛 (MCM/ICM) 中解决“资源分配 + 组合优化”类问题的标准高级解法。
==============================================================================
"""
import pandas as pd     # 数据处理库
import numpy as np      # 矩阵计算库
import random           # 随机数
import math             # 数学函数
import os               # 系统路径
from copy import deepcopy # 深拷贝
import warnings         # 警告控制
import re               # 【新增】正则表达式库，用于高级文本清洗

warnings.filterwarnings('ignore') # 忽略 Pandas 的版本警告

# ==========================================
# 1. 配置参数 (CONFIG)
# 定义比赛规则、薪资限制和算法权重
# ==========================================
CONFIG = {
    # 权重体系：竞技(A), 商业(B), 健康(H), 潜力(P), 结构(F)
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    
    # 目标函数系数
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    
    # 硬性约束
    'SALARY_CAP': 1500000,   # 工资帽
    'ROSTER_MIN': 11,        # 最小人数
    'ROSTER_MAX': 12,        # 最大人数
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)}, # 位置分布要求
    
    # 经济模型参数
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
    'SALARY_MIN_LEAGUE': 64154,   # 联盟底薪
    'SALARY_MAX_LEAGUE': 241984,  # 顶薪
    
    # 时间管理参数
    'TOTAL_GAME_MINUTES': 200,    # 全队总时间 (5人 * 40分钟)
    'PLAYER_MAX_MINUTES': 38.0,   # 单人体能极限
    'PLAYER_MIN_MINUTES': 5.0 ,   # 单人最少上场时间
    
    # 结构风险控制
    'N_HIGHRISK_MAX': 2,   # 玻璃人（易受伤）最多只能有2个
    'N_POTENTIAL_MIN': 2   # 潜力股（年轻球员）至少要有2个
}

# ==========================================
# 2. 数据处理 (DataProcessor)
# 负责数据的读取、清洗、匹配和评分
# ==========================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file
        self.salary_file = salary_file

    def normalize_name(self, series):
        """
        【核心改进】标准化名字以提高匹配率
        使用了正则表达式 re.sub(r"[^a-z0-9]", "", x)
        作用：将 "Caitlin Clark." 变成 "caitlinclark"，移除空格和标点。
        """
        return series.astype(str).str.lower().str.strip().apply(lambda x: re.sub(r"[^a-z0-9]", "", x))

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

        # 初步清洗列名
        stats_raw.columns = [c.lower().strip() for c in stats_raw.columns]
        salary_raw.columns = [c.lower().strip() for c in salary_raw.columns]
        
        # --- 2. 提取统计数据 ---
        # 只保留最新赛季的数据
        if 'season' in stats_raw.columns:
            target_season = stats_raw['season'].max()
            stats = stats_raw[stats_raw['season'] == target_season].copy()
        else:
            stats = stats_raw.copy()

        # 确保关键数值列为 float，非法值转为 0
        stat_cols = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy']
        for c in stat_cols:
            if c not in stats.columns: stats[c] = 0.0
            stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)

        # 聚合：将球员多场比赛数据取平均
        metrics = stats.groupby('player').agg({
            'points': 'mean', 'rebounds': 'mean', 'assists': 'mean', 
            'plus_minus': 'mean', 'minutes': 'mean', 'attendance': 'mean',
            'clutch_proxy': 'mean', 'team': 'last' 
        }).reset_index()

        # --- 3. 提取薪资数据 ---
        salary = salary_raw.copy()
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce')
        salary['years_of_service'] = pd.to_numeric(salary.get('years_of_service', 0), errors='coerce').fillna(1)
        
        # --- 4. 合并数据 (Merge) ---
        # 使用加强版的 normalize_name 生成 key
        metrics['key'] = self.normalize_name(metrics['player'])
        salary['key'] = self.normalize_name(salary['player'])
        
        # 外连接合并，确保不漏掉任何人
        df = pd.merge(metrics, salary[['key', 'salary_2025', 'years_of_service', 'position', 'team']], 
                      on='key', how='outer', suffixes=('', '_sal'))
        
        # 合并后的修补工作
        df['player'] = df['player'].fillna(salary['player'])
        df['team'] = df['team'].fillna(df['team_sal']).fillna('Free Agent')
        
        # --- 手动数据修复字典 ---
        # 用于修正爬虫爬不到或者格式错误的关键球员数据
        manual_fixes = {
            'aliyah boston': {'salary': 99000, 'pos': 'C', 'team': target_team_name, 'pts': 14.5, 'reb': 8.4, 'min': 30},
            'caitlin clark': {'salary': 78066, 'pos': 'G', 'team': target_team_name, 'pts': 19.2, 'reb': 5.7, 'ast': 8.2, 'min': 32, 'att': 17000},
            # ... (其他球员数据，省略以节省空间) ...
            'makayla timpson': {'salary': 65000, 'pos': 'F', 'team': 'Free Agent'},
        }

        # 应用手动修复
        for name, data in manual_fixes.items():
            key = self.normalize_name(pd.Series([name]))[0]
            mask = df['key'] == key
            
            if mask.any():
                # 如果找到了，更新数据
                idx = df[mask].index
                if 'salary' in data: df.loc[idx, 'salary_2025'] = data['salary']
                if 'pos' in data: df.loc[idx, 'position'] = data['pos']
                if 'team' in data: df.loc[idx, 'team'] = data['team']
                if 'pts' in data: df.loc[idx, 'points'] = data['pts']
                if 'reb' in data: df.loc[idx, 'rebounds'] = data['reb']
            else:
                # 如果没找到，创建新的一行
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

        # --- 6. 最终缺失值填充 ---
        fill_zeros = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy', 'years_of_service']
        for c in fill_zeros:
            if c not in df.columns: df[c] = 0.0
            df[c] = df[c].fillna(0.0)
        df['salary_2025'] = df['salary_2025'].fillna(76297) # 默认底薪
        if 'position' not in df.columns: df['position'] = 'G'
        df['position'] = df['position'].fillna('G')
        df['pos_mapped'] = df['position'].apply(self._map_pos)

        # 预计算：为每个球员设定“薪资谈判区间”
        # 假设球员愿意在 [原工资*0.8, 原工资*1.2] 之间浮动
        df['S_min'] = df['salary_2025'].apply(lambda x: max(CONFIG['SALARY_MIN_LEAGUE'], x * 0.8))
        df['S_max'] = df['salary_2025'].apply(lambda x: min(CONFIG['SALARY_MAX_LEAGUE'], x * 1.2))

        # 超级巨星通常不愿意降薪太多，修正下限
        mask_super = df['S_max'] > 200000
        df.loc[mask_super, 'S_min'] = df.loc[mask_super, 'salary_2025'] * 0.9

        # 计算基础评分 Vi
        df = self._calculate_scores(df)

        # 再次标记主队，防止名字写法不一致（如 "Indiana" vs "Fever"）
        target_clean = self.normalize_name(pd.Series([target_team_name]))[0]
        df['team_clean'] = self.normalize_name(df['team'])
        if 'fever' in target_clean:
            mask_home = df['team_clean'].str.contains('fever') | df['team_clean'].str.contains('ind')
            df.loc[mask_home, 'team_clean'] = target_clean
            df.loc[mask_home, 'team'] = target_team_name

        return df

    def _map_pos(self, p):
        """简单的位置映射函数"""
        p = str(p).upper()
        if 'C' in p: return 'C'
        if 'F' in p: return 'F'
        return 'G'

    def _calculate_scores(self, df):
        """计算 Vi_base (静态基础能力评分)"""
        def norm(col):
            min_v, max_v = col.min(), col.max()
            if max_v == min_v: return 0.5
            return (col - min_v) / (max_v - min_v)

        # ViA: 竞技数据加权
        df['ViA'] = (0.3 * norm(df['points']) + 0.2 * norm(df['rebounds']) + 
                     0.2 * norm(df['assists']) + 0.15 * norm(df['plus_minus']) + 
                     0.15 * norm(df['clutch_proxy']))
        
        # ViB: 商业价值 (薪资 + 人气)
        df['ViB'] = 0.5 * norm(df['salary_2025']) + 0.5 * norm(df['attendance'])
        # 给超级巨星（薪资高或名字带Clark）加成
        mask_star = (df['salary_2025'] > 200000) | (df['player'].str.contains('Clark', na=False, case=False))
        df.loc[mask_star, 'ViB'] *= 1.5
        df['ViB'] = df['ViB'].clip(0, 1)

        # ViH_base: 基础健康分 (出场时间越多，证明越耐操)
        df['ViH_base'] = 1 - (0.5 * norm(df['minutes']) * 0.3) 
        # ViP: 潜力分 (工龄越短，潜力越大)
        df['ViP'] = 1 - norm(df['years_of_service'])
        
        # 综合 Vi_base
        df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                         CONFIG['W_H'] * df['ViH_base'] + CONFIG['W_P'] * df['ViP'])
        df['Vi_base'] = df['Vi_base'].fillna(df['Vi_base'].mean())
        return df


# ==========================================
# 3. 求解器 (Solver)
# 核心算法部分：混合变量遗传算法
# ==========================================
class Individual:
    """个体类：代表一个可能的解决方案"""
    def __init__(self, n_total):
        # 变量 1 (离散): 选不选？ (0/1数组)
        self.dna_decisions = np.zeros(n_total, dtype=int)
        
        # 变量 2 (连续): 薪资系数 (0.0 - 1.0)，决定具体给多少钱
        self.dna_salary = np.random.rand(n_total)
        
        # 变量 3 (连续): 时间权重 (0.0 - 1.0)，决定具体上场多久
        self.dna_time = np.random.rand(n_total)

        # 存储计算结果
        self.objectives = None  # 目标函数值 [Cost, NegativePerformance]
        self.rank = 0           # NSGA-II 排序等级
        self.crowding_dist = 0  # 拥挤度距离
        self.violation = 0      # 约束违反程度
        self.z_score = -1e9     # 单一综合评分
        self.decoded_roster = None # 解码后的名单

class StrategicSolver:
    def __init__(self, pool, current_team_name):
        # 初始化：划分主队池和市场池
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        if 'fever' in target:
            mask_home = pool['team_clean'].str.contains('fever') | pool['team_clean'].str.contains('ind')
        else:
            mask_home = pool['team_clean'] == target

        self.current_roster = pool[mask_home].copy().reset_index(drop=True)
        # 为了加速计算，市场池只取前80名高分球员
        self.market_pool = pool[~mask_home].sort_values('Vi_base', ascending=False).head(80).reset_index(drop=True)
        
        self.full_pool = pd.concat([self.current_roster, self.market_pool]).reset_index(drop=True)
        self.n_total = len(self.full_pool)
        self.n_current = len(self.current_roster)
        print(f">>> [SolverInit] 混合变量优化引擎就绪。变量维度: 3 * {self.n_total}")

    def evaluate(self, ind):
        """
        【核心函数】评价一个个体的好坏
        """
        # 1. 解码决策变量：确定谁入选
        indices = np.where(ind.dna_decisions == 1)[0]
        
        # 如果没人入选，给予极大惩罚
        if len(indices) == 0:
            ind.objectives = [1e9, 1e9]
            ind.violation = 1e9
            return

        roster = self.full_pool.iloc[indices].copy()
        
        # 2. 解码薪资变量 S
        # 公式：实际薪资 = 最小值 + 系数 * (最大值 - 最小值)
        s_factors = ind.dna_salary[indices]
        roster['opt_salary'] = roster['S_min'] + s_factors * (roster['S_max'] - roster['S_min'])
        
        # 3. 解码时间变量 t
        t_weights = ind.dna_time[indices]
        total_weight = np.sum(t_weights)
        if total_weight == 0:
            roster['opt_minutes'] = CONFIG['TOTAL_GAME_MINUTES'] / len(roster)
        else:
            # 归一化：全队总时间必须等于 200 分钟
            roster['opt_minutes'] = (t_weights / total_weight) * CONFIG['TOTAL_GAME_MINUTES']
        
        # 软约束：限制单人时间在 [5, 38] 之间
        roster['opt_minutes'] = roster['opt_minutes'].clip(CONFIG['PLAYER_MIN_MINUTES'], CONFIG['PLAYER_MAX_MINUTES'])
        
        # 再次归一化，确保总和准确
        scale_factor = CONFIG['TOTAL_GAME_MINUTES'] / roster['opt_minutes'].sum()
        roster['opt_minutes'] *= scale_factor
        ind.decoded_roster = roster

        # --- 计算各种指标 ---
        TC = roster['opt_salary'].sum() # 总成本
        
        # 动态表现分：表现与时间挂钩
        time_ratio = roster['opt_minutes'] / 40.0 
        
        # 动态健康分：打得越久，风险越大 (ViH 降低)
        roster['ViH_dynamic'] = 1.0 - (time_ratio * 0.8) 
        
        # 贡献值 = 基础能力 * 时间比例
        roster['Vi_contribution'] = roster['Vi_base'] * time_ratio
        
        avg_Vi = roster['Vi_contribution'].sum()
        # 预估收益 (TR) = 对数模型
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        
        # 风险 = 薪资 * 受伤概率
        Risk = (roster['opt_salary'] * (1 - roster['ViH_dynamic'])).sum() * CONFIG['RISK_FACTOR']
        
        # --- 约束违反计算 (Violation) ---
        violation = 0
        count = len(roster)
        
        # 1. 人数约束
        if count < CONFIG['ROSTER_MIN']: violation += (CONFIG['ROSTER_MIN'] - count) * 1000
        if count > CONFIG['ROSTER_MAX']: violation += (count - CONFIG['ROSTER_MAX']) * 1000
        
        # 2. 工资帽约束
        if TC > CONFIG['SALARY_CAP']: violation += (TC - CONFIG['SALARY_CAP']) / 100
        
        # 3. 结构约束：高风险球员不能太多
        n_high_risk = len(roster[roster['ViH_base'] < 0.6])
        if n_high_risk > CONFIG['N_HIGHRISK_MAX']:
             violation += (n_high_risk - CONFIG['N_HIGHRISK_MAX']) * 5000

        # 4. 结构约束：潜力球员不能太少
        n_potential = len(roster[roster['ViP'] > 0.8])
        if n_potential < CONFIG['N_POTENTIAL_MIN']:
             violation += (CONFIG['N_POTENTIAL_MIN'] - n_potential) * 5000

        # 5. 交易逻辑约束：不能为了买人把全队都裁了
        # 计算新来的人数 (Acquired) 和裁掉的人数 (Waived)
        n_acquired = len(roster[roster.index >= self.n_current])
        n_retained = len(roster[roster.index < self.n_current])
        n_waived = self.n_current - n_retained
        
        if n_acquired > n_waived:
            violation += (n_acquired - n_waived) * 10000

        # 必须保留至少4名老队员
        if n_retained < 4: violation += (4 - n_retained) * 2000

        ind.violation = violation

        # --- 构造目标函数 ---
        # 目标 1: 最小化薪资
        obj_salary = TC
        
        # 目标 2: 最大化表现 (取负)
        # 位置适配度 (PosFit)
        pos_raw_score = 0
        pos_counts = roster['pos_mapped'].value_counts()
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: 
                pos_raw_score += 1
            else: 
                pos_raw_score -= 0.5 * abs(c - (min_p + max_p) / 2)
        PosFit = max(0.0, min(1.0, pos_raw_score * 0.3 + 0.5))

        # 风格适配度 (StyleFit): 阵容平均正负值越高越好
        avg_pm = roster['plus_minus'].mean()
        StyleFit = 1.0 / (1.0 + math.exp(-0.5 * avg_pm)) # Sigmoid 函数映射到 0-1

        ViF = 0.5 * PosFit + 0.5 * StyleFit
 
        term_financial = (TR - Risk) / 10000 
        performance_score = (CONFIG['MU_1'] * term_financial + 
                             CONFIG['MU_2'] * avg_Vi * 20 + 
                             CONFIG['MU_3'] * ViF * 50)
        
        obj_perf = -performance_score # 负号用于最小化

        ind.objectives = [obj_salary, obj_perf]
        ind.z_score = performance_score - violation

    def fast_nondominated_sort(self, population):
        """【NSGA-II 算法】非支配排序"""
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in population:
                # 约束优先原则：违反约束少的更优
                if p.violation < q.violation: dominates = True
                elif p.violation > q.violation: dominates = False
                else:
                    # 帕累托支配判断
                    dominates = ((p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and
                                 (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]))
                if dominates:
                    p.dominated_solutions.append(q)
                elif self._check_dominated(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        # 逐层剥离
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
        """【NSGA-II 算法】计算拥挤度距离，保持种群多样性"""
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
        """混合变异操作"""
        child = deepcopy(ind)
        
        # 1. 离散变量变异 (Swap / Flip)
        if random.random() < 0.3:
            ones = np.where(child.dna_decisions == 1)[0]
            zeros = np.where(child.dna_decisions == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                child.dna_decisions[np.random.choice(ones)] = 0
                child.dna_decisions[np.random.choice(zeros)] = 1
        
        if random.random() < rate:
            idx = random.randint(0, self.n_total - 1)
            child.dna_decisions[idx] = 1 - child.dna_decisions[idx]

        # 2. 连续变量变异 (高斯噪声)
        sigma = 0.1
        
        # 薪资变异
        mask_s = np.random.rand(self.n_total) < rate
        noise_s = np.random.randn(self.n_total) * sigma
        child.dna_salary[mask_s] += noise_s[mask_s]
        child.dna_salary = np.clip(child.dna_salary, 0, 1)
        
        # 时间变异
        mask_t = np.random.rand(self.n_total) < rate
        noise_t = np.random.randn(self.n_total) * sigma
        child.dna_time[mask_t] += noise_t[mask_t]
        child.dna_time = np.clip(child.dna_time, 0, 1)

        return child

    def solve(self, generations=100, pop_size=100):
        # 1. 初始化种群
        population = []
        for _ in range(pop_size):
            ind = Individual(self.n_total)
            
            # 启发式初始化：保证至少留一部分老队员
            target_size = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            n_keep = random.randint(4, min(self.n_current, target_size))
            n_buy = target_size - n_keep
            
            idxs_keep = np.random.choice(self.n_current, n_keep, replace=False)
            ind.dna_decisions[idxs_keep] = 1
            
            # 轮盘赌选择新球员
            if n_buy > 0:
                market_indices = np.arange(self.n_current, self.n_total)
                market_probs = self.full_pool.iloc[market_indices]['Vi_base'].values
                market_probs = market_probs / market_probs.sum()
                idxs_buy = np.random.choice(market_indices, n_buy, replace=False, p=market_probs)
                ind.dna_decisions[idxs_buy] = 1
            
            self.evaluate(ind)
            population.append(ind)

        # 2. 进化循环
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

        # 3. 输出最优解
        pareto_front = [p for p in population if p.rank == 0 and p.violation == 0]
        if not pareto_front:
            print("Warning: No fully feasible solution found. Using best approximate.")
            best_ind = max(population, key=lambda x: x.z_score)
        else:
            best_ind = max(pareto_front, key=lambda x: x.z_score)
            
        return best_ind

# ==========================================
# 4. 主程序入口
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
            
            # 区分保留球员和新引进球员
            final_roster['Source'] = final_roster.index.map(
                lambda x: 'Retained' if x < solver.n_current else 'Acquired')
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