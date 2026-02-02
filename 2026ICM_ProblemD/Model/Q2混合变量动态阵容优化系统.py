"""
==============================================================================
项目名称: 混合变量动态阵容优化系统 (Mixed-Variable Dynamic Roster Optimizer)
学习对象: Hanxiang Li (CS Sophomore / Math Modeling)
注释制作: Gemini

【核心升级：混合变量优化】
之前的版本只有“选与不选”。这个版本引入了真实世界的复杂性：
1. 决策变量 (Binary): 选谁？(0/1)
2. 薪资变量 (Continuous): 给多少钱？(S) -> 在 S_min 和 S_max 之间浮动。
3. 时间变量 (Continuous): 上场多久？(t) -> 决定了球员的贡献值和受伤风险。

【算法核心：NSGA-II 框架】
使用了多目标优化算法 NSGA-II 的思想，特别是以下机制：
1. 非支配排序 (Non-dominated Sorting): 将解分层。第一层(Rank 0)是最好的，
   即“没有任何其他解能在所有方面都比我强”。
2. 拥挤度距离 (Crowding Distance): 在同层级中，优先选择“独特”的解，
   保持种群的多样性，防止早熟收敛。
3. 约束支配 (Constraint Dominance): 优先满足约束条件（如工资帽、人数），
   在满足约束的前提下再比拼分数。

【动态评价体系】
- 以前：球员价值 Vi 是固定的。
- 现在：球员价值是动态的。Vi_contribution = Vi_base * (上场时间 / 40)。
  这意味着：如果你选了巨星但只让他打5分钟，他的价值会很低；
  如果你让老将打40分钟，他的受伤风险(Risk)会暴增。

【应用场景】
这是典型的 MCM/ICM 建模竞赛题目解法，适用于资源分配、投资组合优化等复杂问题。
==============================================================================
"""
import pandas as pd     # 数据处理库
import numpy as np      # 矩阵计算库
import random           # 随机数生成
import math             # 数学函数
import os               # 系统路径处理
from copy import deepcopy # 深拷贝，用于遗传算法繁衍后代
import warnings         # 用于忽略烦人的警告信息

warnings.filterwarnings('ignore') # 忽略 Pandas 的一些版本兼容性警告

# ==========================================
# 1. 配置参数 (CONFIG)
# 定义了比赛规则、物理限制和算法权重
# ==========================================
CONFIG = {
    # 权重：竞技(A), 商业(B), 健康(H), 潜力(P), 结构(F)
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    
    # 目标函数系数：MU_1(性价比), MU_2(总能力), MU_3(结构分)
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    
    'SALARY_CAP': 1500000,   # 工资帽 150万
    'ROSTER_MIN': 11,        # 最少 11 人
    'ROSTER_MAX': 12,        # 最多 12 人
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)}, # 位置限制
    
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5, # 收益计算参数
    
    # === 新增：薪资与时间约束 ===
    'SALARY_MIN_LEAGUE': 64154,   # 联盟规定的最低工资
    'SALARY_MAX_LEAGUE': 241984,  # 顶薪限制
    'TOTAL_GAME_MINUTES': 200,    # 一场比赛总时间：5个人 * 40分钟 = 200分钟
    'PLAYER_MAX_MINUTES': 38.0,   # 人的体能极限，不能打满40分钟
    'PLAYER_MIN_MINUTES': 5.0 ,   # 只要上场，至少打5分钟（垃圾时间）
    
    # === 新增：高级结构约束 ===
    'N_HIGHRISK_MAX': 2,   # 玻璃人（健康分低）最多只能有2个
    'N_POTENTIAL_MIN': 2   # 潜力股（年轻球员）至少要有2个
}

# ==========================================
# 2. 数据处理 (DataProcessor)
# 负责脏数据的清洗、合并和基础打分
# ==========================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file
        self.salary_file = salary_file

    def normalize_name(self, series):
        """辅助函数：把名字转小写、去空格、去标点，确保 'A.J. Wilson' 能匹配 'aj wilson'"""
        return series.astype(str).str.lower().str.strip().str.replace('.', '', regex=False)

    def load_and_process(self, target_team_name="Indiana Fever"):
        print(f">>> [DataEngine] 正在加载数据，目标主队: {target_team_name}...")
        
        # --- 1. 读取原始文件 ---
        try:
            # 自动判断是 Excel 还是 CSV
            read_func = pd.read_excel if self.stats_file.endswith('.xlsx') else pd.read_csv
            stats_raw = read_func(self.stats_file)
            
            read_func = pd.read_excel if self.salary_file.endswith('.xlsx') else pd.read_csv
            salary_raw = read_func(self.salary_file)
        except Exception as e:
            print(f"!!! 文件读取失败: {e}")
            return None

        # 清洗列名
        stats_raw.columns = [c.lower().strip() for c in stats_raw.columns]
        salary_raw.columns = [c.lower().strip() for c in salary_raw.columns]
        
        # --- 2. 提取统计数据 ---
        # 如果有赛季列，只取最新赛季的数据
        if 'season' in stats_raw.columns:
            target_season = stats_raw['season'].max()
            stats = stats_raw[stats_raw['season'] == target_season].copy()
        else:
            stats = stats_raw.copy()

        # 确保关键数据列都是数字，如果不是就填0
        stat_cols = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy']
        for c in stat_cols:
            if c not in stats.columns: stats[c] = 0.0
            stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)

        # 聚合：同名球员多行数据取平均值
        metrics = stats.groupby('player').agg({
            'points': 'mean', 'rebounds': 'mean', 'assists': 'mean', 
            'plus_minus': 'mean', 'minutes': 'mean', 'attendance': 'mean',
            'clutch_proxy': 'mean', 'team': 'last' 
        }).reset_index()

        # --- 3. 提取薪资数据 ---
        salary = salary_raw.copy()
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce')
        # 默认工龄设为1年
        salary['years_of_service'] = pd.to_numeric(salary.get('years_of_service', 0), errors='coerce').fillna(1)
        
        # --- 4. 合并数据 ---
        # 生成标准化 key 用于匹配
        metrics['key'] = self.normalize_name(metrics['player'])
        salary['key'] = self.normalize_name(salary['player'])
        
        # 外连接 (outer join)：保证有比赛数据但没薪资，或者有薪资没比赛数据的人都保留
        df = pd.merge(metrics, salary[['key', 'salary_2025', 'years_of_service', 'position', 'team']], 
                      on='key', how='outer', suffixes=('', '_sal'))
        
        # 数据补全逻辑
        df['player'] = df['player'].fillna(salary['player'])
        df['team'] = df['team'].fillna(df['team_sal']).fillna('Free Agent')
        
        # 手动修复字典：这一步在真实项目中是为了修正爬虫爬不到的特殊数据
        # 比如 Caitlin Clark 是新秀，数据可能在不同表格里对不上
        manual_fixes = {
            'aliyah boston': {'salary': 99000, 'pos': 'C', 'team': target_team_name, 'pts': 14.5, 'reb': 8.4, 'min': 30},
            # ... (省略了中间大量的字典内容，逻辑是一样的) ...
            'makayla timpson': {'salary': 65000, 'pos': 'F', 'team': 'Free Agent'},
        }

        # 应用手动修复
        for name, data in manual_fixes.items():
            key = self.normalize_name(pd.Series([name]))[0]
            mask = df['key'] == key
            
            if mask.any():
                # 如果找到了人，更新数据
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

        # --- 6. 全局缺失值最终填充 ---
        fill_zeros = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy', 'years_of_service']
        for c in fill_zeros:
            if c not in df.columns: df[c] = 0.0
            df[c] = df[c].fillna(0.0)
        df['salary_2025'] = df['salary_2025'].fillna(76297)
        if 'position' not in df.columns: df['position'] = 'G'
        df['position'] = df['position'].fillna('G')
        df['pos_mapped'] = df['position'].apply(self._map_pos)

        # === 关键：计算连续变量的边界 ===
        # 即使是现有球员，薪资也可以谈判。我们假设可以在 [原薪资*0.8, 原薪资*1.2] 之间波动
        df['S_min'] = df['salary_2025'].apply(lambda x: max(CONFIG['SALARY_MIN_LEAGUE'], x * 0.8))
        df['S_max'] = df['salary_2025'].apply(lambda x: min(CONFIG['SALARY_MAX_LEAGUE'], x * 1.2))

        # 特殊处理：超级巨星（顶薪球员）很难降薪
        mask_super = df['S_max'] > 200000
        df.loc[mask_super, 'S_min'] = df.loc[mask_super, 'salary_2025'] * 0.9

        # 计算基础评分
        df = self._calculate_scores(df)

        # 标记主队
        target_clean = self.normalize_name(pd.Series([target_team_name]))[0]
        df['team_clean'] = self.normalize_name(df['team'])
        if 'fever' in target_clean:
            mask_home = df['team_clean'].str.contains('fever') | df['team_clean'].str.contains('ind')
            df.loc[mask_home, 'team_clean'] = target_clean
            df.loc[mask_home, 'team'] = target_team_name

        print(f"    - 数据加载完成，全联盟共 {len(df)} 名球员。")
        return df

    def _map_pos(self, p):
        """简单粗暴的位置映射：包含C就是C，包含F就是F，否则是G"""
        p = str(p).upper()
        if 'C' in p: return 'C'
        if 'F' in p: return 'F'
        return 'G'

    def _calculate_scores(self, df):
        """计算 Vi_base (静态基础能力)，归一化处理"""
        def norm(col):
            min_v, max_v = col.min(), col.max()
            if max_v == min_v: return 0.5
            return (col - min_v) / (max_v - min_v)

        # ViA: 竞技分
        df['ViA'] = (0.3 * norm(df['points']) + 0.2 * norm(df['rebounds']) + 
                     0.2 * norm(df['assists']) + 0.15 * norm(df['plus_minus']) + 
                     0.15 * norm(df['clutch_proxy']))
        
        # ViB: 商业分 (薪资高的通常名气大，上座率高的名气大)
        df['ViB'] = 0.5 * norm(df['salary_2025']) + 0.5 * norm(df['attendance'])
        # 给超级巨星（比如 Clark）额外的商业加成
        mask_star = (df['salary_2025'] > 200000) | (df['player'].str.contains('Clark', na=False, case=False))
        df.loc[mask_star, 'ViB'] *= 1.5
        df['ViB'] = df['ViB'].clip(0, 1)

        # ViH_base: 基础健康分 (上场时间越长的，理论上越耐操，但也越累)
        df['ViH_base'] = 1 - (0.5 * norm(df['minutes']) * 0.3) 
        # ViP: 潜力分 (工龄越短潜力越大)
        df['ViP'] = 1 - norm(df['years_of_service'])
        
        # 综合得到 Vi_base
        df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                         CONFIG['W_H'] * df['ViH_base'] + CONFIG['W_P'] * df['ViP'])
        df['Vi_base'] = df['Vi_base'].fillna(df['Vi_base'].mean())
        return df


# ==========================================
# 3. 求解器 (Solver) - 核心部分
# ==========================================
class Individual:
    """
    个体类：这就是混合变量优化的载体。
    一个 Individual 代表一种可能的球队配置方案。
    """
    def __init__(self, n_total):
        # 变量 1 (离散): 选不选这个球员？ (0 或 1)
        self.dna_decisions = np.zeros(n_total, dtype=int)
        
        # 变量 2 (连续): 薪资系数 (0.0 - 1.0)
        # 实际薪资 = 最小值 + 系数 * (最大值 - 最小值)
        self.dna_salary = np.random.rand(n_total)
        
        # 变量 3 (连续): 时间权重 (0.0 - 1.0)
        # 实际时间 = (权重 / 总权重) * 200分钟
        self.dna_time = np.random.rand(n_total)

        # 存储计算结果
        self.objectives = None  # 目标函数值 [成本, 负的表现分]
        self.rank = 0           # Pareto 排序等级
        self.crowding_dist = 0  # 拥挤度距离 (用于保持多样性)
        self.violation = 0      # 违反约束的程度 (罚分)
        self.z_score = -1e9     # 最终单目标得分
    
        self.decoded_roster = None # 解码后的详细名单

class StrategicSolver:
    def __init__(self, pool, current_team_name):
        # 初始化：把球员池分为“自己人”和“外人”
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        if 'fever' in target:
            mask_home = pool['team_clean'].str.contains('fever') | pool['team_clean'].str.contains('ind')
        else:
            mask_home = pool['team_clean'] == target

        self.current_roster = pool[mask_home].copy().reset_index(drop=True)
        # 市场池只取前80名，为了减少计算量，把太差的过滤掉
        self.market_pool = pool[~mask_home].sort_values('Vi_base', ascending=False).head(80).reset_index(drop=True)
        
        # 合并成一个大池子供算法选择
        self.full_pool = pd.concat([self.current_roster, self.market_pool]).reset_index(drop=True)
        self.n_total = len(self.full_pool)
        self.n_current = len(self.current_roster)
        print(f">>> [SolverInit] 混合变量优化引擎就绪。变量维度: 3 * {self.n_total}")

    def evaluate(self, ind):
        """
        【评价函数】最核心的数学计算
        将 DNA (0-1, float, float) 映射为具体的球队名单，并计算分数
        """
        # 1. 解码二值变量：确定谁在名单里
        indices = np.where(ind.dna_decisions == 1)[0]
        
        if len(indices) == 0:
            ind.objectives = [1e9, 1e9]
            ind.violation = 1e9
            return

        roster = self.full_pool.iloc[indices].copy()
        
        # 2. 解码连续变量 S (薪资)
        s_factors = ind.dna_salary[indices]
        # 公式：S_opt = S_min + alpha * (S_max - S_min)
        roster['opt_salary'] = roster['S_min'] + s_factors * (roster['S_max'] - roster['S_min'])
        
        # 3. 解码连续变量 t (时间)
        t_weights = ind.dna_time[indices]
        total_weight = np.sum(t_weights)
        if total_weight == 0:
            roster['opt_minutes'] = CONFIG['TOTAL_GAME_MINUTES'] / len(roster)
        else:
            # 归一化：所有人的时间加起来必须等于 200 分钟
            roster['opt_minutes'] = (t_weights / total_weight) * CONFIG['TOTAL_GAME_MINUTES']
        
        # 软约束：单人不能打太久(38)，也不能打太少(5)
        roster['opt_minutes'] = roster['opt_minutes'].clip(CONFIG['PLAYER_MIN_MINUTES'], CONFIG['PLAYER_MAX_MINUTES'])
        
        # 再次归一化：因为 clip 操作可能导致总和不等于 200
        scale_factor = CONFIG['TOTAL_GAME_MINUTES'] / roster['opt_minutes'].sum()
        roster['opt_minutes'] *= scale_factor
        
        ind.decoded_roster = roster # 保存解码结果

        # --- 计算目标 ---
        TC = roster['opt_salary'].sum() # 总成本
        
        # 动态表现分计算：表现与上场时间挂钩
        time_ratio = roster['opt_minutes'] / 40.0 # 归一化时间
        
        # 动态健康分 ViH: 上场时间越长，健康分(1-风险)越低
        roster['ViH_dynamic'] = 1.0 - (time_ratio * 0.8) 
        
        # 综合贡献值 = 基础能力 * 上场时间比例
        roster['Vi_contribution'] = roster['Vi_base'] * time_ratio
        
        avg_Vi = roster['Vi_contribution'].sum()
        # 商业价值通常和全队总人气有关，用对数收益模型
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        
        # 风险计算 Risk = 薪资 * 受伤概率
        Risk = (roster['opt_salary'] * (1 - roster['ViH_dynamic'])).sum() * CONFIG['RISK_FACTOR']
        
        # === 计算约束违反程度 (Violation) ===
        violation = 0
        count = len(roster)
        
        # 1. 基础人数约束
        if count < CONFIG['ROSTER_MIN']: violation += (CONFIG['ROSTER_MIN'] - count) * 1000
        if count > CONFIG['ROSTER_MAX']: violation += (count - CONFIG['ROSTER_MAX']) * 1000
        
        # 2. 工资帽约束
        if TC > CONFIG['SALARY_CAP']: violation += (TC - CONFIG['SALARY_CAP']) / 100
        
        # 3. 高风险球员限制 (比如太老或太脆的不能超过2个)
        n_high_risk = len(roster[roster['ViH_base'] < 0.6])
        if n_high_risk > CONFIG['N_HIGHRISK_MAX']:
             violation += (n_high_risk - CONFIG['N_HIGHRISK_MAX']) * 5000 

        # 4. 潜力球员下限 (年轻球员必须至少2个)
        n_potential = len(roster[roster['ViP'] > 0.8])
        if n_potential < CONFIG['N_POTENTIAL_MIN']:
             violation += (CONFIG['N_POTENTIAL_MIN'] - n_potential) * 5000 

        # 5. 交易平衡逻辑
        # Acquired (交易获得) vs Waived (裁掉)
        n_acquired = len(roster[roster.index >= self.n_current]) # 新来的
        n_retained = len(roster[roster.index < self.n_current])  # 留下的
        n_waived = self.n_current - n_retained                   # 走的
        
        # 逻辑约束：你买进来的人数不能超过你裁掉的人数太多 (防止名单爆炸)
        if n_acquired > n_waived:
            violation += (n_acquired - n_waived) * 10000

        # 核心保留约束：至少留4个老队员
        if n_retained < 4: violation += (4 - n_retained) * 2000

        ind.violation = violation

        # 构造多目标优化的两个目标：
        # Obj 1: 总薪资 (越小越好)
        obj_salary = TC
        
        # Obj 2: 表现分 (越大越好，所以取负数变成最小化问题)
        # 计算位置匹配度 PosFit
        pos_raw_score = 0
        pos_counts = roster['pos_mapped'].value_counts()
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: 
                pos_raw_score += 1
            else: 
                pos_raw_score -= 0.5 * abs(c - (min_p + max_p) / 2)
        PosFit = max(0.0, min(1.0, pos_raw_score * 0.3 + 0.5))

        # 计算风格匹配度 StyleFit (正负值越高越好)
        avg_pm = roster['plus_minus'].mean()
        StyleFit = 1.0 / (1.0 + math.exp(-0.5 * avg_pm)) # Sigmoid 函数

        ViF = 0.5 * PosFit + 0.5 * StyleFit
 
        # 最终性能分公式
        term_financial = (TR - Risk) / 10000 
        performance_score = (CONFIG['MU_1'] * term_financial + 
                             CONFIG['MU_2'] * avg_Vi * 20 + 
                             CONFIG['MU_3'] * ViF * 50)
        
        obj_perf = -performance_score # 取负数

        ind.objectives = [obj_salary, obj_perf]
        # Z-Score 是用来做最终排名的单指标
        ind.z_score = performance_score - violation

    def fast_nondominated_sort(self, population):
        """
        【NSGA-II 核心 1】快速非支配排序
        把解分成不同的“层级”(Rank)。
        Rank 0 的解最好，Rank 1 次之...
        """
        fronts = [[]]
        for p in population:
            p.domination_count = 0 # 支配我的个数
            p.dominated_solutions = [] # 我支配的解
            for q in population:
                # 约束优先原则：如果违反约束少，就更优
                if p.violation < q.violation: dominates = True
                elif p.violation > q.violation: dominates = False
                else:
                    # 如果约束程度一样，比较目标函数
                    # p 支配 q 当且仅当：p在所有目标上<=q，且至少有一个目标<q
                    dominates = ((p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and
                                 (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]))
                if dominates:
                    p.dominated_solutions.append(q)
                elif self._check_dominated(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        # 剥洋葱：把 Rank 0 拿走，剩下的里面找 Rank 1，以此类推
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
        """辅助函数：检查 p 是否支配 q"""
        if p.violation < q.violation: return True
        if p.violation > q.violation: return False
        return ((p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and
                (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]))

    def calculate_crowding_distance(self, front):
        """
        【NSGA-II 核心 2】拥挤度距离计算
        在同一层级(Rank)中，我们喜欢比较“疏松”的解，这样能保持种群多样性。
        """
        l = len(front)
        if l == 0: return
        for p in front: p.crowding_dist = 0
        for m in range(2): # 对两个目标分别计算
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_dist = float('inf') # 边界上的点保留
            front[-1].crowding_dist = float('inf')
            obj_min = front[0].objectives[m]
            obj_max = front[-1].objectives[m]
            if obj_max == obj_min: continue
            for i in range(1, l-1):
                # 距离 = 前后两个点的差值
                front[i].crowding_dist += (front[i+1].objectives[m] - front[i-1].objectives[m]) / (obj_max - obj_min)

    def crowd_comparison(self, p1, p2):
        """比较两个个体：先看 Rank，Rank 一样看拥挤度"""
        if p1.rank < p2.rank: return True # Rank 越小越好
        if p1.rank == p2.rank and p1.crowding_dist > p2.crowding_dist: return True # 距离越大越好
        return False

    def mutate(self, ind, rate=0.1):
        """
        【变异算子】混合变量变异
        针对不同类型的变量，使用不同的变异策略
        """
        child = deepcopy(ind)
        
        # 1. 二值变异 (Binary Mutation)
        # 策略：Swap (交换)，保证选中人数不会剧烈波动
        if random.random() < 0.3:
            ones = np.where(child.dna_decisions == 1)[0]
            zeros = np.where(child.dna_decisions == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                child.dna_decisions[np.random.choice(ones)] = 0 # 踢掉一个
                child.dna_decisions[np.random.choice(zeros)] = 1 # 加进来一个
        
        # 策略：Flip (翻转)，小概率随机改变状态
        if random.random() < rate:
            idx = random.randint(0, self.n_total - 1)
            child.dna_decisions[idx] = 1 - child.dna_decisions[idx]

        # 2. 连续变量变异 (Continuous Mutation) - 高斯噪声
        sigma = 0.1 # 变异强度
        
        # 薪资变异：在原基础上加一点点随机扰动
        mask_s = np.random.rand(self.n_total) < rate
        noise_s = np.random.randn(self.n_total) * sigma
        child.dna_salary[mask_s] += noise_s[mask_s]
        child.dna_salary = np.clip(child.dna_salary, 0, 1) # 必须限制在 [0,1] 范围内
        
        # 时间变异
        mask_t = np.random.rand(self.n_total) < rate
        noise_t = np.random.randn(self.n_total) * sigma
        child.dna_time[mask_t] += noise_t[mask_t]
        child.dna_time = np.clip(child.dna_time, 0, 1) 

        return child

    def solve(self, generations=100, pop_size=100):
        #         # 算法流程：初始化 -> 评价 -> 排序 -> 选择 -> 交叉变异 -> 新一代
        
        # 1. 初始化种群
        population = []
        for _ in range(pop_size):
            ind = Individual(self.n_total)
            
            # 启发式初始化：保证一开始生成的解大致靠谱
            target_size = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            
            # 强制保留一部分老队员
            n_keep = random.randint(4, min(self.n_current, target_size))
            n_buy = target_size - n_keep
            
            idxs_keep = np.random.choice(self.n_current, n_keep, replace=False)
            ind.dna_decisions[idxs_keep] = 1
            
            # 根据 Vi_base 轮盘赌选择新球员（优先选分高的）
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
            # 对当前种群进行非支配排序
            fronts = self.fast_nondominated_sort(population)
            for front in fronts: self.calculate_crowding_distance(front)
            
            offspring = []
            while len(offspring) < pop_size:
                # 锦标赛选择 (Tournament Selection)
                p1 = random.choice(population)
                p2 = random.choice(population)
                parent = p1 if self.crowd_comparison(p1, p2) else p2
                
                # 变异产生后代
                child = self.mutate(parent, rate=0.1)
                self.evaluate(child)
                offspring.append(child)
            
            # 精英保留策略：父代 + 子代 混合，选出最好的 pop_size 个
            combined = population + offspring
            fronts = self.fast_nondominated_sort(combined)
            new_pop = []
            for front in fronts:
                self.calculate_crowding_distance(front)
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(front)
                else:
                    # 如果这一层加进去会超员，则按拥挤度排序截取
                    front.sort(key=lambda x: x.crowding_dist, reverse=True)
                    new_pop.extend(front[:pop_size - len(new_pop)])
                    break
            population = new_pop

            if gen % 10 == 0:
                best_z = max(p.z_score for p in population)
                print(f"    Iter {gen:3d} | Best Z: {best_z:.4f}")

        # 3. 输出结果
        # 优先选 Rank 0 且没有违反约束的解
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
            
            # 运行求解
            best_ind = solver.solve(generations=120, pop_size=100)
            final_roster = best_ind.decoded_roster.copy()
            
            # 处理结果展示：区分是留队的还是买来的
            final_roster['Source'] = final_roster.index.map(
                lambda x: 'Retained' if x < solver.n_current else 'Acquired')
            
            # 找出被交易走的人
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
            
            # 格式化打印
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