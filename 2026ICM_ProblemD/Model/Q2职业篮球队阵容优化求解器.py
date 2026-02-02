"""
==============================================================================
项目名称: 职业篮球队阵容优化求解器 (Roster Optimization Solver)
学习对象: Hanxiang Li (CS Sophomore)
注释制作: Gemini

【项目核心逻辑】
这个程序的目标是：在有限的预算（工资帽）和人数限制下，从“现役球员”和“自由市场”中
挑选出一组球员，使得球队的综合评分（Z-Score）最大化。

【你必须掌握的关键方法 (Key Methods)】

1. 数据工程 (Data Engineering) - 使用 Pandas 库
   - ETL (Extract, Transform, Load): 读取 Excel/CSV，清洗脏数据。
   - 聚合 (Aggregation): 使用 `.groupby()` 将球员多场比赛的数据合并为赛季平均值。
   - 归一化 (Normalization): 将得分、篮板等不同量纲的数据缩放到 0-1 之间，方便加权计算。
   - 连表 (Merge): 类似于 SQL 的 Left Join，将球员数据和薪资数据合并。

2. 启发式优化算法 (Heuristic Optimization)
   - 这是一个“混合算法”，结合了 遗传算法 (GA) 和 模拟退火 (SA)。
   - 基因编码 (DNA Encoding): 用一串 0 和 1 代表球员是否被选中（1=选中，0=没选）。
   - 适应度函数 (Fitness Function/Objective Function): `get_fitness` 函数，用来给当前阵容打分。
   - 惩罚函数法 (Penalty Method): 算法如果选的人太贵（超工资帽），我们会扣除巨大的分数。
     这迫使算法自动学会“如何在规则内赢球”。

3. 算法核心机制
   - 变异 (Mutation): 随机交换球员，防止算法“死脑筋”只盯着局部最优解。
   - Metropolis 准则: 在 `solve` 函数中，即使新解变差了，也以一定概率接受它。
     这是模拟退火的核心，目的是为了跳出局部陷阱 (Local Optima)。

==============================================================================
"""

import pandas as pd     # DataFrames库，Python数据分析的神器，类似Excel
import numpy as np      # 数学计算库，处理矩阵和数组特别快
import random           # 随机数库，算法的“随机性”来源
import math             # 数学库，用于 log (对数) 和 exp (指数) 计算
import os               # 操作系统库，用于检查文件是否存在
from copy import deepcopy # 深拷贝，防止修改数据时影响到原始变量

# ==========================================
# 1. 配置参数 (Configuration)
# 类似于游戏的“设置”菜单，定义了规则和权重
# ==========================================
CONFIG = {
    # 权重参数：决定了我们看重球员的哪些方面
    # W_A: 竞技(Ability), W_B: 商业(Business), W_H: 健康(Health), W_P: 潜力(Potential)
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    
    # 目标函数系数：决定 Z-Score 的组成结构
    'MU_1': 0.40, # 战绩权重的占比
    'MU_2': 0.30, # 平均能力值的占比
    'MU_3': 0.30, # 阵容结构合理性的占比
    
    # 硬性约束 (Constraints)
    'SALARY_CAP': 1500000,  # 工资帽：全队总工资不能超过 150万
    'ROSTER_MIN': 11,       # 球队最少 11 人
    'ROSTER_MAX': 12,       # 球队最多 12 人
    
    # 位置需求：G(后卫), F(前锋), C(中锋) 各需要多少人
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    
    # 收益计算参数：用于模拟边际收益递减 (Log函数参数)
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
}

# ==========================================
# 2. 数据处理类 (DataProcessor)
# 这是一个 Class (类)，专门负责把脏乱的 Excel 变成干净的数据表
# ==========================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        # 构造函数：初始化时记下文件名
        self.stats_file = stats_file
        self.salary_file = salary_file
        
    def load_data(self):
        """ 读取并清洗数据 """
        print(f">>> 正在读取数据...")
        try:
            # 技巧：根据后缀名判断是用 read_excel 还是 read_csv
            reader_stats = pd.read_excel if self.stats_file.endswith('.xlsx') else pd.read_csv
            reader_salary = pd.read_excel if self.salary_file.endswith('.xlsx') else pd.read_csv
            
            # 读取文件到 DataFrame (表格)
            stats = reader_stats(self.stats_file)
            salary = reader_salary(self.salary_file)
        except Exception as e:
            # 异常处理：如果文件找不到或打不开，不让程序崩溃，而是打印错误
            print(f"文件读取错误: {e}")
            return None

        # 清洗列名：全部转小写，并去掉首尾空格 (防止 'Points ' 和 'points' 不匹配)
        stats.columns = [c.lower().strip() for c in stats.columns]
        salary.columns = [c.lower().strip() for c in salary.columns]
        
        # === 数据清洗核心 ===
        # 强制将这些列转换为数字，遇到非数字变成 NaN (Not a Number)，然后用 0 填充
        # errors='coerce' 是关键，它能强行把乱码变成空值
        for col in ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'season']:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors='coerce').fillna(0)
        
        # 自动找到数据里最新的赛季年份
        target_season = stats['season'].max() if 'season' in stats.columns else 2024
        print(f">>> 使用赛季数据: {target_season}")
        
        # 特征工程：计算“关键时刻表现代理变量” (Clutch Proxy)
        # 简单的线性组合：50% 得分能力 + 50% 正负值
        stats['clutch_proxy'] = stats['points'] * 0.5 + stats['plus_minus'] * 0.5
        
        # === 聚合 (Aggregation) ===
        # 一个球员一个赛季打了很多场，我们需要算出他的“平均表现”
        metrics = stats[stats['season'] == target_season].groupby('player').agg({
            'points': 'mean',       # 平均得分
            'rebounds': 'mean',     # 平均篮板
            'assists': 'mean',      # 平均助攻
            'plus_minus': 'mean',   # 平均正负值
            'minutes': 'mean',      # 平均上场时间
            'attendance': 'mean',   # 平均上座率
            'clutch_proxy': 'mean', # 平均关键表现
            'pos': 'first'          # 位置取第一个记录即可
        }).reset_index()            # 重置索引，让 player 变回普通列

        # 创建 'key' 列用于合并，统一小写去空格，作为唯一身份证
        metrics['key'] = metrics['player'].astype(str).str.lower().str.strip()
        salary['key'] = salary['player'].astype(str).str.lower().str.strip()
        
        # 处理薪资表中的缺失值，默认底薪 76000，默认工龄 2 年
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce').fillna(76000)
        salary['years_of_service'] = pd.to_numeric(salary['years_of_service'], errors='coerce').fillna(2)

        # === 连表 (Merge) ===
        # 类似于 SQL 的 Left Join，保留 Salary 表的所有人，把 Stats 数据拼上去
        df = pd.merge(salary, metrics, on='key', how='left', suffixes=('', '_stat'))
        
        # 再次填充缺失值 (防止有的球员有工资但没比赛数据)
        df['pos'] = df['pos'].fillna('G')
        num_cols = ['points', 'rebounds', 'minutes', 'attendance', 'clutch_proxy']
        for col in num_cols:
            df[col] = df[col].fillna(0)

        # 映射位置函数：把 'Guard/Forward' 这种复杂的变成简单的 'G', 'F', 'C'
        def map_pos(p):
            p = str(p).upper()
            if 'C' in p: return 'C'
            if 'F' in p: return 'F'
            return 'G'
        # .apply() 将函数应用到整列
        df['pos_mapped'] = df['pos'].apply(map_pos)

        # 标记渠道：这些原本就在 Excel 里的球员，如果不是主队，通常需要交易获得
        df['channel'] = 'Trade (q=3)'
        
        return df

    def generate_real_bench(self, df):
        """
        这个函数用于手动“造”一些数据。
        模拟真实的自由市场（Free Agent）或选秀（Draft）球员。
        """
        # 这是一个包含元组的列表，手动录入了一些低薪球员
        real_bench_players = [
            # 格式: (姓名, 位置, 薪资, 类型, 备注)
            ("Kate Martin", "G", 67000, "Rookie", "Fan Favorite"),
            # ... (中间省略) ...
            ("Queen Egbo", "C", 78831, "Veteran Min", "Los Angeles Sparks")
        ]

        dummies = [] # 用于暂存处理好的球员字典
        for name, pos, salary, p_type, note in real_bench_players:
            # 随机生成一个基础评分，模拟球探的不确定性
            base_vi = 0.20 + random.random() * 0.15 
            
            # 逻辑判断：如果是 Rookie (新秀) 就是 Draft (q=1)，否则是 Free Agent (q=2)
            if any(k in p_type for k in ["Rookie", "Prospect", "Scale"]):
                acq_channel = "Draft (q=1)"
            else:
                acq_channel = "Free Agent (q=2)"
            
            # 构建字典
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
        
        # 把列表转成 DataFrame
        dummy_df = pd.DataFrame(dummies)
        # 补齐列名，防止合并时报错
        for col in df.columns:
            if col not in dummy_df.columns:
                dummy_df[col] = 0
        
        # 纵向合并：把原始数据 df 和新造的数据 dummy_df 拼在一起
        return pd.concat([df, dummy_df], ignore_index=True)

    def calculate_values(self, df):
        """ 计算球员的各种价值评分 """
        df = df.copy() # 复制一份，不影响原数据
        
        # 定义归一化函数：(x - min) / (max - min)
        # 作用：把不管是得分(0-30)还是上场时间(0-40)，都压缩到 0-1 之间
        def norm(col):
            min_v, max_v = col.min(), col.max()
            if max_v == min_v: return 0.5 # 防止除以0
            denom = max_v - min_v
            return (col - min_v) / (denom if denom > 0 else 1)

        # 如果没有 Vi_base 列，就开始计算
        if 'Vi_base' not in df.columns or df['Vi_base'].isna().any():
            # ViA: 竞技价值 (得分、篮板、助攻等加权)
            df['ViA'] = (0.25 * norm(df['points']) + 0.20 * norm(df['rebounds']) + 
                         0.20 * norm(df['assists']) + 0.15 * norm(df['plus_minus']) + 
                         0.20 * norm(df['clutch_proxy']))
            # ViB: 商业价值 (上座率、薪资)
            df['ViB'] = 0.6 * norm(df['attendance']) + 0.4 * norm(df['salary_2025'])
            
            # 计算受伤风险
            injury_prob = norm(df['minutes']) * 0.5 + norm(df['years_of_service']) * 0.5
            df['ViH'] = 1 - (injury_prob * 0.4) # 健康度 = 1 - 风险
            
            # ViP: 潜力值 (工龄越低，潜力越高)
            df['ViP'] = 1 - norm(df['years_of_service'])
            
            # 加权汇总得到总基础分
            df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                             CONFIG['W_H'] * df['ViH'] + CONFIG['W_P'] * df['ViP'])
        
        df['Vi_base'] = df['Vi_base'].fillna(0.1) # 兜底，防止空值
        return df

# ==========================================
# 3. 求解器 (StrategicSolver)
# 这是程序的大脑，负责寻找最优解
# ==========================================
class StrategicSolver:
    def __init__(self, pool, current_team_name):
        # 初始化：把所有球员池分为“现在就在队里的”和“外面市场的”
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        
        self.current_roster = pool[pool['team_clean'] == target].copy() # 现役
        self.market_pool = pool[pool['team_clean'] != target].copy()    # 市场
        
        self.n_current = len(self.current_roster)
        self.n_market = len(self.market_pool)

    def get_fitness(self, dna):
        """
        【重要】目标函数 / 适应度函数
        输入：DNA (0/1 数组)
        输出：Z Score (分数，越高越好)
        """
        # 切片 DNA：前一部分是现役的保留情况，后一部分是市场的购买情况
        mask_keep = dna[:self.n_current] == 1
        mask_buy = dna[self.n_current:] == 1
        
        # 拼凑出当前的阵容
        roster = pd.concat([self.current_roster.iloc[mask_keep], self.market_pool.iloc[mask_buy]])
        count = len(roster)
        
        # 边界检查：如果一个人没选，给个极低的负分，让算法赶紧淘汰这个解
        if count == 0: return -1e12 
        
        TC = roster['salary_2025'].sum() # 总薪资
        pos_counts = roster['pos_mapped'].value_counts() # 统计各个位置有几个人
        
        # 计算位置适配度 (Fit Score)
        fit_score = 0
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: fit_score += 1 # 达标加分
            else: fit_score -= 0.5 * abs(c - (min_p+max_p)/2) # 不达标扣分
        
        # 将位置分归一化
        ViF = max(0, min(1, fit_score * 0.3 + 0.5))
        
        avg_Vi = roster['Vi_base'].mean() # 平均球员能力
        
        # 计算预估收益 (TR) - 使用对数 Log 模拟边际效益递减
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        
        # 计算风险
        Risk = (roster['salary_2025'] * (1 - roster['ViH'])).sum()
        
        # === 惩罚项 (Penalty) ===
        # 这是运筹学处理约束的常用方法：把“禁止做的事”变成“做了扣很多分”
        penalty = 0
        
        # 1. 人数惩罚 (少于11人或多于12人都要扣巨分)
        if count < CONFIG['ROSTER_MIN']:
            penalty += 1e7 + (CONFIG['ROSTER_MIN'] - count) * 1e6
        elif count > CONFIG['ROSTER_MAX']:
            penalty += 1e7 + (count - CONFIG['ROSTER_MAX']) * 1e6
            
        # 2. 工资帽惩罚 (超帽1块钱扣10分)
        if TC > CONFIG['SALARY_CAP']:
            penalty += (TC - CONFIG['SALARY_CAP']) * 10 
            
        # 3. 核心保留惩罚 (至少留2个老队员)
        n_kept = mask_keep.sum()
        if n_kept < 2: 
             penalty += (2 - n_kept) * 500000

        # === 计算最终 Z Score ===
        # 核心公式： (收益-风险)/成本 + 能力加分 + 结构加分 - 惩罚
        term1 = (TR - Risk) / (TC + 1e-5) * 50000
        Z = (CONFIG['MU_1'] * term1 + CONFIG['MU_2'] * avg_Vi * 5000 + CONFIG['MU_3'] * ViF * 5000) - penalty
        return Z

    def mutate(self, dna, rate=0.3):
        """
        【变异操作】
        随机改变 DNA，防止算法陷入“局部最优”
        """
        child = deepcopy(dna)
        
        # 策略 A: 交换 (Swap) - 保持总人数不变
        # 找到选了的人(1)和没选的人(0)的下标
        idx_ones = np.where(child == 1)[0]
        idx_zeros = np.where(child == 0)[0]
        
        # 60% 概率进行交换：踢掉一个，换进来一个
        if len(idx_ones) > 0 and len(idx_zeros) > 0 and random.random() < 0.6:
            i_one = np.random.choice(idx_ones)
            i_zero = np.random.choice(idx_zeros)
            child[i_one] = 0
            child[i_zero] = 1
            return child

        # 策略 B: 翻转 (Flip) - 改变总人数
        # 随机把某个位置取反 (0变1，或1变0)
        if random.random() < rate:
            total_len = len(dna)
            m_idx = random.randint(0, total_len-1)
            child[m_idx] = 1 - child[m_idx]
            
        return child

    def solve(self, generations=150, pop_size=60):
        """
        【主循环】执行优化算法
        generations: 迭代多少代
        pop_size: 种群有多少个个体
        """
        total_len = self.n_current + self.n_market
        pop = [] # 种群列表
        print(f">>> Solver Environment: Current {self.n_current}, Market (Classified) {self.n_market}")
        
        # 1. 初始化种群 (Generate Initial Population)
        # 随机生成 60 个初始方案
        for _ in range(pop_size):
            dna = np.zeros(total_len, dtype=int)
            
            # 随机保留 2-4 个老队员
            if self.n_current > 0:
                n_keep = random.randint(2, min(self.n_current, 4))
                idx_keep = np.random.choice(self.n_current, n_keep, replace=False)
                dna[idx_keep] = 1
            else:
                n_keep = 0
            
            # 随机从市场买人，补齐到 11-12 人
            current_count = n_keep
            target = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            needed = target - current_count
            
            if needed > 0 and self.n_market >= needed:
                idx_buy = np.random.choice(self.n_market, needed, replace=False)
                dna[self.n_current + idx_buy] = 1 # 注意索引偏移
            pop.append(dna)
            
        best_dna, best_fitness = None, -float('inf')
        T = 200.0 # 初始温度 (模拟退火参数)
        
        # 2. 迭代循环
        for gen in range(generations):
            # 给种群里每个方案打分
            fitnesses = [self.get_fitness(d) for d in pop]
            curr_max = max(fitnesses)
            
            # 记录历史最佳
            if curr_max > best_fitness:
                best_fitness = curr_max
                best_dna = deepcopy(pop[fitnesses.index(curr_max)])
                
            if gen % 30 == 0:
                print(f"Iter {gen:3d} | Best Z: {best_fitness:.2f}")
                
            new_pop = []
            while len(new_pop) < pop_size:
                # 锦标赛选择 (Tournament Selection): 随机抓3个，选分最高的做父母
                candidates = random.sample(pop, 3)
                candidates_fitness = [self.get_fitness(c) for c in candidates]
                parent = candidates[np.argmax(candidates_fitness)]
                
                # 变异产生孩子
                child = self.mutate(parent, rate=0.2)
                f_child = self.get_fitness(child)
                f_parent = self.get_fitness(parent)
                
                # === 模拟退火接受准则 (Metropolis Criterion) ===
                if f_child >= f_parent:
                    new_pop.append(child) # 孩子更好，直接要
                else:
                    # 孩子更差，也有一定概率要 (为了跳出局部最优)
                    # 温度 T 越低，接受差解的概率越小
                    prob = math.exp((f_child - f_parent) / T)
                    if random.random() < prob:
                        new_pop.append(child)
                    else:
                        new_pop.append(parent) # 否则保留父母
            
            pop = new_pop
            T *= 0.92 # 降温：随时间推移，越来越趋于稳定
            
        return best_dna, best_fitness

# ==========================================
# 4. 程序入口 (Main)
# ==========================================
if __name__ == "__main__":
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'   # 比赛数据文件
    f_salary = 'player_salaries_2025.xlsx'   # 薪资数据文件
    
    if os.path.exists(f_stats):
        # 1. 实例化处理类
        proc = DataProcessor(f_stats, f_salary)
        full_df = proc.load_data()
        
        if full_df is not None:
            # 2. 计算价值
            valued_df = proc.calculate_values(full_df)
            
            # 3. 添加手动录入的球员
            enhanced_df = proc.generate_real_bench(valued_df)
            
            # 4. 实例化求解器，指定“印第安纳狂热队”为当前主队
            my_team = "Indiana Fever"
            solver = StrategicSolver(enhanced_df, my_team)
            
            if solver.n_current == 0:
                print(f"Warning: Team '{my_team}' not found.")
            else:
                # 5. 运行优化
                best_dna, best_z = solver.solve()
                
                # 6. 解析结果 (把 0/1 变回人名)
                mask_keep = best_dna[:solver.n_current] == 1
                mask_buy = best_dna[solver.n_current:] == 1
                
                roster_keep = solver.current_roster.iloc[mask_keep]
                roster_buy = solver.market_pool.iloc[mask_buy]
                final_roster = pd.concat([roster_keep, roster_buy])
                
                # 7. 打印漂亮的结果
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