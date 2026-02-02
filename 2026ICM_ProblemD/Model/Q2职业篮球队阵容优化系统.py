"""
==============================================================================
项目名称: 职业篮球队阵容优化系统 (Roster Optimization Solver)
适用场景: 体育管理 (Sports Management) / 运筹学 (Operations Research)

【核心功能】
该脚本在工资帽 (Salary Cap) 和阵容人数限制下，通过算法自动从“现有球员”和“自由市场”中
选择最佳球员组合，以最大化球队的综合评分 (Z-Score)。

【涉及的关键技术与方法】

1. 数据工程 (Data Engineering) - 使用 Pandas
   - 数据清洗 (Data Cleaning): 处理空值 (fillna)，标准化列名。
   - 数据聚合 (Aggregation): 使用 groupby 将球员的多场比赛数据合并为赛季平均数据。
   - 数据融合 (Merge): 将球员的比赛数据 (Stats) 与薪资数据 (Salary) 结合。
   - 特征工程 (Feature Engineering): 计算球员的高阶数据，如“关键时刻表现代理变量” (Clutch Proxy)。

2. 数学建模 (Mathematical Modeling)
   - 归一化 (Min-Max Normalization): 将得分、篮板等不同量纲的数据映射到 0-1 之间，
     方便进行加权计算。
   - 多目标加权评价 (Weighted Scoring): 通过权重 (W_A, W_B...) 计算球员的综合价值 Vi。
     - ViA: 竞技表现 (Points, Rebounds...)
     - ViB: 商业价值 (Attendance, Salary)
     - ViH: 健康程度 (Minutes, Age)
   - 惩罚函数法 (Penalty Function): 将硬约束（如工资帽、人数限制）转化为软约束。
     如果超出工资帽，通过在分数上扣除巨大的罚分 (Penalty) 来迫使算法寻找合规解。

3. 启发式优化算法 (Heuristic Optimization)
   - 这是一个混合算法，结合了 遗传算法 (Genetic Algorithm, GA) 和 模拟退火 (Simulated Annealing, SA) 的思想。
   - 基因编码 (DNA Encoding): 使用 0/1 数组表示球员是否被选中 (1=选中, 0=落选)。
   - 变异 (Mutation): 随机交换球员 (Swap) 或 改变选中状态 (Flip) 来探索新的阵容组合。
   - 接受准则 (Metropolis Criterion): 
     - 如果新阵容更好，直接保留。
     - 如果新阵容更差，以一定概率接受它（为了跳出局部最优解），这个概率随温度 T 降低而减小。

作者: Hanxiang Li
日期: 2026-02-01
==============================================================================
"""
import pandas as pd     # 用于处理表格数据（DataFrames），就像在Python里操作Excel
import numpy as np      # 用于数值计算和矩阵操作，速度快
import random           # 用于生成随机数，是算法随机性的来源
import math             # 用于数学函数，如 log (对数) 和 exp (指数)
import os               # 用于操作系统交互，比如检查文件是否存在
from copy import deepcopy # 用于深拷贝，防止修改新变量时影响到旧变量

# ==========================================
# 1. 配置参数 (CONFIG)
# 这里定义了所有的“游戏规则”和权重
# ==========================================
CONFIG = {
    # 权重参数：用于计算球员综合价值 (Vi)
    # W_A: 竞技能力权重, W_B: 商业价值权重, W_H: 健康权重, W_P/W_F: 潜力/适应性权重
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    
    # 目标函数中的系数，用于平衡 战绩(Term1)、平均能力(Avg_Vi) 和 结构合理性(ViF)
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    
    # 约束条件
    'SALARY_CAP': 1500000,  # 工资帽：球队总薪资不能超过 150万
    'ROSTER_MIN': 11,       # 阵容最少人数
    'ROSTER_MAX': 12,       # 阵容最多人数
    
    # 位置需求：G(后卫)需要4-6人，F(前锋)需要4-6人，C(中锋)需要2-3人
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    
    # TR (Total Revenue) 计算公式中的参数，模拟边际收益递减
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
}

# ==========================================
# 2. 数据处理类 (DataProcessor)
# 负责读取、清洗、合并数据，并计算球员的基础价值
# ==========================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        # 初始化时传入两个文件名：比赛数据和薪资数据
        self.stats_file = stats_file
        self.salary_file = salary_file
        
    def load_data(self):
        print(f">>> 正在读取数据...")
        try:
            # 根据文件后缀名判断是读取 Excel 还是 CSV
            reader_stats = pd.read_excel if self.stats_file.endswith('.xlsx') else pd.read_csv
            reader_salary = pd.read_excel if self.salary_file.endswith('.xlsx') else pd.read_csv
            stats = reader_stats(self.stats_file)
            salary = reader_salary(self.salary_file)
        except Exception as e:
            # 如果文件读取失败，打印错误信息并返回 None
            print(f"文件读取错误: {e}")
            return None

        # 清洗列名：转为小写并去除首尾空格，防止 'Points ' 和 'points' 不匹配的情况
        stats.columns = [c.lower().strip() for c in stats.columns]
        salary.columns = [c.lower().strip() for c in salary.columns]

        # 定义我们需要用到的关键数据列
        required_cols = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy']
        for col in required_cols:
            if col in stats.columns:
                # 将数据转为数字类型，如果出错（比如原本是字符串）则转为NaN，然后用0填充
                stats[col] = pd.to_numeric(stats[col], errors='coerce').fillna(0)
            else:
                # 如果原始数据里没这一列，自动补全为0，防止后续 agg (聚合) 报错
                stats[col] = 0.0

        # 确保 pos (位置) 列也存在，如果没有则默认为 'G'
        if 'pos' not in stats.columns:
            stats['pos'] = 'G'

        # 自动识别数据中最新的赛季（如果没有season列，默认为2024）
        target_season = stats['season'].max() if 'season' in stats.columns else 2024
        print(f">>> 使用赛季数据: {target_season}")

        # === 核心步骤：数据聚合 (Aggregation) ===
        # 一个球员一个赛季打了很多场球，我们要算出他的平均表现
        metrics = stats[stats['season'] == target_season].groupby('player').agg({
            'points': 'mean',       # 平均得分
            'rebounds': 'mean',     # 平均篮板
            'assists': 'mean',      # 平均助攻
            'plus_minus': 'mean',   # 平均正负值
            'minutes': 'mean',      # 平均上场时间
            'attendance': 'mean',   # 平均上座率（商业价值）
            'clutch_proxy': 'mean', # 关键时刻表现
            'pos': 'first'          # 位置取第一个记录即可
        }).reset_index()            # 重置索引，让 'player' 变回普通列

        # 创建 'key' 列用于合并，统一转换为小写去空格，作为唯一标识符
        metrics['key'] = metrics['player'].astype(str).str.lower().str.strip()
        salary['key'] = salary['player'].astype(str).str.lower().str.strip()
        
        # 处理薪资表中的缺失值，如果没有薪资则设为底薪76000，工龄默认为2年
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce').fillna(76000)
        salary['years_of_service'] = pd.to_numeric(salary['years_of_service'], errors='coerce').fillna(2)

        # === 核心步骤：合并 (Merge) ===
        # 类似于 SQL 的 LEFT JOIN，以薪资表为主，把数据表拼上去
        df = pd.merge(salary, metrics, on='key', how='left', suffixes=('', '_stat'))
        
        # 再次处理合并后产生的缺失值
        df['pos'] = df['pos'].fillna('G')
        num_cols = ['points', 'rebounds', 'minutes', 'attendance', 'clutch_proxy']
        for col in num_cols:
            df[col] = df[col].fillna(0)

        # 定义一个简单的函数来标准化位置名称 (比如 'Center/Forward' -> 'C')
        def map_pos(p):
            p = str(p).upper()
            if 'C' in p: return 'C'
            if 'F' in p: return 'F'
            return 'G'
        # 应用这个函数生成新的 'pos_mapped' 列
        df['pos_mapped'] = df['pos'].apply(map_pos)

        # 标记这些球员是通过交易获得的 (q=3 是你的作业要求里的特定渠道代码)
        df['channel'] = 'Trade (q=3)'
        
        return df

    def generate_real_bench(self, df):
        """
        这个函数用于手动添加一些低薪的“功能型”球员到池子里。
        在实际项目中，这些可能是新秀或者自由市场上的老将。
        """
        # 手动定义的球员列表
        real_bench_players = [
            # 姓名, 位置, 2025薪资, 类型(用于判断渠道), 备注
            ("Kate Martin", "G", 67000, "Rookie", "Fan Favorite"),
            # ... (中间省略了其他球员，逻辑一样) ...
            ("Queen Egbo", "C", 78831, "Veteran Min", "Los Angeles Sparks")
        ]

        dummies = []
        for name, pos, salary, p_type, note in real_bench_players:
            # 随机生成一个基础价值分数，模拟球探的不确定性
            base_vi = 0.20 + random.random() * 0.15 
            
            # 根据球员类型判断是选秀(Draft)还是自由球员(Free Agent)
            if any(k in p_type for k in ["Rookie", "Prospect", "Scale"]):
                acq_channel = "Draft (q=1)"
            else:
                acq_channel = "Free Agent (q=2)"
            
            # 构建一个字典，代表这名球员的所有属性
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
                'channel': acq_channel 
            }
            dummies.append(dummy)
        
        # 将列表转换为 DataFrame 并与原始数据合并
        dummy_df = pd.DataFrame(dummies)
        # 补齐列名，防止合并时出错
        for col in df.columns:
            if col not in dummy_df.columns:
                dummy_df[col] = 0
                
        return pd.concat([df, dummy_df], ignore_index=True)

    def calculate_values(self, df):
        """
        计算球员价值的核心数学逻辑。
        将各项数据归一化（0-1之间），然后加权求和。
        """
        df = df.copy() # 避免修改原始数据
        
        # 定义归一化函数：(当前值 - 最小值) / (最大值 - 最小值)
        def norm(col):
            min_v, max_v = col.min(), col.max()
            if max_v == min_v: return 0.5 # 防止除以0
            denom = max_v - min_v
            return (col - min_v) / (denom if denom > 0 else 1)

        # 如果还没有计算过 Vi_base，则开始计算
        if 'Vi_base' not in df.columns or df['Vi_base'].isna().any():
            # ViA: 竞技价值 (得分、篮板、助攻等)
            df['ViA'] = (0.25 * norm(df['points']) + 0.20 * norm(df['rebounds']) + 
                         0.20 * norm(df['assists']) + 0.15 * norm(df['plus_minus']) + 
                         0.20 * norm(df['clutch_proxy']))
            
            # ViB: 商业价值 (上座率、薪资水平作为身价代理)
            df['ViB'] = 0.6 * norm(df['attendance']) + 0.4 * norm(df['salary_2025'])
            
            # 计算受伤风险：上场时间越长、工龄越长，风险越高
            injury_prob = norm(df['minutes']) * 0.5 + norm(df['years_of_service']) * 0.5
            df['ViH'] = 1 - (injury_prob * 0.4) # ViH 是健康度，所以用 1 减去风险
            
            # ViP: 潜力值 (工龄越短，潜力通常越大)
            df['ViP'] = 1 - norm(df['years_of_service'])
            
            # 加权汇总得到基础价值 Vi_base
            df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                             CONFIG['W_H'] * df['ViH'] + CONFIG['W_P'] * df['ViP'])
        
        df['Vi_base'] = df['Vi_base'].fillna(0.1) # 兜底防止空值
        return df

# ==========================================
# 3. 策略求解器 (StrategicSolver)
# 这是整个程序的“大脑”，负责寻找最优解
# ==========================================
class StrategicSolver:
    def __init__(self, pool, current_team_name):
        # 预处理：把球队名字转小写
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        
        # 将总池子分为两部分：
        # 1. current_roster: 必须从中保留核心球员
        # 2. market_pool: 可以从中购买新球员
        self.current_roster = pool[pool['team_clean'] == target].copy()
        self.market_pool = pool[pool['team_clean'] != target].copy()
        
        self.n_current = len(self.current_roster)
        self.n_market = len(self.market_pool)

    def get_fitness(self, dna):
        """
        【目标函数 / 适应度函数】
        输入一个 DNA（0/1 数组），返回这个阵容的分数 (Z Score)。
        分数越高，阵容越好。
        """
        # dna[:n] 是当前队内的选择，dna[n:] 是市场的选择
        mask_keep = dna[:self.n_current] == 1
        mask_buy = dna[self.n_current:] == 1
        
        # 拼接出选中的所有球员
        roster = pd.concat([self.current_roster.iloc[mask_keep], self.market_pool.iloc[mask_buy]])
        count = len(roster)
        
        # 如果一个球员都没选，给个极低的负分
        if count == 0: return -1e12 
        
        # 计算总薪资 (Total Cost)
        TC = roster['salary_2025'].sum()
        
        # 计算位置结构分 (Fit Score)
        pos_counts = roster['pos_mapped'].value_counts()
        fit_score = 0
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: fit_score += 1 # 满足要求加分
            else: fit_score -= 0.5 * abs(c - (min_p+max_p)/2) # 不满足倒扣分
        
        # 将位置分归一化到 0-1
        ViF = max(0, min(1, fit_score * 0.3 + 0.5))
        
        avg_Vi = roster['Vi_base'].mean()
        
        # 计算总收益 (Total Revenue)，使用对数函数 log 模拟边际收益递减
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        
        # 计算总风险
        Risk = (roster['salary_2025'] * (1 - roster['ViH'])).sum()
        
        # === 核心：惩罚项 (Penalty) ===
        # 优化算法通常不擅长处理硬约束，所以我们把硬约束变成巨大的扣分
        penalty = 0
        
        # 1. 人数限制惩罚
        if count < CONFIG['ROSTER_MIN']:
            penalty += 1e7 + (CONFIG['ROSTER_MIN'] - count) * 1e6
        elif count > CONFIG['ROSTER_MAX']:
            penalty += 1e7 + (count - CONFIG['ROSTER_MAX']) * 1e6
            
        # 2. 工资帽惩罚 (一旦超帽，每超1块钱扣10分)
        if TC > CONFIG['SALARY_CAP']:
            penalty += (TC - CONFIG['SALARY_CAP']) * 10 
            
        # 3. 核心保留惩罚 (至少要保留2名老队员)
        n_kept = mask_keep.sum()
        if n_kept < 2: 
             penalty += (2 - n_kept) * 500000

        # === 最终 Z Score 公式 ===
        # 包含：收益风险比、平均能力、结构合理性，最后减去惩罚
        term1 = (TR - Risk) / (TC + 1e-5) * 50000
        Z = (CONFIG['MU_1'] * term1 + CONFIG['MU_2'] * avg_Vi * 5000 + CONFIG['MU_3'] * ViF * 5000) - penalty
        return Z

    def mutate(self, dna, rate=0.3):
        """
        【变异操作】
        随机改变 DNA 序列，防止算法陷入局部最优（Local Optima）。
        """
        child = deepcopy(dna)
        
        # 策略 A: 交换变异 (Swap) - 保持总人数不变
        # 找到被选中的人(1)和没被选中的人(0)
        idx_ones = np.where(child == 1)[0]
        idx_zeros = np.where(child == 0)[0]
        
        # 60% 的概率执行交换：把一个选中的变成不选，不选的变成选中
        if len(idx_ones) > 0 and len(idx_zeros) > 0 and random.random() < 0.6:
            i_one = np.random.choice(idx_ones)
            i_zero = np.random.choice(idx_zeros)
            child[i_one] = 0
            child[i_zero] = 1
            return child

        # 策略 B: 翻转变异 (Flip) - 改变总人数
        # 简单粗暴地把某个位置取反
        if random.random() < rate:
            total_len = len(dna)
            m_idx = random.randint(0, total_len-1)
            child[m_idx] = 1 - child[m_idx]
            
        return child

    def solve(self, generations=150, pop_size=60):
        """
        【主循环】
        运行优化算法：初始化种群 -> 迭代 -> 选择 -> 变异 -> 更新
        """
        total_len = self.n_current + self.n_market
        pop = [] # 种群 (Population)
        print(f">>> Solver Environment: Current {self.n_current}, Market (Classified) {self.n_market}")
        
        # 1. 初始化种群 (Generate Initial Population)
        for _ in range(pop_size):
            dna = np.zeros(total_len, dtype=int)
            
            # 随机保留 2-4 名老队员
            if self.n_current > 0:
                n_keep = random.randint(2, min(self.n_current, 4))
                idx_keep = np.random.choice(self.n_current, n_keep, replace=False)
                dna[idx_keep] = 1
            else:
                n_keep = 0
            
            # 随机填充直到达到 11-12 人
            current_count = n_keep
            target = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            needed = target - current_count
            
            if needed > 0 and self.n_market >= needed:
                idx_buy = np.random.choice(self.n_market, needed, replace=False)
                dna[self.n_current + idx_buy] = 1 # 市场球员索引要加上 n_current 偏移量
            pop.append(dna)
            
        best_dna, best_fitness = None, -float('inf')
        T = 200.0 # 初始温度 (用于模拟退火逻辑)
        
        # 2. 进化循环 (Evolution Loop)
        for gen in range(generations):
            # 计算当前种群所有个体的分数
            fitnesses = [self.get_fitness(d) for d in pop]
            curr_max = max(fitnesses)
            
            # 记录历史最佳
            if curr_max > best_fitness:
                best_fitness = curr_max
                best_dna = deepcopy(pop[fitnesses.index(curr_max)])
            
            # 每30代打印一次进度
            if gen % 30 == 0:
                print(f"Iter {gen:3d} | Best Z: {best_fitness:.2f}")
                
            new_pop = []
            while len(new_pop) < pop_size:
                # 锦标赛选择 (Tournament Selection): 随机选3个，挑最好的做父母
                candidates = random.sample(pop, 3)
                candidates_fitness = [self.get_fitness(c) for c in candidates]
                parent = candidates[np.argmax(candidates_fitness)]
                
                # 变异产生孩子
                child = self.mutate(parent, rate=0.2)
                f_child = self.get_fitness(child)
                f_parent = self.get_fitness(parent)
                
                # === 模拟退火 (Simulated Annealing) 接受准则 ===
                if f_child >= f_parent:
                    new_pop.append(child) # 如果孩子更好，直接要
                else:
                    # 如果孩子更差，按概率接受 (概率随温差和温度决定)
                    # 这有助于算法跳出局部陷阱
                    prob = math.exp((f_child - f_parent) / T)
                    if random.random() < prob:
                        new_pop.append(child)
                    else:
                        new_pop.append(parent) # 否则保留父母
            
            pop = new_pop
            T *= 0.92 # 降温：随着迭代进行，越来越不愿意接受差的解
            
        return best_dna, best_fitness

# ==========================================
# 4. 程序入口 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 定义文件名
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'
    f_salary = 'player_salaries_2025.xlsx'
    
    if os.path.exists(f_stats):
        # 1. 实例化数据处理器并加载数据
        proc = DataProcessor(f_stats, f_salary)
        full_df = proc.load_data()
        
        if full_df is not None:
            # 2. 计算球员价值
            valued_df = proc.calculate_values(full_df)
            
            # 3. 添加特定的低薪球员
            enhanced_df = proc.generate_real_bench(valued_df)
            
            # 4. 实例化求解器，指定当前队伍
            my_team = "Indiana Fever"
            solver = StrategicSolver(enhanced_df, my_team)
            
            if solver.n_current == 0:
                print(f"Warning: Team '{my_team}' not found.")
            else:
                # 5. 开始跑算法
                best_dna, best_z = solver.solve()
                
                # 6. 解析结果：将 0/1 数组还原为球员名单
                mask_keep = best_dna[:solver.n_current] == 1
                mask_buy = best_dna[solver.n_current:] == 1
                
                roster_keep = solver.current_roster.iloc[mask_keep]
                roster_buy = solver.market_pool.iloc[mask_buy]
                final_roster = pd.concat([roster_keep, roster_buy])
                
                # 7. 打印最终报表
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
        print("No Data! 请确保 Excel 文件在当前目录下。")