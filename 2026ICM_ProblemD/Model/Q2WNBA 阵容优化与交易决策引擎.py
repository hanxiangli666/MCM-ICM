"""
==============================================================================
文件名称: Q2.py (WNBA 阵容优化与交易决策引擎)
文件目的: 
    这是一个“智能球探与总经理助手”。它基于真实数据，自动计算每位球员的综合价值，
    并通过算法决定：
    1. 现有队员谁该留？谁该被交易（清洗）？
    2. 自由市场上谁性价比最高？谁最适合买入？
    3. 最终组成的 11-12 人大名单是否符合工资帽和位置需求？

核心技术点 (初学者必读):
1. 数据清洗与特征工程 (Data Engineering):
   - 加载混乱的 Excel 数据，统一列名，修补缺失值 (NaN)。
   - 手动补丁 (Manual Fixes): 代码里硬编码了一些关键球员的数据，
     这是处理现实世界数据缺失的常见手段（比如新秀可能没有历史数据）。
   - 归一化 (Normalization): 把得分、篮板等不同单位的数据都缩放到 0~1 之间，
     方便加权计算。

2. 多维价值评估模型 (Vi Score Model):
   - 不只看打球好不好，还看“性价比”和“商业价值”。
   - ViA (Ability): 竞技能力（得分、篮板等）。
   - ViB (Business): 商业价值（名气、票房号召力）。
   - ViH (Health): 健康风险（上场时间越长风险越高）。
   - ViP (Potential): 潜力（年轻球员分高）。
   - 最终得出一个综合分数 `Vi_base`，代表这个球员“值不值得买”。

3. 遗传算法 (Genetic Algorithm, GA):
   - 这是一个组合优化问题：从几百个球员里选 12 个，组合数是天文数字。
   - 我们把一个阵容看作一条 DNA（一串 0 和 1）。
     - 0 代表不选，1 代表选中。
   - 通过模拟自然进化（变异、优胜劣汰），快速找到近似最优解。

4. 罚函数法 (Penalty Function):
   - 怎么保证算出来的阵容不超工资帽？
   - 在计算分数时，如果超支了，就扣除巨额分数。这样算法为了得分高，
     就会自动学会“省钱”。
==============================================================================
"""

import pandas as pd     # 数据处理库
import numpy as np      # 数学运算库
import random           # 随机数库
import math             # 数学函数库
import os               # 文件路径处理
from copy import deepcopy # 深度复制对象，防止修改副本影响原件
import warnings         # 警告控制

# 忽略烦人的警告信息
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数 (CONFIG)
# ==========================================
# 这里的参数就像是控制台的旋钮，调整它们可以改变算法的偏好
CONFIG = {
    # 权重参数：决定了我们在乎什么。这里 W_A=0.3 表示我们最看重竞技能力。
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    
    # 目标函数权重：MU_1(财务), MU_2(竞技), MU_3(适配度)
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    
    # 硬约束条件
    'SALARY_CAP': 1500000,   # 2025 WNBA 软工资帽大约在 150万美元
    'ROSTER_MIN': 11,        # 阵容最少 11 人
    'ROSTER_MAX': 12,        # 阵容最多 12 人
    
    # 位置约束：后卫(G)4-6个，前锋(F)4-6个，中锋(C)2-3个
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    
    # 财务与风险系数 (用于计算对数收益和风险扣除)
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
}

# ==========================================
# 2. 数据处理类 (DataProcessor)
# ==========================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file   # 比赛数据文件路径
        self.salary_file = salary_file # 薪资数据文件路径

    # 辅助函数：把名字变成小写并去掉标点，防止 "A.J." 和 "aj" 匹配不上
    def normalize_name(self, series):
        """标准化名字以提高匹配率"""
        return series.astype(str).str.lower().str.strip().str.replace('.', '', regex=False)

    # 核心函数：加载并清洗数据
    def load_and_process(self, target_team_name="Indiana Fever"):
        print(f">>> [DataEngine] 正在加载数据，目标主队: {target_team_name}...")
        
        # --- 1. 读取原始文件 ---
        try:
            # 自动判断是 .xlsx 还是 .csv
            read_func = pd.read_excel if self.stats_file.endswith('.xlsx') else pd.read_csv
            stats_raw = read_func(self.stats_file)
            
            read_func = pd.read_excel if self.salary_file.endswith('.xlsx') else pd.read_csv
            salary_raw = read_func(self.salary_file)
        except Exception as e:
            print(f"!!! 文件读取失败: {e}")
            return None

        # 统一把列名变成小写，方便后续引用 (比如 'Points' -> 'points')
        stats_raw.columns = [c.lower().strip() for c in stats_raw.columns]
        salary_raw.columns = [c.lower().strip() for c in salary_raw.columns]
        
        # --- 2. 提取统计数据 ---
        # 如果数据里有多个赛季，只取最新的那个
        if 'season' in stats_raw.columns:
            target_season = stats_raw['season'].max()
            stats = stats_raw[stats_raw['season'] == target_season].copy()
        else:
            stats = stats_raw.copy()

        # 确保关键的数据列都是数字格式，把非数字的变成 0
        stat_cols = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy']
        for c in stat_cols:
            if c not in stats.columns: stats[c] = 0.0 # 如果列不存在，补全为0
            stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)

        # 聚合球员数据 (防止同一个球员有多行数据)
        metrics = stats.groupby('player').agg({
            'points': 'mean', 'rebounds': 'mean', 'assists': 'mean', 
            'plus_minus': 'mean', 'minutes': 'mean', 'attendance': 'mean',
            'clutch_proxy': 'mean', 'team': 'last' # 球队取最后一次出现的
        }).reset_index()

        # --- 3. 提取薪资数据 ---
        salary = salary_raw.copy()
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce')
        # 如果工龄缺失，默认为 1 年新秀
        salary['years_of_service'] = pd.to_numeric(salary.get('years_of_service', 0), errors='coerce').fillna(1)
        
        # --- 4. 合并数据 ---
        # 创建 'key' 列用于匹配 (标准化后的名字)
        metrics['key'] = self.normalize_name(metrics['player'])
        salary['key'] = self.normalize_name(salary['player'])
        
        # 类似 Excel VLOOKUP，把两张表拼在一起
        df = pd.merge(metrics, salary[['key', 'salary_2025', 'years_of_service', 'position', 'team']], 
                      on='key', how='outer', suffixes=('', '_sal'))
        
        # 合并后可能会有空值，互相填充一下
        df['player'] = df['player'].fillna(salary['player'])
        df['team'] = df['team'].fillna(df['team_sal']).fillna('Free Agent')
        
        # --- 5. 手动补丁 (Manual Fixes) ---
        # 真实数据往往不完美（比如新秀没数据，或者球星刚转会），这里手动“造”一些数据
        # 这是一个非常实用的工程技巧
        manual_fixes = {
            'aliyah boston': {'salary': 99000, 'pos': 'C', 'team': target_team_name, 'pts': 14.5, 'reb': 8.4, 'min': 30},
            'caitlin clark': {'salary': 78066, 'pos': 'G', 'team': target_team_name, 'pts': 19.2, 'reb': 5.7, 'ast': 8.2, 'min': 32, 'att': 17000},
            # ... (省略其他球员配置，逻辑同上)
            'makayla timpson': {'salary': 65000, 'pos': 'F', 'team': 'Free Agent'},
        }

        for name, data in manual_fixes.items():
            key = self.normalize_name(pd.Series([name]))[0]
            mask = df['key'] == key
            
            if mask.any():
                # 如果这人已经在表里了，就更新她的数据
                idx = df[mask].index
                if 'salary' in data: df.loc[idx, 'salary_2025'] = data['salary']
                if 'pos' in data: df.loc[idx, 'position'] = data['pos']
                # ... 更新其他字段
            else:
                # 如果这人不在表里，就新建一行插入进去
                new_row = {
                    'player': name.title(), 'key': key,
                    'team': data.get('team', target_team_name),
                    'salary_2025': data.get('salary', 76297), # 默认底薪
                    'position': data.get('pos', 'G'),
                    # ... 其他默认值
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # --- 6. 全局缺失值填充 ---
        # 再次兜底，确保没有任何 NaN (空值) 会导致计算报错
        fill_zeros = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy', 'years_of_service']
        for c in fill_zeros:
            if c not in df.columns: df[c] = 0.0
            df[c] = df[c].fillna(0.0)
        df['salary_2025'] = df['salary_2025'].fillna(76297) # 填补平均薪资
        if 'position' not in df.columns: df['position'] = 'G'
        df['position'] = df['position'].fillna('G')
        # 映射位置：把详细位置 (e.g., PG, SG) 统一映射为 G, F, C 三大类
        df['pos_mapped'] = df['position'].apply(self._map_pos)

        # --- 7. 计算 Vi 综合评分 ---
        df = self._calculate_scores(df)

        # 标记本队球员和自由球员
        # 清洗主队名字，确保能识别出 "Indiana Fever"
        target_clean = self.normalize_name(pd.Series([target_team_name]))[0]
        df['team_clean'] = self.normalize_name(df['team'])
        
        # 模糊匹配：只要包含 'fever' 或 'ind' 都算本队
        if 'fever' in target_clean:
            mask_home = df['team_clean'].str.contains('fever') | df['team_clean'].str.contains('ind')
            df.loc[mask_home, 'team_clean'] = target_clean
            df.loc[mask_home, 'team'] = target_team_name

        print(f"    - 数据加载完成，全联盟共 {len(df)} 名球员。")
        print(f"    - 目标球队 ({target_team_name}) 现有 {len(df[df['team_clean'] == target_clean])} 人。")
        return df

    # 辅助函数：位置映射
    def _map_pos(self, p):
        p = str(p).upper()
        if 'C' in p: return 'C' # 包含 C 的都是中锋
        if 'F' in p: return 'F' # 包含 F 的都是前锋
        return 'G'              # 其他的算后卫

    # 核心函数：计算评分系统 (Vi Model)
    def _calculate_scores(self, df):
        """计算综合评分 (Vi)，此时 df 中不应有 NaN"""
        # 归一化函数：把一列数据缩放到 0~1 之间
        def norm(col):
            min_v, max_v = col.min(), col.max()
            if max_v == min_v: return 0.5 # 如果所有人都一样，给个中间分
            return (col - min_v) / (max_v - min_v)

        # 1. 竞技分 (ViA): 得分、篮板、助攻、正负值、关键球
        df['ViA'] = (0.3 * norm(df['points']) + 0.2 * norm(df['rebounds']) + 
                     0.2 * norm(df['assists']) + 0.15 * norm(df['plus_minus']) + 
                     0.15 * norm(df['clutch_proxy']))
        
        # 2. 商业分 (ViB): 薪资(代表身价)、上座率
        df['ViB'] = 0.5 * norm(df['salary_2025']) + 0.5 * norm(df['attendance'])
        # 超级巨星加成 (如果工资超20万或叫 Clark，商业分乘 1.5)
        mask_star = (df['salary_2025'] > 200000) | (df['player'].str.contains('Clark', na=False, case=False))
        df.loc[mask_star, 'ViB'] *= 1.5
        df['ViB'] = df['ViB'].clip(0, 1) # 限制在 0-1 之间

        # 3. 健康分 (ViH): 上场时间越多，受伤风险越大，分数越低
        injury_risk = 0.5 * norm(df['minutes'])
        df['ViH'] = 1 - (injury_risk * 0.3) 

        # 4. 潜力分 (ViP): 越年轻(工龄越短)，潜力分越高
        df['ViP'] = 1 - norm(df['years_of_service'])

        # 5. 综合总分 (Vi_base): 加权求和
        df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                         CONFIG['W_H'] * df['ViH'] + CONFIG['W_P'] * df['ViP'])
        
        # 兜底：如果算出来是 NaN，填补平均值
        if df['Vi_base'].isna().any():
            print("    [Warning] Vi_base calculated NaN values, filling with mean.")
            df['Vi_base'] = df['Vi_base'].fillna(df['Vi_base'].mean())
            
        return df

# ==========================================
# 3. 策略求解器类 (StrategicSolver) - 遗传算法核心
# ==========================================
class StrategicSolver:
    def __init__(self, pool, current_team_name):
        # 同样先标准化名字
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        
        # 区分“本队”和“市场”
        if 'fever' in target:
            mask_home = pool['team_clean'].str.contains('fever') | pool['team_clean'].str.contains('ind')
        else:
            mask_home = pool['team_clean'] == target

        self.current_roster = pool[mask_home].copy().reset_index(drop=True) # 现有阵容
        # 自由市场：为了加快计算，只取市场上最好的前 60 人
        self.market_pool = pool[~mask_home].sort_values('Vi_base', ascending=False).head(60).reset_index(drop=True)
        
        self.n_current = len(self.current_roster) # 本队人数
        self.n_market = len(self.market_pool)     # 市场人数
        
        print(f">>> [SolverInit] 现有阵容: {self.n_current} 人 | 优质自由市场: {self.n_market} 人")

    # 适应度函数：评价一个 DNA (阵容方案) 到底好不好
    def get_fitness(self, dna):
        """计算适应度函数 (Z-Score)"""
        # DNA前半段控制本队留谁，后半段控制买谁
        mask_keep = dna[:self.n_current] == 1
        mask_buy = dna[self.n_current:] == 1
        
        # 拼出这个方案对应的完整阵容
        roster = pd.concat([self.current_roster.iloc[mask_keep], self.market_pool.iloc[mask_buy]])
        count = len(roster)
        
        if count == 0: return -1e9 # 没人？直接负无穷分
        
        TC = roster['salary_2025'].sum() # 总成本 (Total Cost)
        
        # 1. 计算位置适配度 (ViF)
        # 检查 G, F, C 的数量是否在 CONFIG 规定的范围内
        pos_counts = roster['pos_mapped'].value_counts()
        fit_score = 0
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: fit_score += 1 # 达标加分
            else: fit_score -= 0.5 * abs(c - (min_p+max_p)/2) # 不达标扣分
        ViF = max(0, min(1, fit_score * 0.3 + 0.5)) # 归一化

        # 2. 计算收益与风险
        avg_Vi = roster['Vi_base'].mean() # 平均能力值
        # 总收益 = 竞技收益(log) + 商业收益(linear)
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        # 风险 = 薪资 * (1 - 健康分) -> 工资越高且越脆的人，风险越大
        Risk = (roster['salary_2025'] * (1 - roster['ViH'])).sum() * CONFIG['RISK_FACTOR']

        # 3. 计算罚分 (硬约束)
        penalty = 0
        
        # 约束A: 人数必须在 11-12 之间
        if count < CONFIG['ROSTER_MIN']: penalty += (CONFIG['ROSTER_MIN'] - count) * 50
        if count > CONFIG['ROSTER_MAX']: penalty += (count - CONFIG['ROSTER_MAX']) * 50
        
        # 约束B: 工资帽 (Salary Cap) - 这是最重要的约束！
        if TC > CONFIG['SALARY_CAP']: 
            penalty += (TC - CONFIG['SALARY_CAP']) / 1000 # 每超 $1000 扣 1 分，惩罚很重
        
        # 约束C: 阵容稳定性，不能把本队人都卖光了，至少留 4 个
        n_kept = mask_keep.sum()
        if n_kept < 4: penalty += (4 - n_kept) * 20

        # 4. 汇总得分 Z-Score
        term_financial = (TR - Risk) / 10000 
        term_performance = avg_Vi * 100      
        
        Z = (CONFIG['MU_1'] * term_financial + 
             CONFIG['MU_2'] * term_performance + 
             CONFIG['MU_3'] * ViF * 50) - penalty
        return Z

    # 变异函数：随机改变 DNA，防止算法陷入死胡同
    def mutate(self, dna, rate=0.2):
        child = deepcopy(dna)
        # 变异方式 1: 翻转 (Flip) - 随机把选变成不选，不选变成选
        if random.random() < rate:
            idx = random.randint(0, len(dna)-1)
            child[idx] = 1 - child[idx]
            
        # 变异方式 2: 交换 (Swap) - 保持总人数不变，踢一个换一个
        if random.random() < 0.3: # 30% 概率发生
            ones = np.where(child == 1)[0] # 找到所有被选中的
            zeros = np.where(child == 0)[0] # 找到没被选中的
            if len(ones) > 0 and len(zeros) > 0:
                idx_1 = np.random.choice(ones)
                idx_0 = np.random.choice(zeros)
                child[idx_1] = 0 # 踢掉
                child[idx_0] = 1 # 加上
                
        return child

    # 遗传算法主循环
    def solve(self, generations=150, pop_size=100):
        total_len = self.n_current + self.n_market
        pop = []
        
        # --- 初始化种群 (智能初始化) ---
        # 相比完全随机，我们给一个“还不错”的初始状态，加快收敛
        for _ in range(pop_size):
            dna = np.zeros(total_len, dtype=int)
            
            # 随机决定想要多少人 (11或12)
            target_size = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            
            # 随机决定留多少个老人 (4个 ~ 全部)
            n_keep = random.randint(4, min(self.n_current, target_size))
            n_buy = target_size - n_keep
            
            # 随机选中留队的人
            if self.n_current > 0:
                idx_keep = np.random.choice(self.n_current, n_keep, replace=False)
                dna[idx_keep] = 1
            
            # 随机选中买入的人 (倾向于买 Vi 高的)
            if self.n_market > 0 and n_buy > 0:
                # 轮盘赌选择：Vi 越高的，被初始选中的概率越大
                weights = np.linspace(1.0, 0.1, self.n_market) # 权重递减
                weights = weights / weights.sum()
                idx_buy = np.random.choice(self.n_market, min(self.n_market, n_buy), replace=False, p=weights)
                dna[self.n_current + idx_buy] = 1
                
            pop.append(dna)
            
        best_dna, best_fitness = None, -float('inf')
        
        # --- 进化开始 ---
        for gen in range(generations):
            # 1. 评分
            fitnesses = np.array([self.get_fitness(d) for d in pop])
            
            # 记录历史最佳
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_dna = deepcopy(pop[max_idx])
                
            if gen % 20 == 0:
                print(f"    Iter {gen:3d} | Best Z: {best_fitness:.4f}")
            
            # 2. 生成下一代
            new_pop = []
            # 精英策略 (Elitism): 最好的那个直接晋级，不许变异
            new_pop.append(best_dna)
            
            while len(new_pop) < pop_size:
                # 锦标赛选择 (Tournament Selection): 随机挑3个，选最强的当父母
                indices = np.random.randint(0, pop_size, 3)
                candidates = [pop[i] for i in indices]
                cand_fits = [fitnesses[i] for i in indices]
                parent = candidates[np.argmax(cand_fits)]
                
                # 繁殖并变异
                child = self.mutate(parent, rate=0.15)
                new_pop.append(child)
            
            pop = new_pop
            
        return best_dna, best_fitness

# ==========================================
# 4. 主程序入口 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 文件名配置
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'
    f_salary = 'player_salaries_2025.xlsx'
    
    # 检查文件是否存在
    if os.path.exists(f_stats):
        # 1. 加载数据
        proc = DataProcessor(f_stats, f_salary)
        full_df = proc.load_and_process("Indiana Fever")
        
        if full_df is not None:
            # 2. 初始化求解器
            solver = StrategicSolver(full_df, "Indiana Fever")
            
            # --- 打印原始阵容 ---
            print("\n" + "="*80)
            print("【初始状态】 Indiana Fever 2025 赛季阵容")
            print("="*80)
            original_roster = solver.current_roster
            cols = ['player', 'pos_mapped', 'salary_2025', 'Vi_base']
            print(original_roster[cols].sort_values('salary_2025', ascending=False).to_string(index=False, formatters={'salary_2025': '${:,.0f}'.format, 'Vi_base': '{:.3f}'.format}))
            print(f"\n初始总薪资: ${original_roster['salary_2025'].sum():,.0f}")
            print(f"初始人数: {len(original_roster)}")
            
            # 3. 运行遗传算法求解
            if solver.n_current == 0:
                print("Error: 现有阵容为空，请检查手动补丁或数据源！")
            else:
                best_dna, best_z = solver.solve(generations=120, pop_size=80)
                
                # 4. 解码最优结果
                mask_keep = best_dna[:solver.n_current] == 1 # 前半段是留下的
                mask_buy = best_dna[solver.n_current:] == 1  # 后半段是买入的
                
                roster_keep = solver.current_roster.iloc[mask_keep]
                roster_trade_out = solver.current_roster.iloc[~mask_keep] # 取反就是被交易走的
                roster_buy = solver.market_pool.iloc[mask_buy]
                final_roster = pd.concat([roster_keep, roster_buy])
                
                # --- 格式化打印最终报告 ---
                print("\n" + "="*80)
                print(f"PRO-INSIGHT 2026 阵容优化结果 (Task 3: Trade & Roster)")
                print("="*80)
                print(f"Optimization Score (Z): {best_z:.4f}")
                print(f"Roster Size:     {len(final_roster)} (Target: 11-12)")
                print(f"Total Cap Used:  ${final_roster['salary_2025'].sum():,.0f} / ${CONFIG['SALARY_CAP']:,.0f}")
                print(f"Cap Space Left:  ${CONFIG['SALARY_CAP'] - final_roster['salary_2025'].sum():,.0f}")
                print(f"Avg Vi Score:    {final_roster['Vi_base'].mean():.4f}")
                print("-" * 80)
                
                # 定义打印格式
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