"""
==============================================================================
文件名称: Q1(1).py (WNBA 球队阵容优化器 Pro Max 版)
文件目的: 
    在工资帽（Salary Cap）和阵容人数限制下，通过算法自动组建一支
    预期利润最高、风险最低的 WNBA 球队阵容。

核心技术点 (初学者必读):
1. 数据清洗 (Pandas): 
   - 就像在 Python 里操作 Excel。用到 read_csv/read_excel 读取数据，
     merge 进行数据拼接 (类似 VLOOKUP)，groupby 进行数据透视。
   
2. 蒙特卡洛模拟 (Monte Carlo Simulation):
   - 一种预测未来的方法。因为球员会受伤、状态会波动，我们不能只算一次。
   - 代码让计算机模拟几千个赛季，引入随机因素（受伤概率、手感波动），
     算出“平均利润”和“最坏情况下的利润(CVaR)”。

3. 遗传算法 (Genetic Algorithm):
   - 一种模仿生物进化的智能算法。
   - 一开始随机生成一堆“球队阵容”（种群）。
   - 优胜劣汰：利润高的阵容留下，利润低的淘汰。
   - 变异：留下的阵容随机换几个人，产生下一代。
   - 重复几十代，最后剩下的就是最优解。

4. 启发式修复 (Heuristic Repair):
   - 随机生成的阵容往往不合规（比如超工资帽了）。
   - 代码里写了一套逻辑：如果超支了，就踢掉性价比最低的人，换个便宜的。
==============================================================================
"""

import pandas as pd     # 数据分析的神器，用于处理表格数据
import numpy as np      # 数学计算库，用于生成随机数、矩阵运算
import random           # Python自带的随机库，用于简单的随机选择
import warnings         # 用于控制警告信息的显示
import os               # 用于处理文件路径
from copy import deepcopy # 用于深度复制（复制一份完全独立的数据，修改不影响原件）

# 忽略运行时的警告信息（比如版本更新提示），让输出界面更干净 更美观
warnings.filterwarnings('ignore')

# 定义一个类，把所有功能封装在一起，方便管理
class WNBA_Optimizer_Pro_Max:
    # 1. 初始化函数：当类被创建时，最先执行这里
    def __init__(self, data_path="."):
        self.data_path = data_path      # 数据文件的存放文件夹，默认为当前目录
        self.salary_cap = 1507100       # 设定2025年的工资帽（总预算上限）
        self.roster_min = 11            # 球队最少人数
        self.roster_max = 12            # 球队最多人数
        
        # 定义“捆绑交易” (Package Deals)：这是业务逻辑
        # 意思是一旦选了 A'ja Wilson，就必须选 Chelsea Gray，模拟球星抱团
        self.package_deals = {
            "A'ja Wilson": "Chelsea Gray",
            "Breanna Stewart": "Sabrina Ionescu",
            "Napheesa Collier": "Kayla McBride" 
        }
        
        # 初始化时直接调用数据加载函数
        self.load_and_process_data()

    # 辅助函数：拼接文件路径，防止不同操作系统路径斜杠不同导致报错
    def get_path(self, filename):
        return os.path.join(self.data_path, filename)

    # 2. 数据加载与处理核心函数
    def load_and_process_data(self):
        print(">>> [Pro-Insight 2.0] 初始化增强版数据引擎 (含手动补丁)...")
        
        try:
            # --- 第一步：加载薪资数据 ---
            # 尝试不同的文件名，增加代码的健壮性
            salaries_file = "player_salaries_2025.xlsx - Sheet1.csv"
            if not os.path.exists(self.get_path(salaries_file)):
                 salaries_file = "player_salaries_2025.xlsx"
            
            # 尝试用 csv 读取，如果报错就用 excel 读取
            try:
                df_salaries = pd.read_csv(self.get_path(salaries_file))
            except:
                df_salaries = pd.read_excel(self.get_path(salaries_file))

            # 数据清洗：处理薪资列
            # 如果薪资是文本格式（比如 "$100,000"），需要把 '$' 和 ',' 去掉
            if df_salaries['salary_2025'].dtype == 'O': # 'O' 代表 Object (文本)
                # 注意：原代码这里有个拼写错误 aacstype 应为 astype，但逻辑是转字符串并替换符号
                df_salaries['salary_2025'] = df_salaries['salary_2025'].astype(str).str.replace(r'[$,]', '', regex=True)
            
            # 转化为数字类型，如果不合法就变成 NaN (空值)，然后用 76297 (平均底薪) 填充空值
            df_salaries['salary_2025'] = pd.to_numeric(df_salaries['salary_2025'], errors='coerce').fillna(76297)
            
            # --- 第二步：加载比赛数据 ---
            stats_file = "30_MASTER_PLAYER_GAME.xlsx - Sheet1.csv"
            if not os.path.exists(self.get_path(stats_file)):
                 stats_file = "30_MASTER_PLAYER_GAME.xlsx"
            try:
                df_stats = pd.read_csv(self.get_path(stats_file))
            except:
                df_stats = pd.read_excel(self.get_path(stats_file))
            
            # 确保关键的数据列都是数字格式
            numeric_cols = ['minutes', 'points', 'rebounds', 'assists', 'turnovers', 'plus_minus']
            for col in numeric_cols:
                if col in df_stats.columns:
                    df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')
            
            # 只筛选 2024 赛季的数据作为参考
            df_stats = df_stats[df_stats['season'] == 2024]

            # 数据透视 (GroupBy)：把每个球员多场比赛的数据加总或取平均
            player_stats = df_stats.groupby('player').agg({
                'minutes': 'sum', 'points': 'sum', 'rebounds': 'sum',
                'assists': 'sum', 'turnovers': 'sum', 'plus_minus': 'mean'
            }).reset_index()
            
            # 尝试获取球员位置信息 (G/F/C)
            if 'pos' in df_stats.columns:
                pos_df = df_stats.groupby('player')['pos'].first().reset_index()
                player_stats = pd.merge(player_stats, pos_df, on='player')
            
            # 过滤掉上场时间太少的边缘球员（总时间小于50分钟）
            player_stats = player_stats[player_stats['minutes'] > 50]
            
            # --- 计算高阶数据 PER (效率值) ---
            # 简单公式：(得分+篮板+助攻-失误) / 上场时间
            player_stats['per_raw'] = (player_stats['points'] + player_stats['rebounds'] + 
                                       player_stats['assists'] - player_stats['turnovers']) / player_stats['minutes']
            
            # 归一化 (Normalization)：把数据缩放到 0 到 1 之间，方便后续计算
            p_min, p_max = player_stats['per_raw'].min(), player_stats['per_raw'].max()
            player_stats['per_norm'] = (player_stats['per_raw'] - p_min) / (p_max - p_min)

            # --- 第三步：合并薪资和比赛数据 ---
            # 类似于 Excel 的 VLOOKUP，以 'player' 为准对齐
            merged = pd.merge(player_stats, df_salaries, on='player', how='left')
            
            # --- 手动补丁 (Manual Corrections) ---
            # 真实数据往往有缺失或错误，这里手动修正一些关键球员的数据
            manual_corrections = {
                "Diana Taurasi": {"salary": 134500, "years": 20},
                "Nneka Ogwumike": {"salary": 169500, "years": 12},
                "Aliyah Boston": {"salary": 78000, "years": 2, "young": 1}, # young=1 标记为潜力新星
                # ... (省略中间名单)
                "Angel Reese": {"salary": 73439, "years": 1, "young": 1}
            }

            # 循环应用补丁
            for name, data in manual_corrections.items():
                mask = merged['player'] == name # 找到该球员的行
                if mask.any():
                    # 修正薪资
                    merged.loc[mask, 'salary_2025'] = data['salary']
                    # 修正工龄
                    if 'years_of_service' in merged.columns:
                        merged.loc[mask, 'years_of_service'] = data.get('years', 5)
                    # 修正是否为新秀
                    if 'young' in data:
                        merged.loc[mask, 'draft_year'] = 2024 

            # 再次填充可能遗漏的缺失值
            merged['salary_2025'] = merged['salary_2025'].fillna(76297)
            
            # 处理位置信息缺失，默认设为 'G' (后卫)
            if 'position' in merged.columns:
                merged['position'] = merged['position'].fillna(merged.get('pos', 'G'))
            else:
                merged['position'] = merged.get('pos', 'G')

            # --- 特征工程：计算伤病概率和潜力 ---
            merged['years_of_service'] = pd.to_numeric(merged.get('years_of_service', 3), errors='coerce').fillna(3)
            # 逻辑：工龄越长，受伤概率越高（基础5% + 每年1.5%）
            merged['injury_prob'] = 0.05 + (merged['years_of_service'] * 0.015)
            # 限制受伤概率在 5% 到 25% 之间
            merged['injury_prob'] = merged['injury_prob'].clip(0.05, 0.25)

            # 潜力标记：2023年以后选秀的算年轻天赋
            merged['draft_year'] = pd.to_numeric(merged.get('draft_year', 2018), errors='coerce').fillna(2018)
            merged['is_young_talent'] = (merged['draft_year'] >= 2023).astype(int)
            
            # 名气值 (Fame)：假设薪资越高，名气越大
            merged['fame'] = (merged['salary_2025'] / merged['salary_2025'].max())

            # --- 随机模拟特性 ---
            np.random.seed(42) # 固定随机种子，保证每次运行结果一致
            # 模拟海外打球：如果薪资低且随机数达标，假定她去海外打球了（会增加疲劳）
            merged['plays_overseas'] = merged['salary_2025'].apply(lambda s: 1 if (s < 150000 and random.random() < 0.6) else 0)
            # 模拟化学反应：随机生成一个 -0.1 到 0.15 的系数
            merged['chemistry'] = np.random.uniform(-0.1, 0.15, len(merged))

            # 保存最终清洗好的球员池
            self.player_pool = merged[['player', 'position', 'salary_2025', 'per_norm', 
                                     'fame', 'chemistry', 'injury_prob', 'plays_overseas', 'is_young_talent']]
            
            print(f"数据加载完毕 (含修正补丁): {len(self.player_pool)} 球员。")

        except Exception as e:
            # 如果上面读文件出错（比如没文件），生成假的模拟数据以防程序崩溃
            self.player_pool = self._generate_advanced_mock_data()

    # 生成模拟数据的函数（仅在读取失败时使用，用于测试代码逻辑）
    def _generate_advanced_mock_data(self):
        count = 150
        names = [f"Player_{i}" for i in range(count)]
        # 手动设置几个真名
        names[0] = "A'ja Wilson"
        names[1] = "Chelsea Gray"
        names[2] = "Breanna Stewart"
        
        # 随机生成薪资分布
        salaries = np.concatenate([np.random.uniform(200000, 250000, 10), np.random.uniform(76000, 190000, 140)])
        df = pd.DataFrame({
            'player': names,
            'position': np.random.choice(['G', 'F', 'C'], count),
            'salary_2025': salaries,
            'per_norm': np.clip((salaries/250000)*0.8 + np.random.normal(0,0.1,count), 0.1, 1.0),
            'fame': (salaries/250000)**2,
            'chemistry': np.random.uniform(-0.1, 0.15, count),
            'injury_prob': np.random.uniform(0.05, 0.20, count), 
            'plays_overseas': np.random.choice([0, 1], count, p=[0.4, 0.6]), 
            'is_young_talent': np.random.choice([0, 1], count, p=[0.8, 0.2]) 
        })
        return df

    # --- 3. 核心功能：蒙特卡洛风险评估 ---
    
    def simulate_season_performance(self, roster, n_sims=50):
        """
        这个函数是用来"算命"的。
        roster: 当前选中的阵容
        n_sims: 模拟多少个赛季（默认50次）
        """
        profits = []
        
        # 提取基础数据，转换成 numpy 数组运算更快
        base_per = roster['per_norm'].values
        salaries = roster['salary_2025'].sum()
        fame_sum = roster['fame'].sum()
        young_count = roster['is_young_talent'].sum()
        
        # 计算未来价值加成 (年轻球员有额外估值)
        future_bonus = young_count * 75000 
        
        # 开始循环模拟
        for _ in range(n_sims):
            # 随机事件 1: 伤病
            # 生成一串随机数，如果大于受伤概率，则是健康的(1.0)，否则受伤(0.0)
            health_status = (np.random.rand(len(roster)) > roster['injury_prob']).astype(float)
            # 如果受伤了，假设能力只剩 20% (0.2)，没受伤是 100% (1.0)
            health_status = np.where(health_status == 0, 0.2, 1.0)
            
            # 随机事件 2: 疲劳
            # 如果去海外打球了，能力打9折 (0.9)
            fatigue_factor = np.where(roster['plays_overseas'] == 1, 0.9, 1.0)
            
            # 随机事件 3: 状态起伏 (Performance Variance)
            # 产生一个正态分布波动 (比如今天状态 1.1倍，明天 0.9倍)
            perf_fluctuation = np.random.normal(1.0, 0.1, len(roster))
            
            # 综合计算本赛季实际表现能力值 = 基础能力 * 健康 * 疲劳 * 波动
            sim_per = base_per * health_status * fatigue_factor * perf_fluctuation
            team_per = np.sum(sim_per) # 全队总能力
            
            # 计算胜率 (使用 Logistic 函数，把能力值映射到 0~1 的胜率)
            win_rate = 1 / (1 + np.exp(-1.5 * (team_per - 5.0)))
            
            # 计算利润模型
            ticket_rev = 14000 * (1 + (win_rate - 0.5)) * 60 * 20 # 票务收入 (跟胜率挂钩)
            merch_rev = 500000 * (1 + np.log(1 + fame_sum))      # 周边收入 (跟名气挂钩)
            sponsor_rev = 2000000 * (1 + win_rate)               # 赞助收入
            
            total_rev = ticket_rev + merch_rev + sponsor_rev
            profit = total_rev - salaries - 3000000 # 利润 = 收入 - 工资 - 运营成本(3M)
            profits.append(profit)
            
        # 统计结果
        profits = np.array(profits)
        expected_profit = np.mean(profits) # 平均利润
        
        # 计算 CVaR (条件在险价值) - 这是一个高级金融风险指标
        # 意思是：在最倒霉的那 10% 的情况里，我们平均会亏多少/赚多少？
        cutoff = int(n_sims * 0.10) 
        if cutoff < 1: cutoff = 1
        worst_profits = np.sort(profits)[:cutoff] # 排序取最小的 10%
        cvar_profit = np.mean(worst_profits)
        
        return expected_profit, cvar_profit, future_bonus

    # --- 4. 目标函数：评价一个阵容好不好 ---
    def objective_function(self, individual):
        """
        individual: 一个具体的解决方案（包含球员名单索引）
        """
        roster_idx, _, _ = individual
        roster = self.player_pool.iloc[roster_idx]
        
        # 硬约束检查：如果不满足条件，直接给负无穷分 (-1e9)
        if len(roster) < self.roster_min or len(roster) > self.roster_max: return -1e9
        if roster['salary_2025'].sum() > self.salary_cap: return -1e9
        
        # 检查裙带关系 (Package Deals)
        roster_names = set(roster['player'].values)
        nepotism_penalty = 0
        for star, partner in self.package_deals.items():
            # 如果有A没B，巨额罚分
            if star in roster_names and partner not in roster_names:
                nepotism_penalty += 5000000 

        # 运行模拟，获取预期利润和风险
        exp_profit, cvar_profit, future_bonus = self.simulate_season_performance(roster, n_sims=20)
        
        # RAROC (风险调整后收益)：利润 - 风险惩罚
        # 意思是：如果两个阵容利润一样，我更喜欢风险小的那个
        risk_penalty = (exp_profit - cvar_profit) * 0.5 
        
        final_score = exp_profit - risk_penalty + future_bonus - nepotism_penalty
        return final_score

    # --- 5. 修复函数：把乱七八糟的阵容修好 ---
    def repair_roster(self, roster_idx):
        # 修复逻辑 A: 裙带关系
        current_names = self.player_pool.iloc[roster_idx]['player'].values
        for star, partner in self.package_deals.items():
            if star in current_names and partner not in current_names:
                # 强制把搭档加进来
                partner_rows = self.player_pool[self.player_pool['player'] == partner]
                if not partner_rows.empty:
                    p_idx = partner_rows.index[0]
                    if p_idx not in roster_idx:
                        roster_idx.append(p_idx)
        
        # 修复逻辑 B: 工资帽贪心算法
        current_roster = self.player_pool.iloc[roster_idx].copy()
        total_sal = current_roster['salary_2025'].sum()
        
        attempts = 0
        # 如果超工资了，循环踢人
        while total_sal > self.salary_cap and attempts < 20:
            # 计算“性价比”：能力 / 薪资，同时保护年轻人和明星
            score_metric = (current_roster['per_norm'] / (current_roster['salary_2025'] + 1)) * \
                           (1 + 0.5 * current_roster['is_young_talent']) * \
                           (1 + 0.5 * current_roster['fame'])
            
            # 找出分数最低的（最该被踢掉的）
            drop_idx = score_metric.idxmin()
            if drop_idx in roster_idx:
                roster_idx.remove(drop_idx)
            
            # 如果人太少了，补一个最便宜的球员凑数
            if len(roster_idx) < 11:
                cheap_players = self.player_pool[
                    (self.player_pool['salary_2025'] < 80000) & 
                    (~self.player_pool.index.isin(roster_idx))
                ]
                if not cheap_players.empty:
                    add_idx = cheap_players['per_norm'].idxmax() # 选便宜里能力最高的
                    roster_idx.append(add_idx)

            # 更新当前状态，准备下一轮检查
            current_roster = self.player_pool.iloc[roster_idx]
            total_sal = current_roster['salary_2025'].sum()
            attempts += 1
            
        return roster_idx[:12] # 确保不超过12人

    # --- 6. 遗传算法主程序 ---
    def run_genetic_algorithm(self, generations=50, population_size=40):
        print(f">>> 启动 Pro-Insight 2.0 优化 (Gens: {generations}, Pop: {population_size})...")
        
        # 初始化种群：随机生成 40 个阵容
        population = []
        for _ in range(population_size):
            r_idx = list(np.random.choice(self.player_pool.index, 12, replace=False))
            r_idx = self.repair_roster(r_idx) # 必须先修复，保证初始阵容合规
            population.append([r_idx, 0, 0]) 

        best_solution = None
        best_score = -np.inf
        
        # 开始进化循环
        for gen in range(generations):
            scores = []
            # 1. 评估：给每个阵容打分
            for indiv in population:
                s = self.objective_function(indiv)
                scores.append(s)
                # 记录历史最高分
                if s > best_score:
                    best_score = s
                    best_solution = deepcopy(indiv)
            
            # 2. 选择：保留前 20% 最好的阵容作为父母
            sorted_idx = np.argsort(scores)[::-1]
            new_pop = [population[i] for i in sorted_idx[:int(population_size*0.2)]]
            
            # 3. 繁殖：补满种群
            while len(new_pop) < population_size:
                # 随机选一个父母
                parent = population[random.choice(sorted_idx[:20])]
                child_roster = deepcopy(parent[0])
                
                # 4. 变异：40% 概率换人
                if random.random() < 0.4:
                    if len(child_roster) > 0:
                        child_roster.remove(random.choice(child_roster)) # 踢一个
                        avail = list(set(self.player_pool.index) - set(child_roster))
                        if avail: child_roster.append(random.choice(avail)) # 加一个
                
                # 再次修复子代，确保合规
                child_roster = self.repair_roster(child_roster)
                new_pop.append([child_roster, 0, 0])
            
            population = new_pop
            if gen % 10 == 0:
                print(f"Gen {gen}: Best RAROC Score = {best_score:,.0f}")

        return best_solution

    # 报告生成函数
    def report(self, solution):
        r_idx = solution[0]
        roster = self.player_pool.iloc[r_idx]
        
        print("\n" + "="*70)
        print("PRO-INSIGHT 2.0 最终决策报告 (风险调整版)")
        print("="*70)
        
        # 运行一次高精度模拟 (1000次) 来获取最终数据
        exp, cvar, future = self.simulate_season_performance(roster, n_sims=1000)
        
        print(f"【财务风险评估 (RAROC Analysis)】")
        print(f"  - 预期利润 (Expected Profit):   ${exp:,.2f}")
        print(f"  - 极端风险 (CVaR 10%):        ${cvar:,.2f} (最坏情况下的平均值)")
        print(f"  - 风险敞口 (Risk Exposure):     ${(exp - cvar):,.2f}")
        print(f"  - 潜力估值加成 (Future Bonus):  ${future:,.2f}")
        
        print(f"\n【阵容结构分析】")
        print(f"  - 总薪资: ${roster['salary_2025'].sum():,.2f}")
        # ... (省略部分打印代码)
        
        print(f"\n【最终名单】")
        # 打印排好序的名单
        cols = ['player', 'position', 'salary_2025', 'per_norm', 'injury_prob', 'is_young_talent']
        print(roster[cols].sort_values('salary_2025', ascending=False).to_string(index=False))

# 程序入口
if __name__ == "__main__":
    opt = WNBA_Optimizer_Pro_Max()
    # 运行遗传算法：进化 60 代，种群规模 500
    best_sol = opt.run_genetic_algorithm(generations=60, population_size=500)
    opt.report(best_sol)