import pandas as pd
import numpy as np
import random
import warnings
import os
from copy import deepcopy

# 忽略警告
warnings.filterwarnings('ignore')

class WNBA_Optimizer_Pro_Max:
    def __init__(self, data_path="."):
        self.data_path = data_path
        self.salary_cap = 1507100  # 2025 Salary Cap
        self.roster_min = 11
        self.roster_max = 12
        
        self.package_deals = {
            "A'ja Wilson": "Chelsea Gray",
            "Breanna Stewart": "Sabrina Ionescu",
            "Napheesa Collier": "Kayla McBride" 
        }
        
        self.load_and_process_data()

    def get_path(self, filename):
        return os.path.join(self.data_path, filename)

    def load_and_process_data(self):
        print(">>> [Pro-Insight 2.0] 初始化增强版数据引擎 (含手动补丁)...")
        
        try:
            # 1. 加载薪资数据
            salaries_file = "player_salaries_2025.xlsx - Sheet1.csv"
            if not os.path.exists(self.get_path(salaries_file)):
                 salaries_file = "player_salaries_2025.xlsx"
            
            try:
                df_salaries = pd.read_csv(self.get_path(salaries_file))
            except:
                df_salaries = pd.read_excel(self.get_path(salaries_file))

            # 清洗薪资
            if df_salaries['salary_2025'].dtype == 'O':
                df_salaries['salary_2025'] = df_salaries['salary_2025'].aacstype(str).str.replace(r'[$,]', '', regex=True)
            df_salaries['salary_2025'] = pd.to_numeric(df_salaries['salary_2025'], errors='coerce').fillna(76297)
            
            # 2. 加载比赛数据
            stats_file = "30_MASTER_PLAYER_GAME.xlsx - Sheet1.csv"
            if not os.path.exists(self.get_path(stats_file)):
                 stats_file = "30_MASTER_PLAYER_GAME.xlsx"
            try:
                df_stats = pd.read_csv(self.get_path(stats_file))
            except:
                df_stats = pd.read_excel(self.get_path(stats_file))
            
            # (数据清洗逻辑同原代码...)
            numeric_cols = ['minutes', 'points', 'rebounds', 'assists', 'turnovers', 'plus_minus']
            for col in numeric_cols:
                if col in df_stats.columns:
                    df_stats[col] = pd.to_numeric(df_stats[col], errors='coerce')
            df_stats = df_stats[df_stats['season'] == 2024]

            player_stats = df_stats.groupby('player').agg({
                'minutes': 'sum', 'points': 'sum', 'rebounds': 'sum',
                'assists': 'sum', 'turnovers': 'sum', 'plus_minus': 'mean'
            }).reset_index()
            
            if 'pos' in df_stats.columns:
                pos_df = df_stats.groupby('player')['pos'].first().reset_index()
                player_stats = pd.merge(player_stats, pos_df, on='player')
            
            player_stats = player_stats[player_stats['minutes'] > 50]
            
            # 计算 PER
            player_stats['per_raw'] = (player_stats['points'] + player_stats['rebounds'] + 
                                       player_stats['assists'] - player_stats['turnovers']) / player_stats['minutes']
            p_min, p_max = player_stats['per_raw'].min(), player_stats['per_raw'].max()
            player_stats['per_norm'] = (player_stats['per_raw'] - p_min) / (p_max - p_min)

            # 3. 合并数据
            merged = pd.merge(player_stats, df_salaries, on='player', how='left')
            
            manual_corrections = {
                "Diana Taurasi": {"salary": 134500, "years": 20},
                "Nneka Ogwumike": {"salary": 169500, "years": 12},
                "Aliyah Boston": {"salary": 78000, "years": 2, "young": 1},
                "Tina Charles": {"salary": 130000, "years": 13},
                "DeWanna Bonner": {"salary": 200000, "years": 15},
                "Jonquel Jones": {"salary": 195000, "years": 8},
                "Allisha Gray": {"salary": 185000, "years": 7},
                "Cheyenne Parker": {"salary": 175000, "years": 9},
                "Dearica Hamby": {"salary": 169000, "years": 9},
                "Shakira Austin": {"salary": 91000, "years": 3, "young": 1},
                "Ariel Atkins": {"salary": 200000, "years": 7},
                "Kamilla Cardoso": {"salary": 76535, "years": 1, "young": 1}, # 2024新秀
                "Rickea Jackson": {"salary": 76535, "years": 1, "young": 1},
                "Angel Reese": {"salary": 73439, "years": 1, "young": 1}
            }

            # 应用补丁
            for name, data in manual_corrections.items():
                mask = merged['player'] == name
                if mask.any():
                    # 只有当原始薪资是 NaN (即未匹配到) 时才覆盖，或者强制覆盖
                    merged.loc[mask, 'salary_2025'] = data['salary']
                    if 'years_of_service' in merged.columns:
                        merged.loc[mask, 'years_of_service'] = data.get('years', 5)
                    # 标记潜力新秀
                    if 'young' in data:
                        merged.loc[mask, 'draft_year'] = 2024 # 设为最近年份以触发 is_young_talent

            # 填充剩余缺失值
            merged['salary_2025'] = merged['salary_2025'].fillna(76297)
            
            # 处理其他字段
            if 'position' in merged.columns:
                merged['position'] = merged['position'].fillna(merged.get('pos', 'G'))
            else:
                merged['position'] = merged.get('pos', 'G')

            # 重新计算伤病和潜力
            merged['years_of_service'] = pd.to_numeric(merged.get('years_of_service', 3), errors='coerce').fillna(3)
            merged['injury_prob'] = 0.05 + (merged['years_of_service'] * 0.015)
            merged['injury_prob'] = merged['injury_prob'].clip(0.05, 0.25)

            # 潜力标记
            merged['draft_year'] = pd.to_numeric(merged.get('draft_year', 2018), errors='coerce').fillna(2018)
            merged['is_young_talent'] = (merged['draft_year'] >= 2023).astype(int)
            
            # Fame 计算
            merged['fame'] = (merged['salary_2025'] / merged['salary_2025'].max())

            # 海外疲劳模拟
            np.random.seed(42)
            merged['plays_overseas'] = merged['salary_2025'].apply(lambda s: 1 if (s < 150000 and random.random() < 0.6) else 0)
            merged['chemistry'] = np.random.uniform(-0.1, 0.15, len(merged))

            self.player_pool = merged[['player', 'position', 'salary_2025', 'per_norm', 
                                     'fame', 'chemistry', 'injury_prob', 'plays_overseas', 'is_young_talent']]
            
            print(f"数据加载完毕 (含修正补丁): {len(self.player_pool)} 球员。")

        except Exception as e:
            self.player_pool = self._generate_advanced_mock_data()

    def _generate_advanced_mock_data(self):
        count = 150
        names = [f"Player_{i}" for i in range(count)]
        names[0] = "A'ja Wilson"
        names[1] = "Chelsea Gray"
        names[2] = "Breanna Stewart"
        
        salaries = np.concatenate([np.random.uniform(200000, 250000, 10), np.random.uniform(76000, 190000, 140)])
        df = pd.DataFrame({
            'player': names,
            'position': np.random.choice(['G', 'F', 'C'], count),
            'salary_2025': salaries,
            'per_norm': np.clip((salaries/250000)*0.8 + np.random.normal(0,0.1,count), 0.1, 1.0),
            'fame': (salaries/250000)**2,
            'chemistry': np.random.uniform(-0.1, 0.15, count),
            'injury_prob': np.random.uniform(0.05, 0.20, count), # 5%-20% 受伤率
            'plays_overseas': np.random.choice([0, 1], count, p=[0.4, 0.6]), # 60% 海外打球
            'is_young_talent': np.random.choice([0, 1], count, p=[0.8, 0.2]) # 20% 潜力股
        })
        return df

    # --- 蒙特卡洛风险评估 ---
    
    def simulate_season_performance(self, roster, n_sims=50):
        """
        运行 N 次赛季模拟，引入随机性 (Stochasticity)
        返回: 
          - expected_profit: 期望利润
          - cvar_profit: 条件在险价值 (最差 10% 的平均利润)
          - future_bonus: 潜力带来的估值加成
        """
        profits = []
        
        # 基础数据提取
        base_per = roster['per_norm'].values
        salaries = roster['salary_2025'].sum()
        fame_sum = roster['fame'].sum()
        young_count = roster['is_young_talent'].sum()
        
        # 潜力加成 (Future Value Dynamics)
        # 每个年轻天赋额外增加 1.5M 的估值 (折合利润约为 1.5M * 0.05 = 75k)
        future_bonus = young_count * 75000 
        
        for _ in range(n_sims):
            # 1. 模拟伤病 (Injury Stochasticity)
            # 生成随机数，如果 < injury_prob，则该球员本赛季"报销"或贡献减半
            # 0 = 报销, 1 = 健康
            health_status = (np.random.rand(len(roster)) > roster['injury_prob']).astype(float)
            # 伤病不完全是0，假设出勤率为 20%
            health_status = np.where(health_status == 0, 0.2, 1.0)
            
            # 2. 模拟海外疲劳 (Overseas Fatigue)
            # 如果 plays_overseas = 1，赛季后半段 PER 下降 10%
            fatigue_factor = np.where(roster['plays_overseas'] == 1, 0.9, 1.0)
            
            # 3. 模拟状态起伏 (Performance Variance)
            # 正态分布波动 +/- 10%
            perf_fluctuation = np.random.normal(1.0, 0.1, len(roster))
            
            # 综合计算本赛季实际 PER
            sim_per = base_per * health_status * fatigue_factor * perf_fluctuation
            team_per = np.sum(sim_per)
            
            # 计算胜率 (Logistic Function)
            win_rate = 1 / (1 + np.exp(-1.5 * (team_per - 5.0)))
            
            # 计算单次模拟的利润 (简化版)
            # 收入 = 票务 + 周边 + 赞助
            ticket_rev = 14000 * (1 + (win_rate - 0.5)) * 60 * 20 # 假设均价60
            merch_rev = 500000 * (1 + np.log(1 + fame_sum))
            sponsor_rev = 2000000 * (1 + win_rate)
            
            total_rev = ticket_rev + merch_rev + sponsor_rev
            profit = total_rev - salaries - 3000000 # 3M 运营成本
            profits.append(profit)
            
        # 统计分析
        profits = np.array(profits)
        expected_profit = np.mean(profits)
        
        # CVaR (Conditional Value at Risk) 5%
        # 衡量最坏情况下的财务状况
        cutoff = int(n_sims * 0.10) # 取最差的 10%
        if cutoff < 1: cutoff = 1
        worst_profits = np.sort(profits)[:cutoff]
        cvar_profit = np.mean(worst_profits)
        
        return expected_profit, cvar_profit, future_bonus

    def objective_function(self, individual):
        """
        RAROC 目标函数
        Score = Expected_Profit - Lambda * Risk + Future_Bonus - Penalties
        """
        roster_idx, _, _ = individual
        roster = self.player_pool.iloc[roster_idx]
        
        # 1. 硬约束检查
        if len(roster) < self.roster_min or len(roster) > self.roster_max: return -1e9
        if roster['salary_2025'].sum() > self.salary_cap: return -1e9
        
        # 2. 裙带关系约束 (Nepotism Constraint)
        # 如果有球星A，必须有球星B
        roster_names = set(roster['player'].values)
        nepotism_penalty = 0
        for star, partner in self.package_deals.items():
            if star in roster_names and partner not in roster_names:
                # 惩罚：这会让球星不开心，极大降低名气效应，或直接罚分
                nepotism_penalty += 5000000 # 500万的罚分，基本等于不可行

        # 3. 运行蒙特卡洛模拟
        # 在训练阶段，为了速度，sims 设少一点 (e.g., 20)
        exp_profit, cvar_profit, future_bonus = self.simulate_season_performance(roster, n_sims=20)
        
        # 4. 计算 RAROC (Risk-Adjusted Return)
        # 我们不仅想要利润高，还想要风险低 (Expected - CVaR 越小越好)
        risk_penalty = (exp_profit - cvar_profit) * 0.5 # 风险厌恶系数 0.5
        
        final_score = exp_profit - risk_penalty + future_bonus - nepotism_penalty
        return final_score

    def repair_roster(self, roster_idx):
        # 1. 裙带关系 (Package Deal Enforcement)
        # 如果阵容里有 A'ja 但没 Chelsea，尝试把 Chelsea 加进来
        current_names = self.player_pool.iloc[roster_idx]['player'].values
        
        for star, partner in self.package_deals.items():
            if star in current_names and partner not in current_names:
                # 找到 partner 的 index
                partner_rows = self.player_pool[self.player_pool['player'] == partner]
                if not partner_rows.empty:
                    p_idx = partner_rows.index[0]
                    if p_idx not in roster_idx:
                        roster_idx.append(p_idx)
        
        # 2. 工资帽贪心修复
        current_roster = self.player_pool.iloc[roster_idx].copy()
        total_sal = current_roster['salary_2025'].sum()
        
        attempts = 0
        while total_sal > self.salary_cap and attempts < 20:
            # 找出性价比最低的 (考虑了 Potential 和 Fame)
            # 我们不想踢掉高潜力的年轻人，也不想踢掉能带来巨大流量的球星
            # Value = PER / Salary * (1 + 0.5*Is_Young) * (1 + 0.5*Fame)
            score_metric = (current_roster['per_norm'] / (current_roster['salary_2025'] + 1)) * \
                           (1 + 0.5 * current_roster['is_young_talent']) * \
                           (1 + 0.5 * current_roster['fame'])
            
            # 找出分数最低的（最该被踢掉的）
            drop_idx = score_metric.idxmin()
            
            # 保护机制：如果是 Package Deal 的一部分，尽量不要踢 (除非没办法)
            # 这里简化处理：直接踢，如果踢掉了关键人，下一轮循环也不会再加回来，导致 Nepotism 惩罚生效，
            # 从而让遗传算法自然淘汰这个解。
            
            if drop_idx in roster_idx:
                roster_idx.remove(drop_idx)
            
            # 填补底薪
            if len(roster_idx) < 11:
                cheap_players = self.player_pool[
                    (self.player_pool['salary_2025'] < 80000) & 
                    (~self.player_pool.index.isin(roster_idx))
                ]
                if not cheap_players.empty:
                    add_idx = cheap_players['per_norm'].idxmax()
                    roster_idx.append(add_idx)

            current_roster = self.player_pool.iloc[roster_idx]
            total_sal = current_roster['salary_2025'].sum()
            attempts += 1
            
        return roster_idx[:12]

    def run_genetic_algorithm(self, generations=50, population_size=40):
        print(f">>> 启动 Pro-Insight 2.0 优化 (Gens: {generations}, Pop: {population_size})...")
        # 初始化种群
        population = []
        for _ in range(population_size):
            r_idx = list(np.random.choice(self.player_pool.index, 12, replace=False))
            r_idx = self.repair_roster(r_idx)
            population.append([r_idx, 0, 0]) # Promo/Price 暂时设为0，由 objective 内部处理或简化

        best_solution = None
        best_score = -np.inf
        
        for gen in range(generations):
            scores = []
            for indiv in population:
                s = self.objective_function(indiv)
                scores.append(s)
                if s > best_score:
                    best_score = s
                    best_solution = deepcopy(indiv)
            
            # 简单的精英+变异进化
            sorted_idx = np.argsort(scores)[::-1]
            new_pop = [population[i] for i in sorted_idx[:int(population_size*0.2)]] # 保留前20%
            
            while len(new_pop) < population_size:
                parent = population[random.choice(sorted_idx[:20])]
                child_roster = deepcopy(parent[0])
                
                # 变异
                if random.random() < 0.4:
                    if len(child_roster) > 0:
                        child_roster.remove(random.choice(child_roster))
                        avail = list(set(self.player_pool.index) - set(child_roster))
                        if avail: child_roster.append(random.choice(avail))
                
                child_roster = self.repair_roster(child_roster)
                new_pop.append([child_roster, 0, 0])
            
            population = new_pop
            if gen % 10 == 0:
                print(f"Gen {gen}: Best RAROC Score = {best_score:,.0f}")

        return best_solution

    def report(self, solution):
        r_idx = solution[0]
        roster = self.player_pool.iloc[r_idx]
        
        print("\n" + "="*70)
        print("PRO-INSIGHT 2.0 最终决策报告 (风险调整版)")
        print("="*70)
        
        # 运行一次高精度模拟用于展示
        exp, cvar, future = self.simulate_season_performance(roster, n_sims=1000)
        
        print(f"【财务风险评估 (RAROC Analysis)】")
        print(f"  - 预期利润 (Expected Profit):   ${exp:,.2f}")
        print(f"  - 极端风险 (CVaR 10%):        ${cvar:,.2f} (最坏情况下的平均值)")
        print(f"  - 风险敞口 (Risk Exposure):     ${(exp - cvar):,.2f}")
        print(f"  - 潜力估值加成 (Future Bonus):  ${future:,.2f}")
        
        print(f"\n【阵容结构分析】")
        print(f"  - 总薪资: ${roster['salary_2025'].sum():,.2f}")
        print(f"  - 平均伤病风险: {roster['injury_prob'].mean()*100:.1f}%")
        print(f"  - 海外疲劳球员: {roster['plays_overseas'].sum()} 人")
        print(f"  - 潜力新秀 (Young Talent): {roster['is_young_talent'].sum()} 人")
        
        # 检查裙带关系
        active_stars = set(roster['player'])
        print(f"\n【球星绑定检查 (Package Deals)】")
        for star, partner in self.package_deals.items():
            if star in active_stars:
                status = "满足" if partner in active_stars else "违约 (严重惩罚)"
                print(f"  - {star} + {partner}: {status}")

        print(f"\n【最终名单】")
        cols = ['player', 'position', 'salary_2025', 'per_norm', 'injury_prob', 'is_young_talent']
        print(roster[cols].sort_values('salary_2025', ascending=False).to_string(index=False))

# 运行
if __name__ == "__main__":
    opt = WNBA_Optimizer_Pro_Max()
    best_sol = opt.run_genetic_algorithm(generations=60, population_size=500)
    opt.report(best_sol)