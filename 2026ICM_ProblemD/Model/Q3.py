import pandas as pd
import numpy as np
import random
import warnings
import math
import os
from copy import deepcopy

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置 (Q2 Core Configuration)
# ==============================================================================
CONFIG = {
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    'SALARY_CAP': 1200000,   # 初始 2025 工资帽
    'ROSTER_MIN': 11,
    'ROSTER_MAX': 12,
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
}

# ==============================================================================
# Part 1: Q1 商业模型 (IndianaFever_Manager_IPSO_SA)
# ==============================================================================

class IndianaFever_Manager_IPSO_SA:
    def __init__(self, data_path="."):
        print(">>> [Q1 Engine] 初始化印第安纳狂热队 (IND) 商业决策模型 ...")
        self.team_name = "Indiana Fever"
        self.data_path = data_path
        
        # 财务基准参数 (基于真实财报估算)
        self.base_revenue_2024 = 34000000     # 2024 预估营收
        self.base_valuation_2025 = 335000000  # 2025 预估估值
        self.fixed_venue_cost = 8000000       # 固定场馆成本
        
        # 初始模型参数
        self.ticket_elasticity = -0.6  # 票价需求弹性
        self.marketing_roi = 3.5       # 营销回报率
        self.salary_cap = CONFIG['SALARY_CAP']

    def simulate_detailed(self, roster, m_budget, p_price, n_sims=50):
        """
        基于给定阵容、营销预算(M)和票价倍率(P)进行蒙特卡洛模拟
        """
        results = {'profit': [], 'total_rev': [], 'valuation': []}
        
        # 计算阵容的基础商业价值
        if roster is None or roster.empty:
            team_fame = 1.0
            total_sal = 800000
        else:
            team_fame = roster['Vi_base'].sum() * 1.5 # 影响力因子
            total_sal = roster['salary_2025'].sum()
        
        # 核心球员加成 (Caitlin Clark 效应)
        has_clark = False
        if not roster.empty:
            has_clark = roster['player'].astype(str).str.contains('Clark', case=False).any()
        
        clark_factor = 1.3 if has_clark else 1.0

        for _ in range(n_sims):
            # 需求模型
            demand_noise = np.random.normal(1.0, 0.05)
            demand = (1 + self.ticket_elasticity * (p_price - 1)) * demand_noise * clark_factor
            
            # 收入流
            rev_ticket = (self.base_revenue_2024 * 0.45) * p_price * demand
            rev_promo = (m_budget * 1e6) * self.marketing_roi * np.log(1 + team_fame)
            rev_spon = (self.base_revenue_2024 * 0.55) * clark_factor # 赞助收入
            
            total_rev = rev_ticket + rev_promo + rev_spon
            total_cost = total_sal + (m_budget * 1e6) + self.fixed_venue_cost
            
            profit = total_rev - total_cost
            valuation = self.base_valuation_2025 + (total_rev - self.base_revenue_2024) * 6
            
            results['profit'].append(profit)
            results['total_rev'].append(total_rev)
            results['valuation'].append(valuation)
            
        avg_res = {k: np.mean(v) for k, v in results.items()}
        avg_res['cvar'] = np.percentile(results['profit'], 5) # 风险底线
        return avg_res

    def ipso_sa_optimizer(self, current_roster, max_iter=20, pop_size=20):
        """
        简化版优化器：在给定阵容下，寻找最优的 M (Budget) 和 P (Price)
        """
        best_p = 1.0
        best_m = 1.0
        max_score = -1e9
        
        # 网格搜索范围：票价 1.0~2.0倍，营销 0.5M~5.0M
        p_range = np.linspace(1.0, 2.0, 10)
        m_range = np.linspace(0.5, 6.0, 10)
        
        for p in p_range:
            for m in m_range:
                metrics = self.simulate_detailed(current_roster, m, p, n_sims=5)
                # 目标函数：60% 利润 + 40% 估值增长
                score = 0.6 * metrics['profit'] + 0.4 * (metrics['valuation'] * 0.05)
                
                if score > max_score:
                    max_score = score
                    best_p = p
                    best_m = m
                    
        return [None, best_m, best_p]

    def calculate_2030_projections(self, profit_2025, valuation_2025):
        g = 0.12 # 长期增长率
        years = 5
        profit_2030 = profit_2025 * ((1 + g) ** years)
        val_2030 = valuation_2025 * ((1 + g) ** years)
        return profit_2030, val_2030

# ==============================================================================
# Part 2: Q2 数据处理与阵容求解 (DataProcessor & StrategicSolver)
# ==============================================================================

class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file
        self.salary_file = salary_file

    def normalize_name(self, series):
        return series.astype(str).str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)

    def load_and_process(self, target_team_name="Indiana Fever"):
        print(f">>> [DataEngine] 正在读取并增强数据...")
        
        try:
            # 1. 读取数据
            df_sal = pd.read_excel(self.salary_file)
            df_stats_raw = pd.read_excel(self.stats_file)
            
            # 统一列名
            df_sal.columns = [c.lower().strip() for c in df_sal.columns]
            df_stats_raw.columns = [c.lower().strip() for c in df_stats_raw.columns]
            
            # 生成 Key
            df_sal['key'] = self.normalize_name(df_sal['player'])
            df_stats_raw['key'] = self.normalize_name(df_stats_raw['player'])
            req_cols = ['player', 'points', 'rebounds', 'assists', 'team', 'key']
            for c in req_cols:
                if c not in df_stats_raw.columns: df_stats_raw[c] = 0
            
            # 聚合统计数据
            df_stats = df_stats_raw.groupby('key', as_index=False).agg({
                'player': 'first', 'team': 'first',
                'points': 'mean', 'rebounds': 'mean', 'assists': 'mean'
            })
            
            # 合并：以统计数据为主表
            df_merged = pd.merge(df_stats, df_sal[['key', 'salary_2025', 'position']], on='key', how='left')

            # 3. 智能填充缺失数据 (Data Imputation)
            # WNBA 2025 底薪约为 $64,154，新秀约为 $70,000
            # 如果没有薪资数据，随机赋予一个 $65k - $90k 的底薪，模拟角色球员
            missing_sal_mask = df_merged['salary_2025'].isna()
            fill_values = np.random.randint(64154, 90000, size=missing_sal_mask.sum())
            df_merged.loc[missing_sal_mask, 'salary_2025'] = fill_values
            
            # 默认位置
            df_merged['position'] = df_merged['position'].fillna('F')
            df_merged['pos_mapped'] = df_merged['position'].apply(self._map_pos)

            # 4. 计算能力值 Vi
            raw_val = df_merged['points'] + df_merged['rebounds']*1.2 + df_merged['assists']*1.5
            df_merged['Vi_base'] = (raw_val - raw_val.min()) / (raw_val.max() - raw_val.min() + 1e-6)
            
            # 5. 队伍名称清洗
            df_merged['team_clean'] = self.normalize_name(df_merged['team'].fillna('Free Agent'))
            
            # 去重
            df_merged = df_merged.drop_duplicates(subset=['key'])
            
            print(f"  球员池扩大 {len(df_merged)} 人")
            print(f"  其中薪资低于 $100k 的球员数: {len(df_merged[df_merged['salary_2025'] < 100000])}")
            
            return df_merged

        except Exception as e:
            import traceback
            print(f"!!! Error: {e}")
            traceback.print_exc()
            return None

    def _map_pos(self, p):
        p = str(p).upper()
        if 'C' in p: return 'C'
        if 'F' in p: return 'F'
        return 'G'
    
class StrategicSolver:
    def __init__(self, pool, current_team_name):
        self.pool = pool
        target = current_team_name.lower().strip().replace('.', '')
        
        if 'fever' in target:
            mask_home = self.pool['team_clean'].str.contains('fever') | self.pool['team_clean'].str.contains('ind')
        else:
            mask_home = self.pool['team_clean'] == target

        self.current_roster = self.pool[mask_home].copy().reset_index(drop=True)
        # 扩大搜索范围：取前 150 名
        self.market_pool = self.pool[~mask_home].sort_values('Vi_base', ascending=False).head(150).reset_index(drop=True)
        
        self.n_current = len(self.current_roster)
        self.n_market = len(self.market_pool)
        self.cap_limit = CONFIG['SALARY_CAP']

    def _repair_dna(self, dna):
        """
        如果超帽，强制将选中的最贵球员替换为未选中的最便宜球员
        """
        # 将 DNA 转换为实际索引
        indices_keep = [i for i, x in enumerate(dna[:self.n_current]) if x == 1]
        indices_buy = [i for i, x in enumerate(dna[self.n_current:]) if x == 1]
        
        # 构建当前选中球员的临时 DataFrame
        roster_keep = self.current_roster.iloc[indices_keep]
        roster_buy = self.market_pool.iloc[indices_buy]
        
        current_salary = roster_keep['salary_2025'].sum() + roster_buy['salary_2025'].sum()
        
        # 循环修复直到工资合规或人数过少
        attempts = 0
        while current_salary > self.cap_limit and attempts < 10:
            # 找出当前阵容里最贵的
            if len(roster_buy) > 0:
                # 找最贵的买入球员
                exp_idx_loc = roster_buy['salary_2025'].idxmax()
                expensive_salary = roster_buy.loc[exp_idx_loc, 'salary_2025']
                
                # 从 DNA 中移除 (找到对应的 market index)
                # 随机关掉一个买入位的 1
                ones_indices = [i for i, x in enumerate(dna[self.n_current:]) if x == 1]
                if ones_indices:
                    # 简单贪婪：关掉任意
                    dna[self.n_current + ones_indices[0]] = 0 
            
            # 重新计算
            mask_keep = dna[:self.n_current] == 1
            mask_buy = dna[self.n_current:] == 1
            cur_sal = self.current_roster[mask_keep]['salary_2025'].sum() + \
                      self.market_pool[mask_buy]['salary_2025'].sum()
            
            if cur_sal <= self.cap_limit:
                break
            attempts += 1
            
        return dna

    def get_fitness(self, dna):
        # 1. 先尝试修复 DNA 
        # dna = self._repair_dna(dna) 
        
        mask_keep = dna[:self.n_current] == 1
        mask_buy = dna[self.n_current:] == 1
        
        roster = pd.concat([self.current_roster.iloc[mask_keep], self.market_pool.iloc[mask_buy]])
        
        tc = roster['salary_2025'].sum()
        count = len(roster)
        
        base_score = roster['Vi_base'].sum() * 1000
        
        # 惩罚项
        penalty = 0
        
        # 工资帽惩罚
        if tc > self.cap_limit:
            ratio = tc / self.cap_limit
            penalty += base_score * (ratio ** 3) # 超得越多罚得越重
            min_spend = self.cap_limit * 0.90
            if tc < min_spend:
                penalty += (min_spend - tc) * 0.5
            
        # 人数约束
        if count < 11:
            penalty += 100000  # 缺人重罚
        elif count > 12:
            penalty += (count - 12) * 50000



        return base_score - penalty

    def solve(self, generations=50, pop_size=50):
        total_len = self.n_current + self.n_market
        # 不再完全随机，而是生成初始解

        pop = []
        for _ in range(pop_size):
            d = np.zeros(total_len, dtype=int)
            # 随机选 3 个现有球员
            d[:self.n_current] = [1 if random.random() < 0.3 else 0 for _ in range(self.n_current)]
            # 随机选 8 个市场球员 
            d[self.n_current:] = [1 if random.random() < 0.1 else 0 for _ in range(self.n_market)]
            pop.append(d)
        
        best_dna = pop[0]
        best_fit = -1e9
        
        for gen in range(generations):
            fits = [self.get_fitness(d) for d in pop]
            max_idx = np.argmax(fits)
            
            if fits[max_idx] > best_fit:
                best_fit = fits[max_idx]
                best_dna = deepcopy(pop[max_idx])
            
            # 进化
            new_pop = [best_dna] # 精英策略
            
            # 锦标赛选择
            while len(new_pop) < pop_size:
                # 随机选两个，谁强选谁
                p1 = pop[random.randint(0, pop_size-1)]
                p2 = pop[random.randint(0, pop_size-1)]
                parent = p1 if self.get_fitness(p1) > self.get_fitness(p2) else p2
                
                child = deepcopy(parent)
                # 变异
                if random.random() < 0.15:
                    idx = random.randint(0, total_len-1)
                    child[idx] = 1 - child[idx]
                new_pop.append(child)
            pop = new_pop
            
        return best_dna, best_fit
    

# ==============================================================================
# Part 3: Q3 动态控制器 (Expansion_Strategy_Controller)
# ==============================================================================

class Expansion_Strategy_Controller:
    def __init__(self, q1_manager, q2_data_processor, team_name="Indiana Fever"):
        print("\n" + "#"*80)
        print(">>> [Q3 Controller] 初始化联盟扩展与动态战略调整系统...")
        print("#"*80)
        
        self.team_name = team_name
        self.q1_manager = q1_manager
        self.data_processor = q2_data_processor
        
        # 外部冲击参数
        self.expansion_shock = {
            'cap_inflation': 1.15,      # 工资帽上涨 15%
            'market_dilution': 0.05,    # 市场关注度稀释
            'draft_loss_count': 2       # 扩展选秀带走 2 名自由市场球员
        }

    def run_dynamic_adjustment(self):
        # 1. 加载全量数据
        full_df = self.data_processor.load_and_process(self.team_name)
        if full_df is None: return None
        
        # 识别当前阵容
        target_clean = self.team_name.lower().strip().replace('.', '')
        if 'fever' in target_clean:
            mask_home = full_df['team_clean'].str.contains('fever') | full_df['team_clean'].str.contains('ind')
        else:
            mask_home = full_df['team_clean'] == target_clean
        
        current_roster = full_df[mask_home]

        # ======================================================
        # Step 1: 环境量化 (Apply Shock)
        # ======================================================
        print("\n[Step 1] 应用联盟扩展冲击参数...")
        old_cap = self.q1_manager.salary_cap
        new_cap = old_cap * self.expansion_shock['cap_inflation']
        
        # 更新 Q1 参数
        self.q1_manager.salary_cap = new_cap
        old_eta = self.q1_manager.ticket_elasticity
        self.q1_manager.ticket_elasticity = old_eta * 1.1 # 竞争加剧，弹性变大
        
        # 更新 Q2 全局配置
        global CONFIG
        CONFIG['SALARY_CAP'] = new_cap
        
        print(f"  - 2026 新工资帽: ${new_cap:,.0f} (Growth: +15%)")
        print(f"  - 票价敏感度调整: {old_eta:.2f} -> {self.q1_manager.ticket_elasticity:.2f}")

        # ======================================================
        # Step 2: 商业参数再优化 (Re-Optimize P & M)
        # ======================================================
        print("\n[Step 2] 扩展环境下的商业决策优化 (Running Q1)...")
        # 假设我们先基于现有阵容优化 P 和 M，为 Q2 提供资金支持信号
        q1_sol = self.q1_manager.ipso_sa_optimizer(current_roster)
        opt_m, opt_p = q1_sol[1], q1_sol[2]
        
        print(f"  >>> 最优策略建议: 票价 {opt_p:.2f}x | 营销投入 ${opt_m:.2f}M")

        # ======================================================
        # Step 3: 阵容动态调整 (Dynamic Roster Solver)
        # ======================================================
        print("\n[Step 3] 扩展选秀与阵容调整 (Running Q2)...")
        
        # 3.1 模拟扩展选秀：移除市场上的前几名高能力值球员
        market_pool = full_df[~mask_home].sort_values('Vi_base', ascending=False)
        print(f"  - 扩展选秀已摘走: {market_pool.iloc[0]['player']}, {market_pool.iloc[1]['player']}")
        
        # 真正用于求解的数据池
        solver_pool = full_df.drop(market_pool.index[:self.expansion_shock['draft_loss_count']])
        
        # 3.2 运行求解器
        solver = StrategicSolver(solver_pool, self.team_name)
        best_dna, best_z = solver.solve(generations=60, pop_size=60)
        
        # ======================================================
        # Step 4: 生成报告
        # ======================================================
        mask_keep = best_dna[:solver.n_current] == 1
        mask_buy = best_dna[solver.n_current:] == 1
        
        roster_keep = solver.current_roster.iloc[mask_keep]
        roster_buy = solver.market_pool.iloc[mask_buy]
        final_roster = pd.concat([roster_keep, roster_buy])
        
        # 再跑一次 Q1 模拟，预测最终财务
        final_metrics = self.q1_manager.simulate_detailed(final_roster, opt_m, opt_p, n_sims=50)
        prof_30, val_30 = self.q1_manager.calculate_2030_projections(final_metrics['profit'], final_metrics['valuation'])
        
        self.print_report(final_roster, new_cap, opt_p, opt_m, final_metrics, prof_30, val_30, roster_buy)
        
        return final_roster, final_metrics

    def print_report(self, roster, cap, p, m, metrics, p30, v30, new_players):
        print("\n" + "="*80)
        print(f"【PRO-INSIGHT 最终战略报告】 联盟扩展适应性方案 ({self.team_name})")
        print("="*80)
        
        total_sal = roster['salary_2025'].sum()
        
        print(f"1. 财务战略调整 (Financial Strategy):")
        print(f"   - 预算上限 (Cap):   ${cap:,.0f}")
        print(f"   - 票价策略 (Price): {p:.2f}x (利用球星效应抵消市场稀释)")
        print(f"   - 营销策略 (Mkt):   ${m:.2f}M (高投入抢占新市场份额)")
        
        print(f"\n2. 2026 预期绩效 (Performance Projection):")
        print(f"   - 赛季预期营收:     ${metrics['total_rev']/1e6:.2f} M")
        print(f"   - 赛季预期利润:     ${metrics['profit']/1e6:.2f} M")
        print(f"   - 2030 品牌估值:    ${v30/1e6:.0f} M")
        
        print(f"\n3. 阵容调整结果 (Roster Moves):")
        print(f"   - 最终人数: {len(roster)} 人")
        print(f"   - 薪资总额: ${total_sal:,.0f} (使用率 {total_sal/cap:.1%})")
        
        if not new_players.empty:
            print("\n   [重点引援名单 (Top Acquisitions)]")
            print(new_players[['player', 'pos_mapped', 'salary_2025', 'Vi_base']].to_string(index=False))
        else:
            print("\n   [无外部引援，维持现有阵容]")

        # =================================================================
        # >>> NEW: 添加完整大名单打印逻辑
        # =================================================================
        print("\n" + "-"*60)
        print(f"【2026 赛季 {self.team_name} 完整大名单 (Full Roster)】")
        print("-"*60)
        
        # 复制数据以避免影响原数据
        view_df = roster.copy()
        
        # 按薪资从高到低排序，如果薪资相同按能力值排序
        view_df = view_df.sort_values(by=['salary_2025', 'Vi_base'], ascending=[False, False])
        
        # 选取要展示的列，确保列存在
        cols_to_show = ['player', 'pos_mapped', 'team', 'salary_2025', 'Vi_base']
        valid_cols = [c for c in cols_to_show if c in view_df.columns]
        
        # 格式化打印
        print(view_df[valid_cols].to_string(
            index=False, 
            formatters={
                'salary_2025': lambda x: f"${x:,.0f}",
                'Vi_base': lambda x: f"{x:.4f}"
            },
            justify='left'
        ))
        print("-"*60)
            
        print("="*80)

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 请确保这两个文件在当前目录下
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'
    f_salary = 'player_salaries_2025.xlsx'
    
    # 检查文件是否存在
    if not os.path.exists(f_stats):
        print(f"Error: 缺少 {f_stats}，无法运行真实数据分析。")
    else:
        # 1. 实例化 Q1 引擎
        q1 = IndianaFever_Manager_IPSO_SA()
        
        # 2. 实例化 Q2 数据处理
        q2_proc = DataProcessor(f_stats, f_salary)
        
        # 3. 实例化 Q3 控制器并运行
        controller = Expansion_Strategy_Controller(q1, q2_proc)
        controller.run_dynamic_adjustment()