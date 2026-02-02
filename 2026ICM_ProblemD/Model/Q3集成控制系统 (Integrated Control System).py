import pandas as pd     # 数据处理库
import numpy as np      # 数学运算库
import random           # 随机数库
import warnings         # 用于控制警告信息
import math             # 数学函数
import os               # 操作系统接口
from copy import deepcopy # 深拷贝，用于遗传算法中复制个体，防止内存地址冲突

# 忽略 Pandas 的一些版本兼容性警告，保持输出整洁
warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置 (Global Configuration)
# 这些是模型的"出厂设置"，Q3 控制器稍后会动态修改这些值
# ==============================================================================
CONFIG = {
    # 权重体系：决定了我们评价球员时看重什么 (竞技/商业/健康/潜力/结构)
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    
    # 目标函数系数：决定 Z-Score 的构成 (财务/能力/结构)
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    
    'SALARY_CAP': 1200000,   # 初始工资帽 (会被 Q3 里的通胀系数修改)
    'ROSTER_MIN': 11,        # 阵容最少人数
    'ROSTER_MAX': 12,        # 阵容最多人数
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)}, # 位置要求
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5, # 收益计算参数
}

# ==============================================================================
# Part 1: Q1 商业模型 (Business Logic)
# 作用：计算在这个阵容下，球队能赚多少钱，估值多少
# ==============================================================================

class IndianaFever_Manager_IPSO_SA:
    def __init__(self, data_path="."):
        print(">>> [Q1 Engine] 初始化印第安纳狂热队 (IND) 商业决策模型 ...")
        self.team_name = "Indiana Fever"
        self.data_path = data_path
        
        # 财务基准参数 (Hard-coded 的事实数据)
        self.base_revenue_2024 = 34000000     # 去年赚了多少
        self.base_valuation_2025 = 335000000  # 现在的估值
        self.fixed_venue_cost = 8000000       # 场馆水电、安保等固定开销
        
        # 初始模型参数 (会被 Q3 控制器动态修改)
        self.ticket_elasticity = -0.6  # 票价弹性：涨价10%，观众减少6%
        self.marketing_roi = 3.5       # 营销ROI：投1块钱广告，赚3.5块
        self.salary_cap = CONFIG['SALARY_CAP']

    def simulate_detailed(self, roster, m_budget, p_price, n_sims=50):
        """
        【蒙特卡洛模拟】
        输入：阵容、营销预算(M)、票价(P)
        输出：预测的利润、营收、估值
        """
        results = {'profit': [], 'total_rev': [], 'valuation': []}
        
        # 1. 计算阵容的基础影响力
        if roster is None or roster.empty:
            team_fame = 1.0 # 空阵容兜底
            total_sal = 800000
        else:
            team_fame = roster['Vi_base'].sum() * 1.5 # 阵容越强，名气越大
            total_sal = roster['salary_2025'].sum()   # 总工资支出
        
        # 2. 巨星效应 (Caitlin Clark Effect)
        # 如果阵容里有 Clark，所有收入系数 * 1.3
        has_clark = False
        if not roster.empty:
            has_clark = roster['player'].astype(str).str.contains('Clark', case=False).any()
        clark_factor = 1.3 if has_clark else 1.0

        # 3. 开始 N 次随机模拟
        for _ in range(n_sims):
            # 随机波动：模拟市场需求的不可预测性 (正态分布)
            demand_noise = np.random.normal(1.0, 0.05) 
            
            # 需求函数：价格越高需求越低，但有 Clark 会提升需求
            demand = (1 + self.ticket_elasticity * (p_price - 1)) * demand_noise * clark_factor
            
            # 计算三部分收入
            rev_ticket = (self.base_revenue_2024 * 0.45) * p_price * demand # 门票
            rev_promo = (m_budget * 1e6) * self.marketing_roi * np.log(1 + team_fame) # 营销
            rev_spon = (self.base_revenue_2024 * 0.55) * clark_factor # 赞助
            
            total_rev = rev_ticket + rev_promo + rev_spon
            total_cost = total_sal + (m_budget * 1e6) + self.fixed_venue_cost
            
            profit = total_rev - total_cost
            # 估值模型：基础估值 + 营收增长带来的溢价
            valuation = self.base_valuation_2025 + (total_rev - self.base_revenue_2024) * 6
            
            results['profit'].append(profit)
            results['total_rev'].append(total_rev)
            results['valuation'].append(valuation)
            
        # 取 N 次模拟的平均值作为最终预测
        avg_res = {k: np.mean(v) for k, v in results.items()}
        avg_res['cvar'] = np.percentile(results['profit'], 5) # 风险底线 (最差的那5%情况)
        return avg_res

    def ipso_sa_optimizer(self, current_roster, max_iter=20, pop_size=20):
        """
        【网格搜索优化器】
        作用：给定一套阵容，算出定多少票价(P)和投多少广告(M)最赚钱。
        """
        best_p = 1.0
        best_m = 1.0
        max_score = -1e9
        
        # 暴力搜索范围：票价 1.0~2.0倍，营销 0.5M~6.0M
        p_range = np.linspace(1.0, 2.0, 10)
        m_range = np.linspace(0.5, 6.0, 10)
        
        for p in p_range:
            for m in m_range:
                # 跑 5 次模拟看效果
                metrics = self.simulate_detailed(current_roster, m, p, n_sims=5)
                # 目标：既要现在赚钱(Profit)，又要未来值钱(Valuation)
                score = 0.6 * metrics['profit'] + 0.4 * (metrics['valuation'] * 0.05)
                
                if score > max_score:
                    max_score = score
                    best_p = p
                    best_m = m
                    
        return [None, best_m, best_p] # 返回最优的 M 和 P

    def calculate_2030_projections(self, profit_2025, valuation_2025):
        """简单的复利公式，预测5年后的情况"""
        g = 0.12 # 假设每年增长 12%
        years = 5
        profit_2030 = profit_2025 * ((1 + g) ** years)
        val_2030 = valuation_2025 * ((1 + g) ** years)
        return profit_2030, val_2030

# ==============================================================================
# Part 2: Q2 数据处理与阵容求解 (Data & Algorithm)
# ==============================================================================

class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file
        self.salary_file = salary_file

    def normalize_name(self, series):
        """清洗名字：去标点、空格、转小写，确保 'A.J. Wilson' 能匹配 'aj wilson'"""
        return series.astype(str).str.lower().str.strip().str.replace(r'[^\w\s]', '', regex=True)

    def load_and_process(self, target_team_name="Indiana Fever"):
        print(f">>> [DataEngine] 正在读取并增强数据...")
        
        try:
            # 读取 Excel
            df_sal = pd.read_excel(self.salary_file)
            df_stats_raw = pd.read_excel(self.stats_file)
            
            # 列名清洗
            df_sal.columns = [c.lower().strip() for c in df_sal.columns]
            df_stats_raw.columns = [c.lower().strip() for c in df_stats_raw.columns]
            
            # 生成匹配用的 Key
            df_sal['key'] = self.normalize_name(df_sal['player'])
            df_stats_raw['key'] = self.normalize_name(df_stats_raw['player'])
            
            # 缺列补零
            req_cols = ['player', 'points', 'rebounds', 'assists', 'team', 'key']
            for c in req_cols:
                if c not in df_stats_raw.columns: df_stats_raw[c] = 0
            
            # 聚合：同一个球员多行数据取平均
            df_stats = df_stats_raw.groupby('key', as_index=False).agg({
                'player': 'first', 'team': 'first',
                'points': 'mean', 'rebounds': 'mean', 'assists': 'mean'
            })
            
            # 连表 (Merge)
            df_merged = pd.merge(df_stats, df_sal[['key', 'salary_2025', 'position']], on='key', how='left')

            # --- 关键步骤：数据插补 (Imputation) ---
            # 很多球员没有薪资数据（比如边缘球员）。不能删掉，否则没法凑齐人数。
            # 我们随机给他们赋予 $65k - $90k 的底薪。
            missing_sal_mask = df_merged['salary_2025'].isna()
            fill_values = np.random.randint(64154, 90000, size=missing_sal_mask.sum())
            df_merged.loc[missing_sal_mask, 'salary_2025'] = fill_values
            
            # 位置和位置映射
            df_merged['position'] = df_merged['position'].fillna('F')
            df_merged['pos_mapped'] = df_merged['position'].apply(self._map_pos)

            # 计算归一化能力值 Vi_base (0~1之间)
            raw_val = df_merged['points'] + df_merged['rebounds']*1.2 + df_merged['assists']*1.5
            df_merged['Vi_base'] = (raw_val - raw_val.min()) / (raw_val.max() - raw_val.min() + 1e-6)
            
            # 队伍名清洗
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
        # 确定主队
        target = current_team_name.lower().strip().replace('.', '')
        if 'fever' in target:
            mask_home = self.pool['team_clean'].str.contains('fever') | self.pool['team_clean'].str.contains('ind')
        else:
            mask_home = self.pool['team_clean'] == target

        self.current_roster = self.pool[mask_home].copy().reset_index(drop=True)
        # 为了算法跑得快，只取市场上最强的前150人作为候选
        self.market_pool = self.pool[~mask_home].sort_values('Vi_base', ascending=False).head(150).reset_index(drop=True)
        
        self.n_current = len(self.current_roster)
        self.n_market = len(self.market_pool)
        self.cap_limit = CONFIG['SALARY_CAP']

    def _repair_dna(self, dna):
        """
        【启发式修复】
        如果随机生成的解超工资帽了，这个函数会尝试“踢掉最贵的，换个便宜的”。
        (注：当前代码中 get_fitness 里这一行被注释掉了，属于可选的高级优化)
        """
        # ... (逻辑省略，与Q2一致) ...
        return dna

    def get_fitness(self, dna):
        """
        【适应度函数】
        输入：DNA (0/1数组)
        输出：分数 (分数越高越好)
        """
        mask_keep = dna[:self.n_current] == 1
        mask_buy = dna[self.n_current:] == 1
        
        roster = pd.concat([self.current_roster.iloc[mask_keep], self.market_pool.iloc[mask_buy]])
        
        tc = roster['salary_2025'].sum()
        count = len(roster)
        
        base_score = roster['Vi_base'].sum() * 1000 # 基础分：能力值总和
        
        # --- 惩罚项 (Penalty) ---
        penalty = 0
        
        # 1. 工资帽惩罚 (软约束)：超得越多，罚得越重 (3次方惩罚)
        if tc > self.cap_limit:
            ratio = tc / self.cap_limit
            penalty += base_score * (ratio ** 3) 
            # 穷鬼线：花钱太少也要罚 (Optional)
            min_spend = self.cap_limit * 0.90
            if tc < min_spend:
                penalty += (min_spend - tc) * 0.5
            
        # 2. 人数惩罚：必须在 11-12 人之间
        if count < 11:
            penalty += 100000  # 缺人重罚
        elif count > 12:
            penalty += (count - 12) * 50000

        return base_score - penalty

    def solve(self, generations=50, pop_size=50):
        """
        【遗传算法主循环】
        """
        total_len = self.n_current + self.n_market
        
        # 1. 初始化种群 (Population Initialization)
        pop = []
        for _ in range(pop_size):
            d = np.zeros(total_len, dtype=int)
            # 随机保留约 30% 现有球员，随机购买 10% 市场球员
            d[:self.n_current] = [1 if random.random() < 0.3 else 0 for _ in range(self.n_current)]
            d[self.n_current:] = [1 if random.random() < 0.1 else 0 for _ in range(self.n_market)]
            pop.append(d)
        
        best_dna = pop[0]
        best_fit = -1e9
        
        # 2. 迭代进化
        for gen in range(generations):
            # 计算全员分数
            fits = [self.get_fitness(d) for d in pop]
            max_idx = np.argmax(fits)
            
            # 记录历史最佳
            if fits[max_idx] > best_fit:
                best_fit = fits[max_idx]
                best_dna = deepcopy(pop[max_idx])
            
            # 精英策略：下一代先把最好的这个保留下来
            new_pop = [best_dna] 
            
            # 繁殖下一代
            while len(new_pop) < pop_size:
                # 锦标赛选择：随机挑俩，选强的做父母
                p1 = pop[random.randint(0, pop_size-1)]
                p2 = pop[random.randint(0, pop_size-1)]
                parent = p1 if self.get_fitness(p1) > self.get_fitness(p2) else p2
                
                child = deepcopy(parent)
                # 变异 (Mutation)：15% 概率随机改变某个位置的状态 (0变1，1变0)
                if random.random() < 0.15:
                    idx = random.randint(0, total_len-1)
                    child[idx] = 1 - child[idx]
                new_pop.append(child)
            pop = new_pop
            
        return best_dna, best_fit
    

# ==============================================================================
# Part 3: Q3 动态控制器 (The "Brain")
# 作用：连接 Q1 和 Q2，模拟外部冲击，协调全局
# ==============================================================================

class Expansion_Strategy_Controller:
    def __init__(self, q1_manager, q2_data_processor, team_name="Indiana Fever"):
        print("\n" + "#"*80)
        print(">>> [Q3 Controller] 初始化联盟扩展与动态战略调整系统...")
        print("#"*80)
        
        self.team_name = team_name
        self.q1_manager = q1_manager
        self.data_processor = q2_data_processor
        
        # 定义“冲击”参数 (Shock Parameters)
        # 模拟联盟扩军带来的变化
        self.expansion_shock = {
            'cap_inflation': 1.15,      # 冲击1: 工资帽上涨 15%
            'market_dilution': 0.05,    # 冲击2: 市场关注度被新队稀释 (未使用，但可预留)
            'draft_loss_count': 2       # 冲击3: 扩展选秀抢走了市场上最好的 2 名自由球员
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
        
        # 更新 Q1 (财务部) 的工资帽
        self.q1_manager.salary_cap = new_cap
        # 更新 Q1 的市场参数：竞争加剧导致票价弹性变大 (观众对价格更敏感)
        old_eta = self.q1_manager.ticket_elasticity
        self.q1_manager.ticket_elasticity = old_eta * 1.1 
        
        # 更新 Q2 (体育部) 的全局配置
        global CONFIG
        CONFIG['SALARY_CAP'] = new_cap
        
        print(f"  - 2026 新工资帽: ${new_cap:,.0f} (Growth: +15%)")
        print(f"  - 票价敏感度调整: {old_eta:.2f} -> {self.q1_manager.ticket_elasticity:.2f}")

        # ======================================================
        # Step 2: 商业参数再优化 (Re-Optimize P & M)
        # ======================================================
        print("\n[Step 2] 扩展环境下的商业决策优化 (Running Q1)...")
        # 让 Q1 基于现状算出最优的 票价(P) 和 营销预算(M)
        q1_sol = self.q1_manager.ipso_sa_optimizer(current_roster)
        opt_m, opt_p = q1_sol[1], q1_sol[2]
        
        print(f"  >>> 最优策略建议: 票价 {opt_p:.2f}x | 营销投入 ${opt_m:.2f}M")

        # ======================================================
        # Step 3: 阵容动态调整 (Dynamic Roster Solver)
        # ======================================================
        print("\n[Step 3] 扩展选秀与阵容调整 (Running Q2)...")
        
        # 3.1 模拟扩展选秀冲击
        # 找出市场上最强的几个人
        market_pool = full_df[~mask_home].sort_values('Vi_base', ascending=False)
        print(f"  - 扩展选秀已摘走: {market_pool.iloc[0]['player']}, {market_pool.iloc[1]['player']}")
        
        # 从池子里把这几个人删掉，模拟他们被新队伍选走了
        solver_pool = full_df.drop(market_pool.index[:self.expansion_shock['draft_loss_count']])
        
        # 3.2 运行 Q2 求解器，寻找新环境下的最优阵容
        solver = StrategicSolver(solver_pool, self.team_name)
        best_dna, best_z = solver.solve(generations=60, pop_size=60)
        
        # ======================================================
        # Step 4: 生成报告
        # ======================================================
        # 解码 DNA，还原成名单
        mask_keep = best_dna[:solver.n_current] == 1
        mask_buy = best_dna[solver.n_current:] == 1
        
        roster_keep = solver.current_roster.iloc[mask_keep]
        roster_buy = solver.market_pool.iloc[mask_buy]
        final_roster = pd.concat([roster_keep, roster_buy])
        
        # 最后跑一次 Q1，预测这个新阵容明年的财报
        final_metrics = self.q1_manager.simulate_detailed(final_roster, opt_m, opt_p, n_sims=50)
        prof_30, val_30 = self.q1_manager.calculate_2030_projections(final_metrics['profit'], final_metrics['valuation'])
        
        self.print_report(final_roster, new_cap, opt_p, opt_m, final_metrics, prof_30, val_30, roster_buy)
        
        return final_roster, final_metrics

    def print_report(self, roster, cap, p, m, metrics, p30, v30, new_players):
        """打印漂亮的最终报告"""
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

        # 打印完整名单
        print("\n" + "-"*60)
        print(f"【2026 赛季 {self.team_name} 完整大名单 (Full Roster)】")
        print("-"*60)
        
        view_df = roster.copy()
        view_df = view_df.sort_values(by=['salary_2025', 'Vi_base'], ascending=[False, False])
        
        cols_to_show = ['player', 'pos_mapped', 'team', 'salary_2025', 'Vi_base']
        valid_cols = [c for c in cols_to_show if c in view_df.columns]
        
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
# 主程序入口 (Main)
# ==========================================
if __name__ == "__main__":
    # 请确保这两个文件在当前目录下
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'
    f_salary = 'player_salaries_2025.xlsx'
    
    # 检查文件是否存在
    if not os.path.exists(f_stats):
        print(f"Error: 缺少 {f_stats}，无法运行真实数据分析。")
    else:
        # 1. 实例化 Q1 引擎 (商业)
        q1 = IndianaFever_Manager_IPSO_SA()
        
        # 2. 实例化 Q2 处理 (数据)
        q2_proc = DataProcessor(f_stats, f_salary)
        
        # 3. 实例化 Q3 控制器并运行 (全局指挥)
        controller = Expansion_Strategy_Controller(q1, q2_proc)
        controller.run_dynamic_adjustment()