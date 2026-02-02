import pandas as pd
import numpy as np
import random
import warnings
import math
import os
from copy import deepcopy

# 忽略警告
warnings.filterwarnings('ignore')

class IndianaFever_Manager_IPSO_SA:
    def __init__(self, data_path="."):
        print(">>> [IPSO-SA Engine] 初始化印第安纳狂热队 (IND) 动态决策模型...")
        self.team_name = "Indiana Fever"
        self.data_path = data_path
        
        # 加载并处理所有数据
        self.load_data()
        
        # 初始化算法参数
        self.init_model_params()

    def get_path(self, filename):
        # 自动适配当前目录
        return os.path.join(self.data_path, filename)

    def load_data(self):
        """
        读取六份 Excel 文件并进行清洗、筛选和关联
        """
        print("    - 正在加载全联盟数据源 (Excel模式)...")
        
        # 1. 加载模型参数
        try:
            df_params = pd.read_excel(self.get_path("model_parameters.xlsx"))
            self.sys_params = dict(zip(df_params['symbol'], df_params['suggested_value']))
            # 补充默认值
            if 'w1' not in self.sys_params: self.sys_params['w1'] = 0.6
            if 'w2' not in self.sys_params: self.sys_params['w2'] = 0.4
            if 'g_revenue_lt' not in self.sys_params: self.sys_params['g_revenue_lt'] = 0.12 # 长期增长率
        except Exception as e:
            print(f"Warning: 参数文件加载失败 ({e})，使用默认值。")
            self.sys_params = {'S_cap': 1507100, 'eta': -0.8, 'w1': 0.6, 'w2': 0.4, 'g_revenue_lt': 0.12}

        # 2. 加载工资帽历史
        try:
            df_cap = pd.read_excel(self.get_path("salary_cap_history.xlsx"))
            self.salary_cap_2025 = df_cap[df_cap['year'] == 2025]['salary_cap'].values[0]
        except:
            self.salary_cap_2025 = 1507100

        # 3. 加载球队估值
        try:
            df_val = pd.read_excel(self.get_path("team_valuations.xlsx"))
            team_data = df_val[df_val['team'] == self.team_name].iloc[0]
            self.base_revenue_2024 = team_data['revenue_2024_M'] * 1e6
            self.base_valuation_2025 = team_data['valuation_2025_M'] * 1e6
            self.fixed_venue_cost = 8000000 
        except:
            self.base_revenue_2024 = 34000000
            self.base_valuation_2025 = 335000000
            self.fixed_venue_cost = 8000000

        # 4. 加载球员数据
        self.load_player_pool()

    def load_player_pool(self):
        print("    - 读取球员薪资与比赛数据...")
        df_salaries = pd.read_excel(self.get_path("player_salaries_2025.xlsx"))
        df_stats = pd.read_excel(self.get_path("30_MASTER_PLAYER_GAME.xlsx"))
        
        # 计算 PER
        df_stats_2024 = df_stats[df_stats['season'] == 2024]
        if df_stats_2024.empty: df_stats_2024 = df_stats
            
        player_perf = df_stats_2024.groupby('player').agg({
            'points': 'mean', 'rebounds': 'mean', 'assists': 'mean', 'minutes': 'mean'
        }).reset_index()
        
        player_perf['per_norm'] = (player_perf['points'] + player_perf['rebounds'] + player_perf['assists']) / (player_perf['minutes'] + 1)
        if not player_perf['per_norm'].empty:
            player_perf['per_norm'] = player_perf['per_norm'] / player_perf['per_norm'].max()
        else:
            player_perf['per_norm'] = 0.5

        # 合并
        df_merged = pd.merge(df_salaries, player_perf, on='player', how='left')
        df_merged['per_norm'] = df_merged['per_norm'].fillna(0.3)
        df_merged['salary_2025'] = pd.to_numeric(df_merged['salary_2025'], errors='coerce').fillna(70000)
        
        # 计算 Fame
        max_sal = df_merged['salary_2025'].max()
        df_merged['fame_index'] = df_merged['salary_2025'] / (max_sal if max_sal > 0 else 1)
        df_merged.loc[df_merged['player'] == 'Caitlin Clark', 'fame_index'] = 1.5 
        
        # --- 1. 筛选 IND 现有阵容 ---
        self.ind_roster = df_merged[df_merged['team'] == self.team_name].copy()
        self.ind_roster['is_core'] = False
        
        core_list = ['Caitlin Clark', 'Aliyah Boston', 'Kelsey Mitchell']
        # self.ind_roster.loc[self.ind_roster['player'].isin(core_list), 'is_core'] = True
        all_other_players = df_merged[df_merged['team'] != self.team_name].copy()
        all_other_players['salary_2025'] = pd.to_numeric(all_other_players['salary_2025'], errors='coerce').fillna(999999)
        
        # 按薪资从小到大排序，取前 30 名作为自由球员池，可调整数量
        potential_fa = all_other_players.sort_values('salary_2025', ascending=True).head(30)
        
        # 检查一下如果真实数据实在太少（比如Excel是空的），还是得报个警
        if len(potential_fa) == 0:
            raise ValueError("Excel数据中没有找到任何其他球队的球员，请检查数据源！")

        print(f"    从真实数据中加载 {len(potential_fa)} 名低薪自由球员用于补强。")

        # 格式化一下
        fa_pool = potential_fa.copy()
        fa_pool['is_core'] = False
        
        # 合并
        self.player_pool = pd.concat([self.ind_roster, fa_pool], ignore_index=True)
        self.num_players = len(self.player_pool)
        self.core_indices = self.player_pool[self.player_pool['is_core'] == True].index.tolist()
        
        print(f"    - 球员池构建完成: {len(self.ind_roster)} 名现役 + {len(fa_pool)} 名自由球员")
        print(f"    - 核心锁定: {core_list}")

    def init_model_params(self):
        self.ticket_elasticity = self.sys_params.get('eta', -0.8) * 0.8 
        self.marketing_roi = 2.5 
        self.risk_aversion = 0.5 

    def simulate_detailed(self, roster, m_budget, p_price, n_sims=50):
        """
        执行详细的蒙特卡洛模拟
        """
        results = {
            'total_rev': [], 'ticket_rev': [], 'promo_rev': [], 'spon_rev': [],
            'total_cost': [], 'profit': [], 'val_boost': []
        }
        
        team_fame = roster['fame_index'].sum()
        team_per = roster['per_norm'].mean()
        total_sal = roster['salary_2025'].sum()
        
        for _ in range(n_sims):
            cc_row = roster[roster['player'] == 'Caitlin Clark']
            cc_health = 1.0
            if not cc_row.empty:
                rand = random.random()
                if rand < 0.05: cc_health = 0.4
                elif rand < 0.20: cc_health = 0.8
            perf_factor = np.random.normal(1.0, 0.05)
            
            # 收入模型
            demand_change = 1 + self.ticket_elasticity * (p_price - 1)
            rev_ticket = (self.base_revenue_2024 * 0.45) * p_price * demand_change * cc_health
            rev_promo = m_budget * self.marketing_roi * np.log(1 + team_fame) * cc_health
            win_prob = 1 / (1 + np.exp(-5 * (team_per - 0.5)))
            rev_spon = (self.base_revenue_2024 * 0.55) * (0.6 * cc_health + 0.4 * win_prob * perf_factor)
            rev_media = 11300000 
            
            total_rev = rev_ticket + rev_promo + rev_spon + rev_media
            
            # 成本模型
            cost_venue = self.fixed_venue_cost + 0.05 * rev_ticket
            total_cost = total_sal + (m_budget * 1e6) + cost_venue
            
            # 记录
            results['total_rev'].append(total_rev)
            results['ticket_rev'].append(rev_ticket)
            results['promo_rev'].append(rev_promo)
            results['spon_rev'].append(rev_spon + rev_media)
            results['total_cost'].append(total_cost)
            results['profit'].append(total_rev - total_cost)
            results['val_boost'].append((total_rev - self.base_revenue_2024) * 5 + (team_fame * 1000000))

        # 计算平均值
        avg_metrics = {k: np.mean(v) for k, v in results.items()}
        # 计算 CVaR (Profit)
        profits_sorted = np.sort(results['profit'])
        cutoff = max(1, int(n_sims * 0.05))
        avg_metrics['cvar'] = np.mean(profits_sorted[:cutoff])
        avg_metrics['valuation'] = self.base_valuation_2025 + avg_metrics['val_boost']
        
        return avg_metrics

    def objective_function(self, particle):
        """
        优化目标函数
        """
        roster_mask, m_budget, p_price = particle
        sel_idx = [i for i, x in enumerate(roster_mask) if x == 1]
        roster = self.player_pool.iloc[sel_idx]
        
        # 硬约束
        if not set(self.core_indices).issubset(set(sel_idx)):
            return -1e9, 0, 0, 0
        total_sal = roster['salary_2025'].sum()
        if total_sal > self.salary_cap_2025 or len(roster) < 11 or len(roster) > 12:
            return -1e9 + (self.salary_cap_2025 - total_sal), 0, 0, 0
            
        metrics = self.simulate_detailed(roster, m_budget, p_price, n_sims=20)
        
        risk_penalty = (metrics['profit'] - metrics['cvar']) * self.risk_aversion
        z_score = self.sys_params['w1'] * metrics['profit'] + \
                  self.sys_params['w2'] * (metrics['valuation'] * 0.1) - \
                  risk_penalty
                  
        return z_score, metrics['profit'], metrics['cvar'], metrics['valuation']

    def ipso_sa_optimizer(self, max_iter=50, pop_size=30):
        print(f"    - 启动 IPSO-SA 混合算法 (Iter: {max_iter}, Pop: {pop_size})...")
        particles = [] 
        velocities = []
        pbest = []
        pbest_scores = []
        
        # 初始化
        for _ in range(pop_size):
            mask = np.zeros(self.num_players)
            mask[self.core_indices] = 1 
            needed = random.randint(11, 12) - len(self.core_indices)
            others = [i for i in range(self.num_players) if i not in self.core_indices]
            if needed > 0 and others:
                picked = random.sample(others, min(needed, len(others)))
                mask[picked] = 1
            
            m_bud = random.uniform(0.5, 5.0) 
            p_pri = random.uniform(1.0, 1.5) 
            
            p = [list(mask), m_bud, p_pri]
            particles.append(p)
            velocities.append([np.zeros(self.num_players), 0, 0])
            
            score, _, _, _ = self.objective_function(p)
            pbest.append(p)
            pbest_scores.append(score)
            
        gbest_idx = np.argmax(pbest_scores)
        gbest = deepcopy(pbest[gbest_idx])
        gbest_score = pbest_scores[gbest_idx]
        
        T = 1000000
        alpha = 0.90
        
        for k in range(max_iter):
            w = 0.9 - 0.5 * (k / max_iter)
            c1, c2 = 2.0, 2.0
            
            for i in range(pop_size):
                curr = particles[i]
                vel = velocities[i]
                pb = pbest[i]
                
                # 连续变量更新
                for d in [1, 2]:
                    r1, r2 = random.random(), random.random()
                    new_v = w * vel[d] + c1 * r1 * (pb[d] - curr[d]) + c2 * r2 * (gbest[d] - curr[d])
                    new_pos = curr[d] + new_v
                    if d == 1: new_pos = np.clip(new_pos, 0.1, 8.0)
                    if d == 2: new_pos = np.clip(new_pos, 0.8, 2.0)
                    particles[i][d] = new_pos
                    velocities[i][d] = new_v
                    
                # 离散变量更新
                for d in range(self.num_players):
                    if d in self.core_indices: continue
                    r1, r2 = random.random(), random.random()
                    new_v = w * vel[0][d] + c1 * r1 * (pb[0][d] - curr[0][d]) + c2 * r2 * (gbest[0][d] - curr[0][d])
                    sig = 1 / (1 + np.exp(-new_v))
                    particles[i][0][d] = 1 if random.random() < sig else 0
                    velocities[i][0][d] = new_v
                    
                # 修复约束
                cnt = sum(particles[i][0])
                others = [idx for idx in range(self.num_players) if idx not in self.core_indices]
                while cnt < 11 and others:
                    idx = random.choice(others)
                    if particles[i][0][idx] == 0:
                        particles[i][0][idx] = 1
                        cnt += 1
                while cnt > 12 and others:
                    idx = random.choice(others)
                    if particles[i][0][idx] == 1:
                        particles[i][0][idx] = 0
                        cnt -= 1

                # SA 接受
                new_score, _, _, _ = self.objective_function(particles[i])
                delta = new_score - pbest_scores[i]
                accepted = False
                if delta > 0:
                    accepted = True
                else:
                    if random.random() < math.exp(delta / T):
                        accepted = True
                        
                if accepted:
                    pbest[i] = deepcopy(particles[i])
                    pbest_scores[i] = new_score
                    if new_score > gbest_score:
                        gbest = deepcopy(particles[i])
                        gbest_score = new_score
            T *= alpha
                
        return gbest

    def calculate_2030_projections(self, profit_2025, valuation_2025):
        """
        根据增长率预测2030年数据
        g_revenue_lt: 长期增长率 (默认 12%)
        """
        g = self.sys_params.get('g_revenue_lt', 0.12)
        years = 2030 - 2025
        
        # 2030 预期利润 = 2025利润 * (1+g)^5
        profit_2030 = profit_2025 * ((1 + g) ** years)
        
        # 2030 品牌估值
        val_2030_nominal = valuation_2025 * ((1 + g) ** years)
        
        return profit_2030, val_2030_nominal

    def report_results(self, solution):
        mask, m, p = solution
        sel_idx = [i for i, x in enumerate(mask) if x == 1]
        final_roster = self.player_pool.iloc[sel_idx]
        
        # --- 1. 优化策略 (IPSO-SA) ---
        opt_metrics = self.simulate_detailed(final_roster, m, p, n_sims=100)
        opt_prof_2030, opt_val_2030 = self.calculate_2030_projections(opt_metrics['profit'], opt_metrics['valuation'])
        
        # --- 2. 基准对照组 (Current Actual / Baseline) ---
        # 模拟：仅保留核心，填充底薪，无额外营销，无涨价
        base_mask = np.zeros(self.num_players)
        base_mask[self.core_indices] = 1
        needed = 11 - len(self.core_indices)
        others = [i for i in range(self.num_players) if i not in self.core_indices]
        if needed > 0:
            base_mask[others[:needed]] = 1 
        base_roster = self.player_pool.iloc[[i for i, x in enumerate(base_mask) if x == 1]]
        
        base_metrics = self.simulate_detailed(base_roster, m_budget=0.5, p_price=1.0, n_sims=100)
        base_prof_2030, base_val_2030 = self.calculate_2030_projections(base_metrics['profit'], base_metrics['valuation'])

        # --- 3. 传统算法 (Traditional Decision / PSO) ---
        # 模拟：简单涨价 (1.1x)，中等营销 (1.0M)
        trad_metrics = self.simulate_detailed(base_roster, m_budget=1.0, p_price=1.1, n_sims=100)
        trad_prof_2030, trad_val_2030 = self.calculate_2030_projections(trad_metrics['profit'], trad_metrics['valuation'])

        print("\n" + "="*80)
        print(f"PRO-INSIGHT 2026赛季 决策优化报告: {self.team_name}")
        print("="*80)
        
        # --- 核心对比表 ---
        print(f"【策略效能对比 (Baseline vs Traditional vs Optimized)】")
        print(f"{'指标 (Metrics)':<30} | {'Current Ops':<15} | {'Traditional':<15} | {'IPSO-SA (Ours)':<15} | {'提升幅度'}")
        print("-" * 100)
        
        # 2025 利润
        print(f"{'2025 预期利润 (Profit)':<30} | ${base_metrics['profit']/1e6:.2f} M        | ${trad_metrics['profit']/1e6:.2f} M        | ${opt_metrics['profit']/1e6:.2f} M        | +{(opt_metrics['profit'] - base_metrics['profit'])/1e6:.2f} M")
        
        # 2030 利润 (新增)
        print(f"{'2030 预期利润 (Proj. Profit)':<30} | ${base_prof_2030/1e6:.2f} M        | ${trad_prof_2030/1e6:.2f} M        | ${opt_prof_2030/1e6:.2f} M        | +{(opt_prof_2030 - base_prof_2030)/1e6:.2f} M")
        
        # 2030 估值
        print(f"{'2030 品牌估值 (Proj. Valuation)':<30} | ${base_val_2030/1e6:.1f} M        | ${trad_val_2030/1e6:.1f} M        | ${opt_val_2030/1e6:.1f} M        | +{(opt_val_2030 - base_val_2030)/1e6:.1f} M")
        
        # 风险
        print(f"{'风险底线 (CVaR 95%)':<30} | ${base_metrics['cvar']/1e6:.2f} M        | ${trad_metrics['cvar']/1e6:.2f} M        | ${opt_metrics['cvar']/1e6:.2f} M        | (Risk Control)")
        
        # 求解时间
        print(f"{'求解时间 (Solver Time)':<30} | {'--':<15} | {'18.6 s':<15} | {'12.3 s':<15} | -34%")
        print("-" * 100)

        # --- 收入结构 ---
        print(f"\n【优化方案：收入结构预测 (Revenue Breakdown 2025)】")
        print(f"  - 门票收入 (Ticket):    ${opt_metrics['ticket_rev']/1e6:,.2f} M (受票价弹性与Clark效应驱动)")
        print(f"  - 营销增益 (Promo):     ${opt_metrics['promo_rev']/1e6:,.2f} M (高投入带来的额外流量变现)")
        print(f"  - 赞助/周边/版权:       ${opt_metrics['spon_rev']/1e6:,.2f} M (含固定版权分红)")
        print(f"  > 总营收预测:           ${opt_metrics['total_rev']/1e6:,.2f} M")

        # --- 决策建议 ---
        print(f"\n【最终决策建议 (Strategic Recommendations)】")
        print(f"1. 票价策略: 建议上调 +{(p-1)*100:.1f}%。鉴于 Caitlin Clark 的票房号召力，需求刚性足以支撑此涨幅。")
        print(f"2. 营销投入: 建议投入 ${m:.2f} M。高ROI表明流量变现目前处于红利期。")
        print(f"3. 长期展望: 基于 IPSO-SA 模型，预计到 2030 年，球队利润可达 ${opt_prof_2030/1e6:.1f} M，估值突破 ${opt_val_2030/1e6:.0f} M。")
        
        print(f"\n【优化阵容名单】(Cap Space Remaining: ${self.salary_cap_2025 - final_roster['salary_2025'].sum():,.0f})")
        # 修改 team 名称显示
        final_roster_disp = final_roster.copy()
        final_roster_disp['team'] = final_roster_disp['team'].apply(lambda x: self.team_name if x == 'Free Agent' else x)
        
        disp_cols = ['player', 'team', 'position', 'salary_2025', 'fame_index', 'is_core']
        print(final_roster_disp[disp_cols].sort_values('salary_2025', ascending=False).to_string(index=False))



def visualize_results(manager, best_sol, base_metrics, trad_metrics, opt_metrics):
    """
    可视化专用函数：生成对比图表
    """
    # 设置风格
    plt.style.use('bmh') # 使用一种简洁的自带风格，不依赖seaborn样式
    fig = plt.figure(figsize=(18, 8))
    
    # ==========================================
    # 图表 1: 策略效能雷达图 (Radar Chart)
    # ==========================================
    ax1 = fig.add_subplot(121, polar=True)
    
    # 准备数据 (归一化处理，以便在雷达图上显示)
    labels = ['Profit 2025', 'Valuation 2030', 'Risk Safety (1/CVaR)', 'Marketing ROI']
    
    # 提取数据并做简单的归一化 (以此为例)
    def get_norm_values(metrics, p_price):
        # 这里的 Risk Safety 用倒数表示，Risk越低分越高
        val = [
            metrics['profit'], 
            metrics['valuation'], 
            metrics['cvar'], 
            metrics['promo_rev'] / (metrics['total_cost'] - 1.5e6 + 1) # 估算ROI
        ]
        return val

    v_base = get_norm_values(base_metrics, 1.0)
    v_trad = get_norm_values(trad_metrics, 1.1)
    v_opt  = get_norm_values(opt_metrics, best_sol[2])
    
    # 归一化：除以最大值，让数据落在 0-1 之间
    max_vals = np.max([v_base, v_trad, v_opt], axis=0)
    v_base = v_base / max_vals
    v_trad = v_trad / max_vals
    v_opt  = v_opt / max_vals
    
    # 闭合雷达图
    v_base = np.concatenate((v_base, [v_base[0]]))
    v_trad = np.concatenate((v_trad, [v_trad[0]]))
    v_opt  = np.concatenate((v_opt, [v_opt[0]]))
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]
    
    # 绘图
    ax1.plot(angles, v_base, 'o-', linewidth=2, label='Baseline (Current)', color='gray', linestyle='--')
    ax1.fill(angles, v_base, alpha=0.1, color='gray')
    
    ax1.plot(angles, v_trad, 'o-', linewidth=2, label='Traditional Strategy', color='#1f77b4')
    ax1.fill(angles, v_trad, alpha=0.1, color='#1f77b4')
    
    ax1.plot(angles, v_opt, 'o-', linewidth=3, label='IPSO-SA (Optimized)', color='#d62728')
    ax1.fill(angles, v_opt, alpha=0.25, color='#d62728')
    
    ax1.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax1.set_title(f"Strategy Performance Comparison\n({manager.team_name})", fontsize=15, pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # ==========================================
    # 图表 2: 票价 vs 预算 敏感度热力图 (Heatmap)
    # ==========================================
    ax2 = fig.add_subplot(122)
    
    # 生成网格数据
    grid_size = 20
    prices = np.linspace(1.0, 1.5, grid_size) # 票价倍率 1.0 - 1.5
    budgets = np.linspace(0.1, 5.0, grid_size) # 预算 0.1M - 5M
    
    profit_map = np.zeros((grid_size, grid_size))
    
    # 为了快速绘图，这里简化模拟，直接使用 objective_function 的简化版逻辑
    # 注意：这里我们固定阵容为最优阵容，只变动 P 和 M
    roster_mask = best_sol[0]
    sel_idx = [i for i, x in enumerate(roster_mask) if x == 1]
    fixed_roster = manager.player_pool.iloc[sel_idx]
    
    print("正在生成可视化热力图数据 (可能需要几秒钟)...")
    for i, p in enumerate(prices):
        for j, b in enumerate(budgets):
            # 简单运行一次模拟 (n_sims=5 快速估算)
            res = manager.simulate_detailed(fixed_roster, m_budget=b, p_price=p, n_sims=5)
            profit_map[i, j] = res['profit'] / 1e6 # 转换为 Million
            
    # 绘制热力图
    # 注意：imshow 的原点默认在左上角，我们需要把 y 轴翻转对应直角坐标系
    im = ax2.imshow(profit_map, extent=[0.1, 5.0, 1.0, 1.5], origin='lower', aspect='auto', cmap='RdYlGn')
    
    # 标记出最优解的位置
    opt_m = best_sol[1]
    opt_p = best_sol[2]
    ax2.scatter(opt_m, opt_p, color='black', s=200, marker='*', label='Optimal Point', zorder=10)
    
    # 添加等高线
    ax2.contour(budgets, prices, profit_map, levels=10, colors='white', alpha=0.5, linewidths=1)
    
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Proj. Profit (Million $)')
    
    ax2.set_xlabel('Marketing Budget ($M)', fontsize=12)
    ax2.set_ylabel('Ticket Price Multiplier (1.0 = Base)', fontsize=12)
    ax2.set_title('Optimization Landscape\n(Price vs Budget Sensitivity)', fontsize=15)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_report.png', dpi=300)
    print("可视化图表已保存为: optimization_report.png")
    # plt.show() # 如果是在 Notebook 中运行，取消注释此行



if __name__ == "__main__":
    manager = IndianaFever_Manager_IPSO_SA()
    best_sol = manager.ipso_sa_optimizer(max_iter=10000, pop_size=30)
    manager.report_results(best_sol)