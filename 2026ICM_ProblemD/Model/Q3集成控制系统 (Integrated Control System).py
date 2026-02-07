import pandas as pd
import numpy as np
import random
import warnings
import math
import os
from copy import deepcopy
import io
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor 
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局配置 (Q2 Core Configuration)
# ==============================================================================
CONFIG = {
    'W_A': 0.30,  'W_B': 0.20,  'W_H': 0.20,  'W_P': 0.15,  'W_F': 0.15,
    'MU_1': 0.40, 'MU_2': 0.30, 'MU_3': 0.30,
    'SALARY_CAP': 1500000,   # Initial, will be overwritten by prediction
    'ROSTER_MIN': 11,
    'ROSTER_MAX': 12,
    'POS_REQ': {'G': (4, 6), 'F': (4, 6), 'C': (2, 3)},
    'TR_ALPHA': 500000, 'TR_BETA': 1000, 'RISK_FACTOR': 0.5,
    'SALARY_MIN_LEAGUE': 64154,
    'SALARY_MAX_LEAGUE': 241984,
    'TOTAL_GAME_MINUTES': 200,
    'PLAYER_MAX_MINUTES': 38.0,
    'PLAYER_MIN_MINUTES': 5.0,
    'N_HIGHRISK_MAX': 2,
    'N_POTENTIAL_MIN': 2
}


# ==============================================================================
# Part 0: Auxiliary Models (Forecasting & Impact)
# ==============================================================================
class SalaryCapForecaster:
    """
    使用 MLP (多层感知机)
    """
    def __init__(self):
        # 历史数据
        self.raw_data = pd.read_csv(io.StringIO("""year,salary_cap
2020,1324200
2021,1339066
2022,1379237
2023,1421014
2024,1463200
2025,1507100
2026,1552300"""))
        
    def predict_next_years(self, start_year=2026, n_years=1):
        # 构造时序窗口特征 (模拟 LSTM window)
        df = self.raw_data
        X = np.array(df['year']).reshape(-1, 1)
        y = np.array(df['salary_cap'])
        
        # 使用 MLP 拟合非线性增长趋势
        model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='lbfgs', max_iter=5000)
        model.fit(X, y)
        
        predictions = {}
        for i in range(n_years):
            target_year = start_year + i
            pred = model.predict([[target_year]])[0]
            predictions[target_year] = pred
        return predictions
    
class MarketShockModel:
    """
    实现 Logit 市场份额模型和渠道溢价逻辑。
    """
    def __init__(self, home_city="Indianapolis"):
        self.home_city = home_city  
        self.S_market_0 = 1.0       
        self.sigma_k = 0.15
        
    def calculate_logit_share(self, expansion_cities_count=2):
        """
        Logit 模型: S_market = S_0 * Product(1 - sigma * m_k)
        """
        # m_k 为冲击强度，随扩展城市数量非线性增加
        m_k = np.array([0.8] * expansion_cities_count)
        
        # 连乘公式实现
        dilution_factor = np.prod(1 - self.sigma_k * m_k)
        S_market = self.S_market_0 * dilution_factor
        return S_market

    def get_acquisition_premium(self, channel_type):
        """
        收购成本溢价模型
        C' = C * (1 + tau * sum(m_k))
        """
        tau = 0.2   # 溢价系数
        shock_sum = 0.5 # sum(m_k) 模拟值
        
        # 区分渠道 (q=1 选秀, q=2 自由市场, q=3 交易)
        if channel_type == 'draft': return 0.0      # 选秀无溢价
        if channel_type == 'free_agent': return tau * shock_sum * 0.5 
        if channel_type == 'trade': return tau * shock_sum * 1.0
        return 0.0

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
        self.ticket_elasticity_base = -0.6 # D0
        self.marketing_roi = 3.5       # 营销回报率
        self.salary_cap = CONFIG['SALARY_CAP']

    def simulate_detailed(self, roster, m_budget, p_price, n_sims=50):
        results = {'profit': [], 'total_rev': [], 'valuation': []}
        
        # 计算阵容的基础商业价值
        if roster is None or roster.empty:
            team_fame = 1.0
            total_sal = 800000
        else:
            team_fame = roster['Vi_base'].sum() * 1.5 # 影响力因子
            total_sal = roster['salary_2025'].sum()
        
        # 核心球员加成
        has_clark = False
        if not roster.empty:
            has_clark = roster['player'].astype(str).str.contains('Clark', case=False).any()
        
        clark_factor = 1.3 if has_clark else 1.0

        phi = min(m_budget / 5.0, 1.0) 
        delta = 0.2
        media_impact = (phi - delta * (phi ** 2)) * 2.0
        shock_model = MarketShockModel()
        S_market = shock_model.calculate_logit_share()
        results = {'profit': [], 'total_rev': [], 'valuation': []}

        for _ in range(n_sims):
            # R_ticket = D0 * P^(e+1) * S_market
  
            # 动态弹性：价格越高，弹性越负
            real_elasticity = self.ticket_elasticity_base - 0.1 * (p_price - 1.0)
            
            revenue_ticket = (self.base_revenue_2024 * 0.45) * \
                             (p_price ** (1 + real_elasticity)) * \
                             S_market * clark_factor
            
            # 联动效应 (Lambda_link): 媒体投入增加票务需求
            lambda_link = 1.0 + 0.15 * media_impact
            revenue_ticket *= lambda_link

            rev_promo = (m_budget * 1e6) * 3.5 * np.log(1 + team_fame)
            rev_spon = (self.base_revenue_2024 * 0.55) * clark_factor 
            
            total_rev = revenue_ticket + rev_promo + rev_spon
            total_cost = total_sal + (m_budget * 1e6) + self.fixed_venue_cost
            
            profit = total_rev - total_cost
            valuation = self.base_valuation_2025 + (total_rev - self.base_revenue_2024) * 6
            
            results['profit'].append(profit)
            results['total_rev'].append(total_rev)
            results['valuation'].append(valuation)
            
        avg_res = {k: np.mean(v) for k, v in results.items()}
        avg_res['cvar'] = np.percentile(results['profit'], 5)
        return avg_res

    def ipso_sa_optimizer(self, current_roster, max_iter=20, pop_size=20):
        print("   -> 启动贝叶斯优化 (GP-UCB) 求解最优 P* 和 M* ...")
        
        # 定义搜索空间: P [0.8, 1.5], M [0.5, 5.0]
        bounds = np.array([[0.8, 1.5], [0.5, 5.0]])
        
        # 初始样本点
        X_init = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(10, 2))
        Y_init = []
        
        for x in X_init:
            res = self.simulate_detailed(current_roster, x[1], x[0], n_sims=5)
            # 目标函数：利润与估值的加权
            score = 0.6 * res['profit'] + 0.4 * (res['valuation'] * 0.05)
            Y_init.append(score)
            
        X_sample = X_init
        Y_sample = np.array(Y_init).reshape(-1, 1)
        
        # 高斯过程回归
        kernel = C(1.0, (1e-3, 1e3)) * Matern(1.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        
        # 简单的贝叶斯迭代
        for i in range(max_iter):
            gp.fit(X_sample, Y_sample)
            
            # 采集函数 (UCB): 寻找均值大且方差大的点
            X_candidates = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(100, 2))
            mu, sigma = gp.predict(X_candidates, return_std=True)
            ucb = mu + 1.96 * sigma
            
            next_idx = np.argmax(ucb)
            x_next = X_candidates[next_idx]
            
            # 评估新点
            res = self.simulate_detailed(current_roster, x_next[1], x_next[0], n_sims=5)
            y_next = 0.6 * res['profit'] + 0.4 * (res['valuation'] * 0.05)
            
            X_sample = np.vstack((X_sample, x_next))
            Y_sample = np.vstack((Y_sample, y_next))
            
        best_idx = np.argmax(Y_sample)
        best_p, best_m = X_sample[best_idx]
        
        return [None, best_m, best_p]

    def calculate_2030_projections(self, profit_2025, valuation_2025):
        g = 0.12 # 长期增长率
        years = 5
        profit_2030 = profit_2025 * ((1 + g) ** years)
        val_2030 = valuation_2025 * ((1 + g) ** years)
        return profit_2030, val_2030

# ==============================================================================
# Part 2: Q2 数据处理与阵容求解
# ==============================================================================
class DataProcessor:
    def __init__(self, stats_file, salary_file):
        self.stats_file = stats_file
        self.salary_file = salary_file

    def normalize_name(self, series):
        """标准化名字以提高匹配率"""
        return series.astype(str).str.lower().str.strip().str.replace('.', '', regex=False)

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

        # 标准化列名
        stats_raw.columns = [c.lower().strip() for c in stats_raw.columns]
        salary_raw.columns = [c.lower().strip() for c in salary_raw.columns]
        
        # --- 2. 提取统计数据 ---
        if 'season' in stats_raw.columns:
            target_season = stats_raw['season'].max()
            stats = stats_raw[stats_raw['season'] == target_season].copy()
        else:
            stats = stats_raw.copy()

        # 确保数值列无误
        stat_cols = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy']
        for c in stat_cols:
            if c not in stats.columns: stats[c] = 0.0
            stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)

        # 聚合球员数据
        metrics = stats.groupby('player').agg({
            'points': 'mean', 'rebounds': 'mean', 'assists': 'mean', 
            'plus_minus': 'mean', 'minutes': 'mean', 'attendance': 'mean',
            'clutch_proxy': 'mean', 'team': 'last' 
        }).reset_index()

        # --- 3. 提取薪资数据 ---
        salary = salary_raw.copy()
        salary['salary_2025'] = pd.to_numeric(salary['salary_2025'], errors='coerce')
        salary['years_of_service'] = pd.to_numeric(salary.get('years_of_service', 0), errors='coerce').fillna(1)
        
        # --- 4. 合并数据 ---
        metrics['key'] = self.normalize_name(metrics['player'])
        salary['key'] = self.normalize_name(salary['player'])
        
        df = pd.merge(metrics, salary[['key', 'salary_2025', 'years_of_service', 'position', 'team']], 
                      on='key', how='outer', suffixes=('', '_sal'))
        
        # 字段合并与修补
        df['player'] = df['player'].fillna(salary['player'])
        df['team'] = df['team'].fillna(df['team_sal']).fillna('Free Agent')
        
        manual_fixes = {
            'aliyah boston': {'salary': 99000, 'pos': 'C', 'team': target_team_name, 'pts': 14.5, 'reb': 8.4, 'min': 30}, # 新秀合同第3年
            'caitlin clark': {'salary': 78066, 'pos': 'G', 'team': target_team_name, 'pts': 19.2, 'reb': 5.7, 'ast': 8.2, 'min': 32, 'att': 17000},
            'kelsey mitchell': {'salary': 250000, 'pos': 'G', 'team': target_team_name, 'pts': 16.0}, # 假定保留/核心
            'erica wheeler': {'salary': 140000, 'pos': 'G', 'team': target_team_name, 'pts': 9.0},
            'nalyssa smith': {'salary': 91000, 'pos': 'F', 'team': target_team_name, 'pts': 10.0, 'reb': 7.0},
            'lexie hull': {'salary': 84000, 'pos': 'G', 'team': target_team_name, 'pts': 6.0},
            'grace berger': {'salary': 73439, 'pos': 'G', 'team': target_team_name, 'pts': 3.0},
            'victaria saxton': {'salary': 69000, 'pos': 'F', 'team': target_team_name, 'pts': 2.0},
            'temi fagbenle': {'salary': 80000, 'pos': 'C', 'team': target_team_name, 'pts': 5.0}, 
            'katie lou samuelson': {'salary': 175000, 'pos': 'F', 'team': target_team_name, 'pts': 8.0},
            'breanna stewart': {'salary': 205000, 'pos': 'F', 'team': 'New York Liberty', 'pts': 23.0, 'reb': 9.3},
            'napheesa collier': {'salary': 214284, 'pos': 'F', 'team': 'Minnesota Lynx', 'pts': 21.5, 'reb': 8.5},
            'aja wilson': {'salary': 210000, 'pos': 'F', 'team': 'Las Vegas Aces', 'pts': 22.8}, 
            'kelsey plum': {'salary': 200000, 'pos': 'G', 'team': 'Las Vegas Aces', 'pts': 18.7},
            'jewell loyd': {'salary': 241984, 'pos': 'G', 'team': 'Seattle Storm', 'pts': 24.0}, 
            'arike ogunbowale': {'salary': 241984, 'pos': 'G', 'team': 'Dallas Wings', 'pts': 21.0},
            'kahleah copper': {'salary': 241984, 'pos': 'G', 'team': 'Phoenix Mercury', 'pts': 18.0},
            'brionna jones': {'salary': 212000, 'pos': 'C', 'team': 'Connecticut Sun', 'pts': 15.0},
            'dewanna bonner': {'salary': 200000, 'pos': 'F', 'team': 'Connecticut Sun', 'pts': 17.4},
            'nneka ogwumike': {'salary': 160000, 'pos': 'F', 'team': 'Seattle Storm', 'pts': 19.1},
            'skylar diggins-smith': {'salary': 214284, 'pos': 'G', 'team': 'Seattle Storm', 'pts': 16.0},
            'chelsea gray': {'salary': 196267, 'pos': 'G', 'team': 'Las Vegas Aces', 'pts': 15.0},
            'jonquel jones': {'salary': 210000, 'pos': 'C', 'team': 'New York Liberty', 'pts': 16.0},
            'alyssa thomas': {'salary': 218000, 'pos': 'F', 'team': 'Connecticut Sun', 'pts': 15.0},
            'natasha cloud': {'salary': 200000, 'pos': 'G', 'team': 'Phoenix Mercury', 'pts': 12.0},
            'diana taurasi': {'salary': 234936, 'pos': 'G', 'team': 'Phoenix Mercury', 'pts': 16.0},
            'allisha gray': {'salary': 185000, 'pos': 'G', 'team': 'Atlanta Dream', 'pts': 17.1},
            'cheyenne parker': {'salary': 190000, 'pos': 'F', 'team': 'Atlanta Dream', 'pts': 15.0},
            'courtney williams': {'salary': 175000, 'pos': 'G', 'team': 'Minnesota Lynx', 'pts': 10.0},
            'dearica hamby': {'salary': 169000, 'pos': 'F', 'team': 'Los Angeles Sparks', 'pts': 16.0},
            'marina mabrey': {'salary': 208000, 'pos': 'G', 'team': 'Chicago Sky', 'pts': 15.0},
            'satou sabally': {'salary': 210000, 'pos': 'F', 'team': 'Dallas Wings', 'pts': 18.0},
            'natasha howard': {'salary': 224675, 'pos': 'F', 'team': 'Dallas Wings', 'pts': 15.0},
            'azura stevens': {'salary': 195000, 'pos': 'F', 'team': 'Los Angeles Sparks', 'pts': 12.0},
            'betnijah laney': {'salary': 185000, 'pos': 'G', 'team': 'New York Liberty', 'pts': 14.0},
            'brittney griner': {'salary': 150000, 'pos': 'C', 'team': 'Phoenix Mercury', 'pts': 17.0},
            'sophie cunningham': {'salary': 130000, 'pos': 'G', 'team': 'Phoenix Mercury'},
            'sydney colson': {'salary': 76297, 'pos': 'G', 'team': 'Las Vegas Aces'}, 
            'kyra lambert': {'salary': 64154, 'pos': 'G', 'team': 'Free Agent'},
            'makayla timpson': {'salary': 65000, 'pos': 'F', 'team': 'Free Agent'},
        }

        for name, data in manual_fixes.items():
            key = self.normalize_name(pd.Series([name]))[0]
            mask = df['key'] == key
            
            if mask.any():
                idx = df[mask].index
                if 'salary' in data: df.loc[idx, 'salary_2025'] = data['salary']
                if 'pos' in data: df.loc[idx, 'position'] = data['pos']
                if 'team' in data: df.loc[idx, 'team'] = data['team']
                if 'pts' in data: df.loc[idx, 'points'] = data['pts']
                if 'reb' in data: df.loc[idx, 'rebounds'] = data['reb']
            else:
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

        # --- 6. 全局缺失值填充 ---
        fill_zeros = ['points', 'rebounds', 'assists', 'plus_minus', 'minutes', 'attendance', 'clutch_proxy', 'years_of_service']
        for c in fill_zeros:
            if c not in df.columns: df[c] = 0.0
            df[c] = df[c].fillna(0.0)
        df['salary_2025'] = df['salary_2025'].fillna(76297)
        if 'position' not in df.columns: df['position'] = 'G'
        df['position'] = df['position'].fillna('G')
        df['pos_mapped'] = df['position'].apply(self._map_pos)

        # 预计算：为了支持连续变量 S_j，为每个球员确定薪资谈判区间
        df['S_min'] = df['salary_2025'].apply(lambda x: max(CONFIG['SALARY_MIN_LEAGUE'], x * 0.8))
        df['S_max'] = df['salary_2025'].apply(lambda x: min(CONFIG['SALARY_MAX_LEAGUE'], x * 1.2))

        # 对于超级巨星，下限较高
        mask_super = df['S_max'] > 200000
        df.loc[mask_super, 'S_min'] = df.loc[mask_super, 'salary_2025'] * 0.9

        df = self._calculate_scores(df)

        target_clean = self.normalize_name(pd.Series([target_team_name]))[0]
        df['team_clean'] = self.normalize_name(df['team'])
        if 'fever' in target_clean:
            mask_home = df['team_clean'].str.contains('fever') | df['team_clean'].str.contains('ind')
            df.loc[mask_home, 'team_clean'] = target_clean
            df.loc[mask_home, 'team'] = target_team_name

        print(f"    - 数据加载完成，全联盟共 {len(df)} 名球员。")
        return df

    def _map_pos(self, p):
        p = str(p).upper()
        if 'C' in p: return 'C'
        if 'F' in p: return 'F'
        return 'G'


    def _calculate_scores(self, df):
        # 静态能力评分 Vi
        def norm(col):
            min_v, max_v = col.min(), col.max()
            if max_v == min_v: return 0.5
            return (col - min_v) / (max_v - min_v)

        df['ViA'] = (0.3 * norm(df['points']) + 0.2 * norm(df['rebounds']) + 
                     0.2 * norm(df['assists']) + 0.15 * norm(df['plus_minus']) + 
                     0.15 * norm(df['clutch_proxy']))
        
        df['ViB'] = 0.5 * norm(df['salary_2025']) + 0.5 * norm(df['attendance'])
        mask_star = (df['salary_2025'] > 200000) | (df['player'].str.contains('Clark', na=False, case=False))
        df.loc[mask_star, 'ViB'] *= 1.5
        df['ViB'] = df['ViB'].clip(0, 1)

        # ViH 健康分现在作为静态基准，具体风险在 evaluate 中结合时间 tj 计算
        df['ViH_base'] = 1 - (0.5 * norm(df['minutes']) * 0.3) 
        df['ViP'] = 1 - norm(df['years_of_service'])
        
        # 基础综合分
        df['Vi_base'] = (CONFIG['W_A'] * df['ViA'] + CONFIG['W_B'] * df['ViB'] + 
                         CONFIG['W_H'] * df['ViH_base'] + CONFIG['W_P'] * df['ViP'])
        df['Vi_base'] = df['Vi_base'].fillna(df['Vi_base'].mean())
        return df

   
# ==========================================
# 3. 求解器
# ==========================================
class Individual:
    """个体类，存储DNA、目标函数值、Rank和拥挤度"""
    def __init__(self, n_total):
        # Part 1: Binary (0/1)
        self.dna_decisions = np.zeros(n_total, dtype=int)
        
        # Part 2: Continuous (0.0 - 1.0) 
        # Actual Salary = S_min + S_factor * (S_max - S_min)
        self.dna_salary = np.random.rand(n_total)
        
        # Part 3: Continuous (0.0 - 1.0)
        # Real Time = (Weight_i / Sum(Weights)) * 200
        self.dna_time = np.random.rand(n_total)

        self.objectives = None 
        self.rank = 0
        self.crowding_dist = 0
        self.violation = 0
        self.z_score = -1e9
    
        self.decoded_roster = None

class StrategicSolver:
    def __init__(self, pool, current_team_name, strategy_mode='expansion'):
        self.strategy_mode = strategy_mode 
        self.shock_model = MarketShockModel()
        pool['team_clean'] = pool['team'].astype(str).str.lower().str.strip()
        target = current_team_name.lower().strip()
        if 'fever' in target:
            mask_home = pool['team_clean'].str.contains('fever') | pool['team_clean'].str.contains('ind')
        else:
            mask_home = pool['team_clean'] == target

        self.current_roster = pool[mask_home].copy().reset_index(drop=True)
        self.market_pool = pool[~mask_home].sort_values('Vi_base', ascending=False).head(80).reset_index(drop=True)

        # 为球员分配渠道标签
        self.full_pool = pd.concat([self.current_roster, self.market_pool]).reset_index(drop=True)
        self.n_total = len(self.full_pool)
        self.n_current = len(self.current_roster)
        print(f">>> [SolverInit] 混合变量优化引擎就绪。变量维度: 3 * {self.n_total}")

        def assign_channel(row):
            if row.name < len(self.current_roster): return 'retained' # 留队
            if row['years_of_service'] <= 2: return 'draft'
            if row['salary_2025'] > 180000: return 'free_agent'
            return 'trade'
            
        self.full_pool['channel'] = self.full_pool.apply(assign_channel, axis=1)
        
        self.n_total = len(self.full_pool)
        self.n_current = len(self.current_roster)
        self.dynamic_weights = self._calculate_dynamic_weights()

    def _calculate_dynamic_weights(self):
        dw = CONFIG.copy()
        if self.strategy_mode == 'expansion':
            # 扩张期：重视商业价值 (B) 和潜力 (P)
            dw['W_A'] = 0.25; dw['W_B'] = 0.30; dw['W_P'] = 0.25
            # 双目标权重：重视估值
            dw['OBJ_PROFIT'] = 0.4; dw['OBJ_VAL'] = 0.6
            # 渠道权重：重视选秀和自由市场
            dw['CH_DRAFT'] = 0.5; dw['CH_FA'] = 0.3; dw['CH_TRADE'] = 0.2
        else:
            # 盈利期/稳定期：重视即战力 (A) 和 稳定性 (H)
            dw['W_A'] = 0.40; dw['W_B'] = 0.15; dw['W_H'] = 0.30
            # 双目标权重：重视利润
            dw['OBJ_PROFIT'] = 0.6; dw['OBJ_VAL'] = 0.4
             # 渠道权重：一视同仁
            dw['CH_DRAFT'] = 0.33; dw['CH_FA'] = 0.33; dw['CH_TRADE'] = 0.33
            
        return dw

    def evaluate(self, ind):
        # 1. 解码二值变量：确定名单
        indices = np.where(ind.dna_decisions == 1)[0]
        
        if len(indices) == 0:
            ind.objectives = [1e9, 1e9]
            ind.violation = 1e9
            return

        roster = self.full_pool.iloc[indices].copy()
        
        # 2. 解码连续变量 S

        s_factors = ind.dna_salary[indices]
        # C' = C * (1 + tau * sum(m))
        def apply_premium(row):
            base_sal = row['S_min'] + ind.dna_salary[row.name] * (row['S_max'] - row['S_min']) # 临时逻辑
            premium_rate = self.shock_model.get_acquisition_premium(row['channel'])
            return base_sal * (1 + premium_rate)
        roster['opt_salary'] = roster.apply(apply_premium, axis=1) 
        
        # 3. 解码连续变量 t (时间)
        t_weights = ind.dna_time[indices]
        total_weight = np.sum(t_weights)
        if total_weight == 0:
            roster['opt_minutes'] = CONFIG['TOTAL_GAME_MINUTES'] / len(roster)
        else:
            roster['opt_minutes'] = (t_weights / total_weight) * CONFIG['TOTAL_GAME_MINUTES']
        
        # 软约束：单人时间限制
        roster['opt_minutes'] = roster['opt_minutes'].clip(CONFIG['PLAYER_MIN_MINUTES'], CONFIG['PLAYER_MAX_MINUTES'])
        
        # 重新归一化以确保总和接近 200
        scale_factor = CONFIG['TOTAL_GAME_MINUTES'] / roster['opt_minutes'].sum()
        roster['opt_minutes'] *= scale_factor
        ind.decoded_roster = roster

        # --- 计算目标 ---
        TC = roster['opt_salary'].sum()
        
        # 表现分计算 (结合时间 t)
        time_ratio = roster['opt_minutes'] / 40.0 # 标准化时间
        
        # 动态健康分 ViH: 时间越长，风险越大
        roster['ViH_dynamic'] = 1.0 - (time_ratio * 0.8) 

        # 使用动态权重计算综合能力 Vis
        w = self.dynamic_weights
        roster['Vi_dynamic'] = (w['W_A'] * roster['ViA'] + w['W_B'] * roster['ViB'] + 
                                w['W_H'] * roster['ViH_dynamic'] + w['W_P'] * roster['ViP'])
        
        # 应用渠道权重 (Channel Weights)
        def get_channel_weight(ch):
            if ch == 'draft': return w['CH_DRAFT']
            if ch == 'free_agent': return w['CH_FA']
            return w['CH_TRADE']
        

        channel_w = roster['channel'].apply(get_channel_weight)

        # 将渠道权重作为系数乘入贡献值，模拟 Solver 对不同渠道的偏好
        roster['Vi_contribution'] = roster['Vi_dynamic'] * time_ratio * (1 + channel_w)
        
        avg_Vi = roster['Vi_contribution'].sum()
        TR = CONFIG['TR_ALPHA'] * math.log(1 + roster['ViA'].sum()) + CONFIG['TR_BETA'] * roster['ViB'].sum()
        Risk = (roster['opt_salary'] * (1 - roster['ViH_dynamic'])).sum() * CONFIG['RISK_FACTOR']
        violation = 0
        count = len(roster)
        
        # 1. 基础人数约束
        if count < CONFIG['ROSTER_MIN']: violation += (CONFIG['ROSTER_MIN'] - count) * 1000
        if count > CONFIG['ROSTER_MAX']: violation += (count - CONFIG['ROSTER_MAX']) * 1000
        if TC > CONFIG['SALARY_CAP']: violation += (TC - CONFIG['SALARY_CAP']) / 100


        
        # 3. 高风险球员限制 (Constraint: Sum(ViH < 0.6) <= N_highrisk)
        # Vi^H < 0.6 的人数小于等于阈值
        n_high_risk = len(roster[roster['ViH_base'] < 0.6])
        if n_high_risk > CONFIG['N_HIGHRISK_MAX']:
             violation += (n_high_risk - CONFIG['N_HIGHRISK_MAX']) * 5000  # 强惩罚

        # 4. 潜力球员下限 (Constraint: Sum(ViP > 0.8) >= N_potential)
        # Vi^P > 0.8 的人数大于等于阈值
        n_potential = len(roster[roster['ViP'] > 0.8])
        if n_potential < CONFIG['N_POTENTIAL_MIN']:
             violation += (CONFIG['N_POTENTIAL_MIN'] - n_potential) * 5000 # 强惩罚

        # 5. 交易人数平衡
        # Acquired: 索引 >= n_current 的球员
        n_acquired = len(roster[roster.index >= self.n_current])
        
        # 初始阵容中被保留的人数
        n_retained = len(roster[roster.index < self.n_current])
        # 被放弃的人数 = 初始总人数 - 保留人数
        n_waived = self.n_current - n_retained
        
        # 交易获得人数 不得超过 放弃人数
        if n_acquired > n_waived:
            violation += (n_acquired - n_waived) * 10000

        # 保留原有核心班底约束
        if n_retained < 4: violation += (4 - n_retained) * 2000

        ind.violation = violation

        # 构造目标函数
        obj_salary = TC
        
        # 表现分计算
        fit_score = 0
        pos_raw_score = 0
        pos_counts = roster['pos_mapped'].value_counts()
        for pos, (min_p, max_p) in CONFIG['POS_REQ'].items():
            c = pos_counts.get(pos, 0)
            if min_p <= c <= max_p: 
                pos_raw_score += 1
            else: 
                pos_raw_score -= 0.5 * abs(c - (min_p + max_p) / 2)
        
        # 归一化 PosFit (映射到 0.0 ~ 1.0)
        PosFit = max(0.0, min(1.0, pos_raw_score * 0.3 + 0.5))

        # --- Part 2: StyleFit ---
        # 阵容平均正负值(avg_pm)越高，协同得分越高。
        # 使用 Logistic Sigmoid 函数平滑映射，中心点为 0
        # avg_pm = 0 -> 0.5分; avg_pm = +4 -> ~0.88分; avg_pm = -4 -> ~0.12分
        avg_pm = roster['plus_minus'].mean()
        # 系数 0.5 控制曲线陡峭程度，可根据需要调整
        StyleFit = 1.0 / (1.0 + math.exp(-0.5 * avg_pm))

        # --- Part 3: 最终 V^F 合成 ---
        ViF = 0.5 * PosFit + 0.5 * StyleFit
 

        term_financial = (TR - Risk) / 10000 

        performance_score = (w['OBJ_PROFIT'] * term_financial + 
                             w['OBJ_VAL'] * avg_Vi * 30) 
    
        
        obj_perf = -performance_score 

        ind.violation = violation
        ind.objectives = [TC, -performance_score]
        ind.z_score = performance_score - violation


    def fast_nondominated_sort(self, population):
        """标准 NSGA-II 非支配排序"""
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in population:
                if p.violation < q.violation: dominates = True
                elif p.violation > q.violation: dominates = False
                else:
                    dominates = ((p.objectives[0] <= q.objectives[0] and p.objectives[1] <= q.objectives[1]) and
                                 (p.objectives[0] < q.objectives[0] or p.objectives[1] < q.objectives[1]))
                if dominates:
                    p.dominated_solutions.append(q)
                elif self._check_dominated(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
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
        """混合变量变异算子"""
        child = deepcopy(ind)
        
        # 1. Binary Mutation (Flip)
        if random.random() < 0.3:
            ones = np.where(child.dna_decisions == 1)[0]
            zeros = np.where(child.dna_decisions == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                child.dna_decisions[np.random.choice(ones)] = 0
                child.dna_decisions[np.random.choice(zeros)] = 1
        
        if random.random() < rate:
            idx = random.randint(0, self.n_total - 1)
            child.dna_decisions[idx] = 1 - child.dna_decisions[idx]

        # 2. Continuous Mutation - Gaussian Noise
        sigma = 0.1 # 标准差
        
        # Salary Mutation
        mask_s = np.random.rand(self.n_total) < rate
        noise_s = np.random.randn(self.n_total) * sigma
        child.dna_salary[mask_s] += noise_s[mask_s]
        child.dna_salary = np.clip(child.dna_salary, 0, 1) 
        
        # Time Mutation
        mask_t = np.random.rand(self.n_total) < rate
        noise_t = np.random.randn(self.n_total) * sigma
        child.dna_time[mask_t] += noise_t[mask_t]
        child.dna_time = np.clip(child.dna_time, 0, 1)

        return child

    def solve(self, generations=100, pop_size=100):
        # 1. 初始化
        population = []
        for _ in range(pop_size):
            ind = Individual(self.n_total)
            
            # 初始化 Decisions
            target_size = random.randint(CONFIG['ROSTER_MIN'], CONFIG['ROSTER_MAX'])
            
            # 必须从当前阵容选一部分
            n_keep = random.randint(4, min(self.n_current, target_size))
            n_buy = target_size - n_keep
            
            idxs_keep = np.random.choice(self.n_current, n_keep, replace=False)
            ind.dna_decisions[idxs_keep] = 1
            
            # 从市场选一部分
            if n_buy > 0:
                market_indices = np.arange(self.n_current, self.n_total)
                market_probs = self.full_pool.iloc[market_indices]['Vi_base'].values
                market_probs = market_probs / market_probs.sum()
                idxs_buy = np.random.choice(market_indices, n_buy, replace=False, p=market_probs)
                ind.dna_decisions[idxs_buy] = 1
            
            # Salary/Time 随机初始化
            self.evaluate(ind)
            population.append(ind)

        # 2. 进化
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

        # 3. 结果
        pareto_front = [p for p in population if p.rank == 0 and p.violation == 0]
        if not pareto_front:
            print("Warning: No fully feasible solution found. Using best approximate.")
            best_ind = max(population, key=lambda x: x.z_score)
        else:
            best_ind = max(pareto_front, key=lambda x: x.z_score)
            
        return best_ind
    

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
        self.cap_predictor = SalaryCapForecaster()
        self.market_model = MarketShockModel(home_city="Indianapolis")
        
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
        # 假设：如果预测的工资帽增长 > 10%，则判定为扩展期，否则为稳定期
        cap_growth = (self.expansion_shock['cap_inflation'] - 1.0)
        current_strategy = 'expansion' if cap_growth > 0.10 else 'profit'
        print(f"\n[Strategy] 检测到环境变化，激活策略模式: {current_strategy.upper()}")
        
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
        # Step 3: 阵容动态调整 
        # ======================================================

        print("\n[Step 3] 扩展选秀与阵容调整 (Running Q2)...")
        
        # 3.1 模拟扩展选秀：移除市场上的前几名高能力值球员
        market_pool = full_df[~mask_home].sort_values('Vi_base', ascending=False)
        print(f"  - 扩展选秀已摘走: {market_pool.iloc[0]['player']}, {market_pool.iloc[1]['player']}")
        
        # 真正用于求解的数据池
        solver_pool = full_df.drop(market_pool.index[:self.expansion_shock['draft_loss_count']])
        solver = StrategicSolver(solver_pool, self.team_name, strategy_mode=current_strategy)

        # best_dna, best_z = solver.solve(generations=60, pop_size=60)
        best_ind = solver.solve(generations=60, pop_size=60)


        # best_dna = best_ind.dna_decisions 
        # best_z = best_ind.z_score     
        
        # # ======================================================
        # # Step 4: 生成报告
        # # ======================================================
        # mask_keep = best_dna[:solver.n_current] == 1
        # mask_buy = best_dna[solver.n_current:] == 1
        # roster_keep = solver.current_roster.iloc[mask_keep]
        # roster_buy = solver.market_pool.iloc[mask_buy]
        # final_roster = pd.concat([roster_keep, roster_buy])
        # final_metrics = self.q1_manager.simulate_detailed(final_roster, opt_m, opt_p, n_sims=50)

        final_roster = best_ind.decoded_roster.copy()
        
        # 区分保留队员和新援
        roster_buy = final_roster[final_roster['channel'] != 'retained']
        
        # 注意：后续调用 simulate_detailed 时，
        final_roster['salary_2025'] = final_roster['opt_salary']
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
        print("\n" + "-"*60)
        print(f"【2026 赛季 {self.team_name} 完整大名单 (Full Roster)】")
        print("-"*60)

        view_df = roster.copy()
        # 按薪资从高到低排序，如果薪资相同按能力值排序
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
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 请确保这两个文件在当前目录下
    f_stats = '30_MASTER_PLAYER_GAME.xlsx'
    f_salary = 'player_salaries_2025.xlsx'
    
    # 检查文件是否存在 不存在提示错误
    if not os.path.exists(f_stats):
        print(f"Error: 缺少 {f_stats}，无法运行真实数据分析。")
    else:
        # 1. 实例化 Q1 引擎
        q1 = IndianaFever_Manager_IPSO_SA()
        
        # 2. 实例化 Q2 数据处理
        q2_proc = DataProcessor(f_stats, f_salary)
        
        # 3. 实例化 Q3  控制器并运行
        controller = Expansion_Strategy_Controller(q1, q2_proc)
        controller.run_dynamic_adjustment()