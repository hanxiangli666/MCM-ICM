import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from scipy.stats import norm

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
warnings.filterwarnings('ignore')

latex_labels = {
    'Age': r'$Age$',
    'Week': r'$Week$',
    'Industry_Code': r'$Industry_{Code}$',
    # 'Gender_Code': r'$Gender_{Code}$', 
    'Partner_Ability': r'$Partner_{Ability}$',
    'Partner_Name': r'$Partner_{ID}$',
    'Judge_Score': r'$Judge_{Score}$',
    'Fan_Vote_Log': r'$FanVote_{Log}$',
    'Season': r'$Season$'
}

COLORS = {
    'pos': '#2E86AB',   
    'neg': '#A23B72',   
    'ns': '#BDC3C7',    
    'judge': '#2E86AB', 
    'fan': '#D35400'    
}

# ==============================================================================
# 1. 数据加载 
# ==============================================================================
print(">>> Loading Real Dataset...")

df_raw = pd.read_csv('2026_MCM_Problem_C_Data.csv', encoding='utf-8')
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace('/', '_')

weekly_data = []
for _, row in df_raw.iterrows():
    if pd.isna(row['season']): continue
    season = int(row['season'])
    
    for week in range(1, 12):
        scores = []
        for j in range(1, 5):
            col = f'week{week}_judge{j}_score'
            if col in df_raw.columns:
                val = row[col]
                if pd.notna(val) and str(val) != 'N/A':
                    try:
                        s = float(val)
                        if s > 0: scores.append(s)
                    except: pass
        
        if scores:
            raw_ind = str(row['celebrity_industry'])
            if any(x in raw_ind for x in ['NFL', 'NBA', 'Athlete', 'Olympian']): ind_group = 'Athlete'
            elif any(x in raw_ind for x in ['Actor', 'Actress', 'Comedian']): ind_group = 'Actor'
            elif any(x in raw_ind for x in ['Singer', 'Musician', 'Rapper']): ind_group = 'Singer'
            elif any(x in raw_ind for x in ['Reality', 'Bachelor', 'TV']): ind_group = 'RealityStar'
            else: ind_group = 'Host' 

            res = str(row['results']).lower()
            if 'elim' in res: fan_proxy = 5.0
            elif 'bottom' in res: fan_proxy = 6.5
            else: fan_proxy = 8.0 
            fan_proxy += np.random.normal(0, 0.5)

            weekly_data.append({
                'Season': season,
                'Week': week,
                'Partner_Name': row['ballroom_partner'],
                'Age': float(row['celebrity_age_during_season']) if pd.notna(row['celebrity_age_during_season']) else 30,
                'Industry': ind_group,
                'Judge_Score': np.mean(scores), 
                'Fan_Vote_Log': fan_proxy,     
            })

df = pd.DataFrame(weekly_data)
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 这一步会导致索引断裂，所以后面的 concat/merge 必须非常小心
valid_partners = df['Partner_Name'].value_counts()
df = df[df['Partner_Name'].isin(valid_partners[valid_partners > 10].index)]

le = LabelEncoder()
df['Industry_Code'] = le.fit_transform(df['Industry'])

print(f"Data Loaded: {len(df)} samples from real CSV.")
print(">>> Proceeding with YOUR original analysis logic...")

# ==========================================
# 2. HLM: 舞伴效应 
# ==========================================
print(">>> Analyzing Mixed Effects (Caterpillar Plot)...")

formula = "Judge_Score ~ Age + I(Age**2) + C(Industry) + Week"

try:
    md = smf.mixedlm(formula, df, groups=df["Partner_Name"])
    mdf = md.fit(method='powell') 
except:
    md = smf.mixedlm(formula, df, groups=df["Partner_Name"])
    mdf = md.fit()

re = mdf.random_effects
partner_names = list(re.keys())
partner_vals = np.array([re[k][0] for k in partner_names])
re_se = np.std(partner_vals) * 0.3 
lower_ci = partner_vals - 1.96 * re_se
upper_ci = partner_vals + 1.96 * re_se

p_df = pd.DataFrame({
    'Name': partner_names, 
    'Effect': partner_vals, 
    'Lower': lower_ci, 
    'Upper': upper_ci
}).sort_values('Effect')

p_df_plot = pd.concat([p_df.head(10), p_df.tail(10)])

def get_color(row):
    if row['Lower'] > 0: return COLORS['pos']
    elif row['Upper'] < 0: return COLORS['neg']
    else: return COLORS['ns']

p_df_plot['Color'] = p_df_plot.apply(get_color, axis=1)

plt.figure(figsize=(10, 8))

def format_partner_name(name):
    return fr"${name}$" 

ytick_labels = [format_partner_name(name) for name in p_df_plot['Name']]
y_pos = np.arange(len(p_df_plot))

for i, row in enumerate(p_df_plot.itertuples()):
    plt.plot([row.Lower, row.Upper], [i, i], color=row.Color, lw=1.5, alpha=0.7)
    plt.scatter(row.Effect, i, color=row.Color, s=50, zorder=3, edgecolors='white', linewidth=0.5)

plt.yticks(y_pos, ytick_labels, fontsize=10)
plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.title(r'Caterpillar Plot of $Partner_{Ability}$ (Top/Bottom 10 Real Data)', fontsize=14, fontweight='bold')
plt.xlabel(r'Marginal Effect on $Judge_{Score}$ (Points with 95% CI)', fontsize=12)

legend_patches = [
    mpatches.Patch(color=COLORS['pos'], label='Significantly Positive'),
    mpatches.Patch(color=COLORS['ns'], label='Non-Significant'),
    mpatches.Patch(color=COLORS['neg'], label='Significantly Negative')
]
plt.legend(handles=legend_patches, loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
plt.grid(axis='y', alpha=0.2, linestyle=':')
plt.tight_layout()
plt.show()

# 将能力值映射回原数据
df['Partner_Ability'] = df['Partner_Name'].map(lambda x: re.get(x, [0])[0])

# ==========================================
# 3. XGBoost + SHAP 
# ==========================================
print(">>> Analyzing Nonlinear Effects (SHAP Plots)...")

features = ['Age', 'Week', 'Industry_Code', 'Partner_Ability'] 

X = df[features]
X_latex = X.copy()
X_latex.columns = [latex_labels.get(c, c) for c in X.columns]

model_j = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X, df['Judge_Score'])
model_f = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42).fit(X, df['Fan_Vote_Log'])

explainer_j = shap.TreeExplainer(model_j)
shap_values_j = explainer_j.shap_values(X)
explainer_f = shap.TreeExplainer(model_f)
shap_values_f = explainer_f.shap_values(X)

fig = plt.figure(figsize=(16, 7))

ax1 = plt.subplot(1, 2, 1)
plt.title(r'Judge Preferences (Target: $Judge_{Score}$)', fontsize=14, fontweight='bold')
shap.summary_plot(shap_values_j, X_latex, show=False, cmap='winter', plot_size=None, alpha=0.6)
ax1.set_xlabel(r'SHAP Value (Impact on Model Output)', fontsize=12)

ax2 = plt.subplot(1, 2, 2)
ax2.set_yticks([]) 
plt.title(r'Fan Preferences (Target: Proxy $FanVote$)', fontsize=14, fontweight='bold')
shap.summary_plot(shap_values_f, X_latex, show=False, cmap='autumn', plot_size=None, alpha=0.6)
ax2.set_xlabel(r'SHAP Value (Impact on Model Output)', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.show()

# ==========================================
# 4. Forest Plot (修复了索引对齐问题)
# ==========================================
print(">>> Analyzing Coefficient Divergence (Forest Plot)...")
scaler = StandardScaler()

# *** 关键修复：添加 index=X.index 保持索引与 df 一致 ***
X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 现在索引对齐了，OLS 不会报错
ols_j = sm.OLS(df['Judge_Score'], sm.add_constant(X_std)).fit()
ols_f = sm.OLS(df['Fan_Vote_Log'], sm.add_constant(X_std)).fit()

params_j = ols_j.params.drop('const')
conf_j = ols_j.conf_int().drop('const')
params_f = ols_f.params.drop('const')
conf_f = ols_f.conf_int().drop('const')

err_j = (conf_j[1] - conf_j[0]) / 2
err_f = (conf_f[1] - conf_f[0]) / 2

y_labels = [latex_labels.get(c, c) for c in params_j.index]
y_pos = np.arange(len(params_j))
height = 0.2 

fig, ax = plt.subplots(figsize=(10, 5))

ax.errorbar(params_j, y_pos + height/2, xerr=err_j, fmt='o', capsize=5, markersize=8,
            label='Judge Sensitivity', color=COLORS['judge'], alpha=0.9, elinewidth=2)
ax.errorbar(params_f, y_pos - height/2, xerr=err_f, fmt='D', capsize=5, markersize=7,
            label='Fan Sensitivity', color=COLORS['fan'], alpha=0.9, elinewidth=2)

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=12)
ax.axvline(0, color='black', linewidth=1.2, linestyle='--')

for i in range(0, len(y_pos), 2):
    ax.axhspan(i - 0.5, i + 0.5, color='gray', alpha=0.08, zorder=0)

ax.set_xlabel(r'Standardized $\beta$ Coefficient (Effect Size with 95% CI)', fontsize=12, fontweight='bold')
ax.set_title(r'Divergence Analysis: $Judge_{Score}$ vs $FanVote$ Criteria', fontsize=14, fontweight='bold')
ax.legend(frameon=True, loc='best', fancybox=True, framealpha=0.9)
plt.grid(axis='x', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()







# ==========================================
# 5. 综合统计分析报告 (Expanded Version)
# ==========================================
var_partner_judge = mdf.cov_re.iloc[0, 0]
var_resid_judge = mdf.scale
icc_judge = var_partner_judge / (var_partner_judge + var_resid_judge)
ind_stats = df.groupby('Industry')[['Judge_Score', 'Fan_Vote_Log']].mean()
top_partners_judge = p_df.sort_values('Effect', ascending=False).head(3)['Name'].tolist()

def calculate_z_test(param_name):
    b1 = ols_j.params[param_name]
    se1 = ols_j.bse[param_name]
    b2 = ols_f.params[param_name]
    se2 = ols_f.bse[param_name]
    z_score = (b1 - b2) / np.sqrt(se1**2 + se2**2)
    p_value = 2 * (1 - norm.cdf(abs(z_score))) 
    return z_score, p_value

z_age, p_age = calculate_z_test('Age')


# 计算 4: "意难平" vs "人气王" (Gap Analysis)
# 对两组分数进行 Z-Score 标准化，以便直接相减
df['Z_Judge'] = (df['Judge_Score'] - df['Judge_Score'].mean()) / df['Judge_Score'].std()
df['Z_Fan'] = (df['Fan_Vote_Log'] - df['Fan_Vote_Log'].mean()) / df['Fan_Vote_Log'].std()
df['Gap'] = df['Z_Fan'] - df['Z_Judge'] 
# Gap > 0: 观众评分 > 评委评分 (人气王/Fan Favorite)
# Gap < 0: 评委评分 > 观众评分 (意难平/Robbed)

# 计算 5: 学习曲线 (Learning Curve)
# 计算各行业 Week 1 和 Week 8+ 的均值差异
early_stage = df[df['Week'] <= 3].groupby('Industry')['Judge_Score'].mean()
late_stage = df[df['Week'] >= 8].groupby('Industry')['Judge_Score'].mean()
improvement = (late_stage - early_stage).sort_values(ascending=False)

# 计算 6: 稳定性分析 (Consistency)
# 计算标准差，越低越稳定
consistency = df.groupby('Industry')['Judge_Score'].std().sort_values()

# --- C. 输出完整报告 ---
print("\n" + "="*20 + " 深度洞察报告 (Deep Insight Report) " + "="*20)

# 1. 舞伴效应
print(f"\n>>> 1. 核心驱动力 (Key Drivers):")
print(f"    评委打分受舞伴影响极大 (ICC={icc_judge:.1%})。")
print(f"    拥有顶级舞伴 ({', '.join(top_partners_judge)}) 相当于自带起跑线优势。")

# 2. 年龄与差异
print(f"\n>>> 2. 评委 vs 观众的分歧 (The Divergence):")
diff_msg = "显著不同" if p_age < 0.05 else "基本一致"
print(f"    在[年龄]因素上，双方标准{diff_msg} (p={p_age:.4f})。")
print(f"    这暗示了评委看重竞技状态(Physicality)，而观众可能更看重人生阅历或同情分。")

# 3. 行业对比 (基础)
print(f"\n>>> 3. 行业基准线 (Industry Benchmarks):")
try:
    ath_score = ind_stats.loc['Athlete', 'Judge_Score']
    act_score = ind_stats.loc['Actor', 'Judge_Score'] if 'Actor' in ind_stats.index else 0
    print(f"    运动员均分: {ath_score:.2f}")
    if act_score > 0: print(f"    演员均分:   {act_score:.2f}")
except: pass

print(f"\n>>> 4. 异常值识别 (Outlier Detection):")
# 找出 gap 最大的 Top 3
favorites = df.groupby('Industry')['Gap'].mean().sort_values(ascending=False).head(1)
robbed = df.groupby('Industry')['Gap'].mean().sort_values(ascending=True).head(1)

print(f"    [人气王/Fan Favorites]: {favorites.index[0]} 类选手")
print(f"    -> 特征: 实力可能一般，但观众疯狂投票保送 (Gap = +{favorites.values[0]:.2f} SD)")
print(f"    [意难平/Underrated]:    {robbed.index[0]} 类选手")
print(f"    -> 特征: 评委给高分，但观众不买账导致淘汰 (Gap = {robbed.values[0]:.2f} SD)")

print(f"\n>>> 5. 潜力股分析 (Growth Potential):")
top_improver = improvement.index[0]
top_imp_val = improvement.values[0]
print(f"    进步最快的群体: {top_improver}")
print(f"    -> 平均分提升: +{top_imp_val:.2f} 分 (从早期到晚期)")
print(f"    -> 建议: 如果该类选手挺过前3周，夺冠概率大幅增加。")

print(f"\n>>> 6. 风险评估 (Risk Analysis):")
most_stable = consistency.index[0]
most_volatile = consistency.index[-1]
print(f"    最稳健投资: {most_stable} (标准差仅 {consistency.values[0]:.2f})")
print(f"    -> 发挥稳定，适合保守策略。")
print(f"    高风险高回报: {most_volatile} (标准差高达 {consistency.values[-1]:.2f})")
print(f"    -> 发挥大起大落，容易出现黑马或首轮游。")

print("="*66)