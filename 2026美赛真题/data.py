import pandas as pd
import requests
from io import StringIO  # 新增：用于解决 FutureWarning 警告

def export_lakers_2026_stats():
    url = "https://www.basketball-reference.com/teams/LAL/2026.html"
    output_file = "Lakers_2026_Player_Stats.xlsx"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        print(f"正在从 {url} 获取数据...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 修复 1: 使用 StringIO 包装 response.text 以消除警告
        dfs = pd.read_html(StringIO(response.text))

        target_df = None
        # 寻找包含 'PTS' 和 'Player' 的主表格
        for df in dfs:
            if 'PTS' in df.columns and 'Player' in df.columns:
                target_df = df
                break
        
        if target_df is not None:
            # 数据清洗：移除表头重复行
            target_df = target_df[target_df['Player'] != 'Player']
            
            # 修复 2: 安全地处理非数值列
            # 球队页通常只有 Player 和 Pos，没有 Tm。我们只移除存在的列。
            non_numeric_cols = ['Player', 'Pos', 'Tm']
            # 过滤出当前表格中实际存在的列进行移除，避免 KeyError
            cols_to_convert = [c for c in target_df.columns if c not in non_numeric_cols]

            # 将数值列转换为数字类型
            for col in cols_to_convert:
                target_df[col] = pd.to_numeric(target_df[col], errors='ignore')

            # 导出到 Excel
            target_df.to_excel(output_file, index=False)
            print(f"成功！数据已保存为: {output_file}")
            print(f"共导出 {len(target_df)} 名球员的数据。")
            
        else:
            print("未找到包含球员数据的表格。")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    export_lakers_2026_stats()