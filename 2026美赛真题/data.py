import pandas as pd     # 导入数据分析库 pandas，并给它起个别名叫 pd（行业惯例）
import requests         # 导入请求库，用于向网站发送网络请求
from io import StringIO # 从 io 库导入 StringIO 工具，用于将文本处理成文件流

# 定义一个函数，所有的核心逻辑都包在这个函数里
# def 是 definition 的缩写，表示定义函数
def export_lakers_2026_stats():
    # 1. 设置目标网址：这是我们要抓取的湖人队 2026 年数据的网页地址
    url = "https://www.basketball-reference.com/teams/LAL/2026.html"
    
    # 2. 设置输出文件名：抓取到的数据最后要保存成这个 Excel 文件
    output_file = "Lakers_2026_Player_Stats.xlsx"

    # 3. 设置请求头（Headers）：这是为了伪装
    # 很多网站如果不设置 User-Agent，会识别出你是爬虫程序并拒绝访问。
    # 这里我们假装自己是一个在 Windows 电脑上使用 Chrome 浏览器的普通用户。
# --- 核心修改开始 ---
    # 我们升级了“伪装包”，加入了更多真实浏览器会发送的信息
    headers = {
        # 1. User-Agent: 告诉服务器我是 Chrome 浏览器（更新了版本号）
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # 2. Accept-Language: 告诉服务器通过英语沟通（脚本通常不带这个，容易露馅）
        "Accept-Language": "en-US,en;q=0.9",
        # 3. Referer: 告诉服务器我是从 Google 搜索点进来的（这招很管用）
        "Referer": "https://www.google.com/",
        # 4. Accept-Encoding: 告诉服务器我支持压缩格式
        "Accept-Encoding": "gzip, deflate, br",
        # 5. Connection: 保持连接
        "Connection": "keep-alive",
        # 6. Upgrade-Insecure-Requests: 表示我希望用加密连接
        "Upgrade-Insecure-Requests": "1"
    }
    # --- 核心修改结束 ---

    # try...except 是 Python 的错误处理机制
    # 意思是：尝试执行 try 下面的代码，如果出错了，就跳到 except 那里去处理，不要直接让程序崩溃。
    try:
        print(f"正在从 {url} 获取数据...") # 在屏幕上打印提示信息，f代表格式化字符串，可以把变量url放进去
        
        # 4. 发送请求：让 requests 带着我们要去的 url 和伪装头(headers)去下载网页
        response = requests.get(url, headers=headers, timeout=10)
        
        # 检查状态码：如果网页返回 404（找不到）或 500（服务器错误），这行代码会直接报错，停止运行
        response.raise_for_status()

        # 5. 解析网页表格
        # response.text 是网页的源代码（HTML）
        # StringIO(response.text) 把这些源代码包装成一个“内存里的文件”，因为 pandas 喜欢读文件
        # pd.read_html() 会自动扫描 HTML，找到所有的 <table> 标签，并把它们变成一个列表 (list)
        dfs = pd.read_html(StringIO(response.text))

        # 初始化一个变量 target_df，用来存放我们要找的那个具体表格，先设为 None (空)
        target_df = None
        
        # 6. 寻找目标表格
        # 网页里可能有很多表格（比如薪资表、赛程表等）。我们需要遍历刚才找到的所有表格 (dfs)
        for df in dfs:
            # 判断条件：如果这个表格的列名里同时包含 'PTS' (得分) 和 'Player' (球员)
            # 那么我们就认为这是我们要找的“球员数据表”
            if 'PTS' in df.columns and 'Player' in df.columns:
                target_df = df # 把这个表格赋值给 target_df
                break          # 找到了就立刻停止循环 (break)，不用再看后面的表格了
        
        # 7. 处理数据（如果找到了表格）
        if target_df is not None:
            # 数据清洗 A：有些网页表格每隔几行会重复一次表头（Player, PTS...），我们需要把这些重复行删掉
            # 意思是：只保留 'Player' 这一列的值不等于 "Player" 的那些行
            target_df = target_df[target_df['Player'] != 'Player']
            
            # 数据清洗 B：准备处理数字
            # 定义不需要转换成数字的列（球员名、位置、球队代码）
            non_numeric_cols = ['Player', 'Pos', 'Tm']
            
            # 筛选出需要转换的列：即“当前表格所有的列” 减去 “上面定义的非数值列”
            # 这是一个列表推导式写法
            cols_to_convert = [c for c in target_df.columns if c not in non_numeric_cols]

            # 循环处理每一列，把它们转成数字
            for col in cols_to_convert:
                # pd.to_numeric：尝试把文本转成数字
                # errors='ignore'：如果某一格真的转不了（比如是空的），就忽略它，不要报错
                target_df[col] = pd.to_numeric(target_df[col], errors='ignore')

            # 8. 导出结果
            # to_excel：把整理好的 pandas 表格保存为 Excel 文件
            # index=False：不要把 pandas 自带的行号（0, 1, 2...）写进 Excel 里，保持整洁
            target_df.to_excel(output_file, index=False)
            
            # 打印成功信息
            print(f"成功！数据已保存为: {output_file}")
            print(f"共导出 {len(target_df)} 名球员的数据。") # len() 计算有多少行数据
            
        else:
            # 如果循环结束了 target_df 还是 None，说明没找到想要的表格
            print("未找到包含球员数据的表格。")

    # 这里的 e 代表具体的错误信息
    except Exception as e:
        print(f"发生错误: {e}")

# 这是 Python 的程序入口判断
# 如果你直接运行这个文件，__name__ 就等于 "__main__"，下面的代码就会执行
# 如果这个文件被别的程序 import 引用了，下面的代码就不会执行
if __name__ == "__main__":
    export_lakers_2026_stats() # 调用上面定义的函数，开始干活