Matlab集成了很多的函数和工具箱,所以占用内存很大

数据分析的两个角度: 描述性分析和推断性分析

**MATLAB 速成课的第一节 (Lesson 1)** 内容非常固定，主要涵盖 **界面认识、变量操作、矩阵基础和简单的逻辑** 。

---

# 🚀 MATLAB Crash Course - Level 1: Pilot Training

# MATLAB 速成课第一章：飞行员入门

> **Core Concept (核心理念):**
>
> MATLAB = **Mat**rix **Lab**oratory (矩阵实验室).
>
> In MATLAB, *everything* is a Matrix. Even a single number like `1` is a **$1 \times 1$** matrix.
>
> 在 MATLAB 的世界里， **万物皆矩阵** 。哪怕只是一个数字 `1`，系统也把它看作是 1行1列的矩阵。

---

## 1. The Cockpit: Interface Overview

### 驾驶舱：界面概览

想象你坐在飞机的驾驶舱里，MATLAB 的界面通常分为四个主要区域：

| **Zone (区域)** | **English Name**   | **Analogy (形象比喻)**                  | **Function (功能)**                                                                            |
| --------------------- | ------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **中间/上方**   | **Command Window** | **Calculator / Chatbot**(计算器/聊天框) | 你发号施令的地方。输入代码，按回车，它立刻执行。适合测试短代码。                                     |
| **左侧**        | **Current Folder** | **File Explorer**(资源管理器)           | 你的“当前工作台”。MATLAB 只能看到这个文件夹里的文件。                                              |
| **右侧**        | **Workspace**      | **Backpack / Memory**(背包/内存)        | 你的“记忆库”。你定义的所有变量（x, A, ans）都存在这里。**如果不在这里，MATLAB 就不认识它。** |
| **中间/上方**   | **Editor**         | **Script Writer**(剧本/代码编辑器)      | 编写长篇代码（`.m`文件）的地方。就像写文章，写完后点 "Run" 统一执行。                              |

---

## 2. Basic Grammar & "The Magic Semicolon"

### 基础语法与“神奇的分号”

在 MATLAB 中，有一个符号决定了它的“性格”： **分号 (`;`)** 。

* **Without Semicolon (不加分号):** MATLAB is talkative. It calculates and **shouts** the result back to you.
  * *MATLAB 很话痨，算出结果后会立刻打印在屏幕上。*
* **With Semicolon (加分号):** MATLAB is in "Stealth Mode". It calculates, saves the result to memory, but  **keeps quiet** .
  * *MATLAB 开启“静音模式”。它会在后台默默算好并记住，但不刷屏。*

**Example (举例):**

**Matlab**

```
a = 1 + 1   % Output: a = 2 (Screen shows result)
b = 5 * 5;  % Output: (Nothing on screen, but 'b' is now 25 in Workspace)
```

---

## 3. The Heart: Creating Matrices

### 心脏：矩阵的创建

既然万物皆矩阵，第一节课最重要的就是学会“造砖”。

#### A. Manual Entry (手动输入)

* **Rule:** Brackets `[]` are the walls. Space/Comma separates columns. Semicolon `;` separates rows.
* **口诀：** “中括号是墙，空格隔开列，分号隔开行。”

**Matlab**

```
% A Row Vector (行向量): 1 row, 3 columns
row_vec = [1 2 3]; 

% A Column Vector (列向量): 3 rows, 1 column
col_vec = [1; 2; 3];

% A Matrix (矩阵): 2 rows, 3 columns
A = [1 2 3; 4 5 6];
% Visualizes as:
% 1  2  3
% 4  5  6
```

#### B. Quick Generators (快速生成器)

* `zeros(m, n)`: Creates a matrix full of zeros. (一张白纸)
* `ones(m, n)`: Creates a matrix full of ones. (全1矩阵)
* `eye(n)`: Identity matrix. (单位矩阵，对角线是1，其他是0。谐音 "I")
* `rand(m, n)`: Random numbers between 0 and 1. (撒骰子，生成0-1之间的随机小数)

---

## 4. The Operator: The Dot `.`

### 运算符号：那个“点”很重要

这是新手最容易报错的地方！区分 **Linear Algebra Math (线性代数运算)** 和  **Element-wise Math (点对点运算)** 。

#### A. Matrix Multiplication (`*`)

* **Concept:** Standard math rule. (Row **$\times$** Column). Requires inner dimensions to match.
* **概念:** 也就是大学线代课学的矩阵乘法，要求 **前一个矩阵的列数 = 后一个矩阵的行数** 。

**Matlab**

```
C = A * B; % Matrix multiplication
```

#### B. Element-wise Multiplication (`.*`)

* **Concept:** "You multiply your neighbor". Matrices must have the  **exact same shape** .
* **概念:** “点乘”。对应位置的元素直接相乘。要求两个矩阵长宽完全一样。
* **Analogy:** Imagine two egg cartons. You multiply the egg in the top-left slot of carton A with the egg in the top-left slot of carton B.

**Matlab**

```
C = A .* B; % Element-wise multiplication
% Also applies to division (./) and power (.^)
y = x.^2;   % Square every element in x independently! (Very common in plotting)
```

---

## 5. The GPS: Indexing / Slicing

### 定位系统：索引与切片

**⚠️ WARNING:** MATLAB starts counting at  **1** , not 0!

**⚠️ 警告:** MATLAB 的世界里，第一层楼是 1 楼，没有 0 楼！(这一点和 Python/C 不同)

假设我们有一个矩阵 `M`:

$$
M = \begin{bmatrix} 10 & 20 & 30 \\ 40 & 50 & 60 \\ 70 & 80 & 90 \end{bmatrix}
$$

| **Command** | **Meaning (含义)**                    | **Result (结果)** |
| ----------------- | ------------------------------------------- | ----------------------- |
| `M(1, 2)`       | Row 1, Column 2 (第1行第2列)                | `20`                  |
| `M(2, :)`       | Row 2,**All**Columns (第2行，所有列)  | `[40 50 60]`          |
| `M(:, 3)`       | **All**Rows, Column 3 (所有行，第3列) | `[30; 60; 90]`        |
| `M(end, end)`   | The very last element (最后一个元素)        | `90`                  |
| `M(1:2, 1:2)`   | Rows 1 to 2, Cols 1 to 2 (切左上角那一块)   | `[10 20; 40 50]`      |

---

## 6. Practical Toolbox: Essential Functions

### 实用工具箱：第一节课必会的函数

| **Function** | **Description**                                                        | **Example**           |
| ------------------ | ---------------------------------------------------------------------------- | --------------------------- |
| `clc`            | **Clear Command Window** . Wipes the text off the screen (Clean desk). | `clc`(清屏，不删变量)     |
| `clear`          | **Clear Workspace** . Deletes all variables from memory (Brain wipe).  | `clear`(清空内存，慎用！) |
| `size(A)`        | Returns dimensions of matrix A.                                              | `[r, c] = size(A);`       |
| `length(v)`      | Returns the length of the longest dimension (for vectors).                   | `len = length(vec);`      |
| `sum(A)`         | Sums elements. (Usually sums columns by default).                            | `total = sum(v);`         |
| `disp()`         | Displays text or value cleanly.                                              | `disp('Hello World');`    |

---

## 7. Plotting: The Artist (Visuals)

### 绘图：初级画师

MATLAB 最强大的功能之一就是画图。

**Matlab**

```
x = 0 : 0.01 : 2*pi;  % Create a vector from 0 to 2pi, step 0.01
y = sin(x);           % Calculate sine for every x

plot(x, y);           % The basic drawing command
title('My First Plot'); % Give it a name
xlabel('Time');       % Label the floor
ylabel('Amplitude');  % Label the wall
grid on;              % Turn on the grid lines (开启网格)
```

---

### 💡 Pro Tip for Lesson 1 (第一课的小贴士)

* **Help is everywhere:** If you forget how to use a function (e.g., `sum`), just type `doc sum` or `help sum` in the Command Window. It’s like asking Siri.
  * *忘记函数怎么用了？直接在命令窗口输入 `doc sum`，官方文档是最好的老师。*
* **Variable Names:** Can contain letters, numbers, underscores. **Must start with a letter.** Case sensitive (`a` is not `A`).
  * *变量名区分大小写，且不能以数字开头。*
