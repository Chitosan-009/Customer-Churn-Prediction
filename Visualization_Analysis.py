import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from Data_Preprocessing import create_bin_features,load_data,data_quality_check

def analyze_international_plan_churn(train_df):# 国际套餐与流失率关系
    # 数据验证
    print("数据验证")
    print(f"国际套餐取值: {train_df['international_plan'].unique()}")
    print(f"流失状态取值: {train_df['churn'].unique()}")

    # 检查必要列
    required_columns = ['international_plan', 'churn']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要列: {missing_columns}")

    #将 churn 转为数值型：yes=1, no=0
    train_df['churn_num'] = train_df['churn'].str.lower().map({"yes": 1, "no": 0}).astype(int)

    # 国际套餐与流失率统计分析
    intl_churn = train_df.groupby('international_plan')['churn_num'].agg([('客户总数','count'),('流失客户数','sum'),('流失率','mean')]).round(3)
    intl_churn['非流失客户数'] = intl_churn['客户总数'] - intl_churn['流失客户数']
    intl_churn['流失占比'] = (intl_churn['流失客户数'] / intl_churn['流失客户数'].sum()).round(3)

    print("统计差异分析:")
    no_plan_data = train_df[train_df['international_plan'] == 'no']['churn_num']
    yes_plan_data = train_df[train_df['international_plan'] == 'yes']['churn_num']

    churn_rate_no = no_plan_data.mean()
    churn_rate_yes = yes_plan_data.mean()

    print(f"无国际套餐客户流失率: {churn_rate_no:.3f} ({churn_rate_no * 100:.1f}%)")
    print(f"有国际套餐客户流失率: {churn_rate_yes:.3f} ({churn_rate_yes * 100:.1f}%)")
    print(f"流失率绝对差异: {churn_rate_yes - churn_rate_no:.3f}")
    print(f"流失率相对差异: {(churn_rate_yes - churn_rate_no) / churn_rate_no * 100:.1f}%")
    return train_df, churn_rate_no, churn_rate_yes

def plot_international_plan_churn(train_df, churn_rate_no, churn_rate_yes): # 可视化国际套餐与流失率统计分析
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5))
    fig.suptitle('国际套餐与客户流失深度分析', fontsize=9, fontweight='bold')
    # 流失率对比柱状图
    ax1 = axes[0, 0]
    bar_plot = sns.barplot(
        x='international_plan',
        y='churn_num',
        hue='international_plan',
        data=train_df,
        palette=['#66c2a5', '#fc8d62'],
        alpha=0.8,
        ax=ax1,
        legend=False
    )
    ax1.set_title('国际套餐与客户流失率关系', fontsize=9, fontweight='bold')
    ax1.set_xlabel('是否开通国际套餐', fontsize=9)
    ax1.set_ylabel('流失率', fontsize=9)
    ax1.tick_params(axis='both', labelsize=9)  # 坐标轴刻度标签
    # 柱状图上标注流失率
    for i, p in enumerate(bar_plot.patches):
        height = p.get_height()
        ax1.annotate(
            f'{height:.1%}',
            (p.get_x() + p.get_width() / 2., height + 0.01),
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    # 客户数量分布饼图
    ax2 = axes[0, 1]
    plan_counts = train_df['international_plan'].value_counts()
    colors = ['#66c2a5', '#fc8d62']
    wedges, texts, autotexts = ax2.pie(
        plan_counts.values,
        labels=plan_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 9}  # 饼图内部文字
    )
    ax2.set_title('国际套餐客户分布', fontsize=9, fontweight='bold')

    # 流失客户构成堆叠图
    ax3 = axes[1, 0]
    churn_composition = pd.crosstab(train_df['international_plan'], train_df['churn'])
    churn_composition.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=ax3)
    ax3.set_title('国际套餐客户流失构成', fontsize=9, fontweight='bold')
    ax3.set_xlabel('是否开通国际套餐', fontsize=9)
    ax3.set_ylabel('客户数量', fontsize=9)
    ax3.legend(title='流失状态', loc='upper right', fontsize=9)  # 图例
    ax3.tick_params(axis='x', rotation=0, labelsize=9)  # x轴刻度
    ax3.tick_params(axis='y', labelsize=9)  # y轴刻度
    # 堆叠图上标注客户数量
    for i, (idx, row) in enumerate(churn_composition.iterrows()):
        total = row.sum()
        no_churn = row['no']
        yes_churn = row['yes']
        ax3.text(
            i, no_churn / 2, f'{no_churn}',
            ha='center', va='center', fontweight='bold', fontsize=9
        )
        ax3.text(
            i, no_churn + yes_churn / 2, f'{yes_churn}',
            ha='center', va='center', fontweight='bold', color='white', fontsize=9
        )

    # 流失率对比点图
    ax4 = axes[1, 1]
    plan_labels = ['无国际套餐', '有国际套餐']
    churn_rates = [churn_rate_no, churn_rate_yes]

    for i, (label, rate) in enumerate(zip(plan_labels, churn_rates)):
        ax4.scatter(i, rate, s=200, color=colors[i], alpha=0.7, label=label)
        ax4.text(
            i, rate + 0.02, f'{rate:.1%}',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    ax4.set_xlim(-0.5, 1.5)
    ax4.set_ylim(0, max(churn_rates) + 0.1)
    ax4.set_xticks([])
    ax4.set_ylabel('流失率', fontsize=9)
    ax4.set_title('流失率对比', fontsize=9, fontweight='bold')
    ax4.legend(fontsize=9)  # 图例
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelsize=9)  # y轴刻度

    # 标注流失率差异
    ax4.plot([0, 1], churn_rates, 'k--', alpha=0.5)
    ax4.text(
        0.5, (churn_rates[0] + churn_rates[1]) / 2,
        f'差异: {(churn_rates[1] - churn_rates[0]):.1%}',
        ha='center', va='bottom', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

    return train_df

def analyze_number_customer_service_calls(train_df): # 客服呼叫次数与客户流失率的关系
    # 统计：按客服呼叫次数分组，计算流失率
    service_churn = train_df.groupby('number_customer_service_calls')['churn_num'].agg(['count', 'sum', 'mean'])
    service_churn.columns = ['客户总数', '流失客户数', '流失率']
    service_churn['流失率'] = service_churn['流失率'].round(3)
    print("\n客服呼叫次数与流失率关联：")
    print(service_churn)

    # 可视化客服呼叫次数与流失率关联
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=False)

    ax1.plot(service_churn.index, service_churn['流失率'], marker='o', linewidth=2, color='#e74c3c')
    ax1.set_ylabel('流失率', fontsize=8)  # 轴标签字体
    ax1.set_title('流失率随呼叫次数的趋势', fontsize=10)  # 标题字体
    ax1.tick_params(axis='both', labelsize=8)  # 刻度字体
    ax1.grid(alpha=0.3)

    # 标注高风险点
    high_risk_calls = service_churn[service_churn['流失率'] > 0.3].index
    for call_num in high_risk_calls:
        ax1.annotate(
            f'次数{call_num}\n流失率{service_churn["流失率"].loc[call_num]:.1%}',
            (call_num, service_churn['流失率'].loc[call_num]),
            xytext=(5, 10),
            textcoords='offset points',
            fontsize=8  # 标注字体
        )

    ax2.bar(service_churn.index, service_churn['客户总数'], alpha=0.6, color='#3498db')
    ax2.set_xlabel('客服呼叫次数', fontsize=8)
    ax2.set_ylabel('客户总数', fontsize=8)
    ax2.set_title('各呼叫次数的客户数量分布', fontsize=10)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return train_df

def analyze_call_duration(train_df): # 通话时长分析
    # 选择通话时长字段
    duration_cols = ['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes']

    # 通话时长基本统计量
    print("通话时长字段基本统计：")
    print(train_df[duration_cols].describe())

    # 各时段通话时长分布
    plt.figure(figsize=(20, 4))
    for i, col in enumerate(duration_cols):
        plt.subplot(1, 4, i + 1)
        sns.histplot(train_df[col], kde=True, color='m', bins=30)
        plt.title(f'{col} 分布', fontsize=12)
        plt.xlabel('时长（分钟）', fontsize=10)
        plt.ylabel('客户数', fontsize=10)
    plt.tight_layout()
    plt.show()

    # 通话时长与流失率的关系（分箱分析）
    duration_bin_rules = {
        # 日间时长：四分位分箱（0-143.3→143.3-180.5→180.5-216.2→216.2+）
        'total_day_minutes': pd.qcut(
            train_df['total_day_minutes'],
            q=4,
            labels=['day时长_低(0-143)', 'day时长_中低(143-180)', 'day时长_中高(180-216)', 'day时长_高(216+)'],
            duplicates='drop'
        ),
        # 晚间时长：三分位分箱（0-165.9→165.9-233.8→233.8+）
        'total_eve_minutes': pd.qcut(
            train_df['total_eve_minutes'],
            q=3,
            labels=['eve时长_低(0-166)', 'eve时长_中(166-234)', 'eve时长_高(234+)'],
            duplicates='drop'
        ),
        # 夜间时长：自定义分箱（基于25%/75%分位数，0-167.2→167.2-234.7→234.7+）
        'total_night_minutes': pd.cut(
            train_df['total_night_minutes'],
            bins=[0, 167.2, 234.7, float('inf')],
            labels=['night时长_低(0-167)', 'night时长_中(167-235)', 'night时长_高(235+)']
        ),
        # 国际时长：四分位分箱（0-8.5→8.5-10.3→10.3-12.0→12.0+），捕捉高国际通话用户
        'total_intl_minutes': pd.qcut(
            train_df['total_intl_minutes'],
            q=4,
            labels=['intl时长_低(0-8.5)', 'intl时长_中低(8.5-10.3)', 'intl时长_中高(10.3-12.0)', 'intl时长_高(12.0+)'],
            duplicates='drop'
        )
    }
    # 添加分箱列（原字段名+"_bin"）
    for col, bin_series in duration_bin_rules.items():
        train_df[f'{col}_bin'] = bin_series

    print("\n分箱后各组样本量：")
    for col in duration_cols:
        bin_col = f'{col}_bin'
        print(f"\n{col} 分箱样本量：")
        print(train_df[bin_col].value_counts().sort_index())

    # 计算各分箱的流失率
    duration_churn_result = {}
    for col in duration_cols:
        bin_col = f'{col}_bin'
        # 计算每组客户数、流失数、流失率
        churn_stats = train_df.groupby(bin_col, observed=True)['churn_num'].agg(
            客户数='count',
            流失数='sum',
            流失率=lambda x: x.mean().round(2)  # 保留4位小数
        ).reset_index()
        duration_churn_result[col] = churn_stats
        print(f"\n{col} 分箱流失率统计：")
        print(churn_stats)
    # 可视化分箱与流失率关系
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()  # 转为一维数组，方便循环

    for i, col in enumerate(duration_cols):
        bin_col = f'{col}_bin'
        stats = duration_churn_result[col]

        # 绘制流失率柱状图
        sns.barplot(
            x=bin_col,
            y='流失率',
            data=stats,
            hue=bin_col,
            palette='Greens',
            legend=False,
            ax=axes[i]
        )
        axes[i].set_title(f'{col} 分箱与流失率关系', fontsize=12)
        axes[i].set_xlabel('通话时长分箱（分钟）', fontsize=10)
        axes[i].set_ylabel('流失率', fontsize=10)
        axes[i].tick_params(axis='x', rotation=15, labelsize=9)
        axes[i].set_ylim(0, max(stats['流失率']) + 0.1)  # 预留标注空间

        # 标注流失率和样本量
        for j, row in stats.iterrows():
            axes[i].annotate(
                f'流失率：{row["流失率"]:.1%}\n样本：{row["客户数"]}',
                xy=(j, row["流失率"]),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
            )

    plt.tight_layout()
    plt.show()

    return train_df

def analyze_call_charge(train_df): # 对通话费用进行分析
   # 选择费用字段
    charge_cols = ['total_day_charge','total_eve_charge','total_night_charge','total_intl_charge']
    print(" 费用字段基本统计：")
    print(train_df[charge_cols].describe())

    # 可视化各时段费用分布
    plt.figure(figsize=(20, 4))
    for col in charge_cols:
        sns.kdeplot(train_df[col], label=col)
    plt.title('各项费用分布对比')
    plt.xlabel('费用（元）')
    plt.ylabel('密度')
    plt.legend()

    # 查看四个费用字段的直方图
    plt.figure(figsize=(20, 4))
    for i, col in enumerate(charge_cols):
        plt.subplot(1, 4, i + 1)
        sns.histplot(train_df[col], kde=True, color='steelblue', bins=30)
        plt.title(f'{col} 分布', fontsize=12)
        plt.xlabel('费用（元）', fontsize=10)
        plt.ylabel('客户数', fontsize=10)

    plt.tight_layout()
    plt.show()

    # 费用字段间的相关性分析（热力图）
    print("\n 费用字段相关性分析：")
    charge_corr = train_df[charge_cols].corr()

    plt.figure(figsize=(10, 4))
    mask = np.triu(np.ones_like(charge_corr, dtype=bool))  # 隐藏上三角
    sns.heatmap(
        charge_corr,
        mask=mask,
        annot=True,  # 显示相关系数值
        cmap='coolwarm',
        vmin=-1, vmax=1,
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('各时段费用相关性热力图')
    plt.tight_layout()
    plt.show()

    # 通话费用分箱
    # 分布定分箱，右偏用 qcut，集中自定义，减少冗余分组。
    bin_rules = {
        # day_charge右偏→qcut4组；标签含具体区间，方便后续解读
        'total_day_charge': pd.qcut(train_df['total_day_charge'], q=4,
                                    labels=['day_低(0-17)', 'day_中低(17-26)', 'day_中高(26-36)', 'day_高(36+)']),
        # eve_charge偏态平缓→qcut3组
        'total_eve_charge': pd.qcut(train_df['total_eve_charge'], q=3,
                                    labels=['eve_低(0-11)', 'eve_中(11-17)', 'eve_高(17+)']),
        # night_charge集中→自定义3组（按中位数拆分）
        'total_night_charge': pd.cut(train_df['total_night_charge'], bins=[0, 8, 11, 20],
                                     labels=['night_低(0-8)', 'night_中(8-11)', 'night_高(11+)']),
        # intl_charge右偏→qcut4组
        'total_intl_charge': pd.qcut(train_df['total_intl_charge'], q=4,
                                     labels=['intl_低(0-2.7)', 'intl_中低(2.7-4.0)', 'intl_中高(4.0-5.4)',
                                             'intl_高(5.4+)'])
    }
    # 批量添加分箱列（列名：原字段名+"_bin"）
    for col, bin_series in bin_rules.items():
        train_df[f'{col}_bin'] = bin_series

    # 查看分箱后每组的样本量（验证是否均匀）
    print("分箱后各组样本量：")
    for col in charge_cols:
        bin_col = f'{col}_bin'
        print(f"\n{col} 分箱样本量：")
        print(train_df[bin_col].value_counts().sort_index())

    # 批量计算每个分箱的流失率
    charge_churn_result = {}

    for col in charge_cols:
        bin_col = f'{col}_bin'
        churn_stats = train_df.groupby(bin_col, observed=True)['churn_num'].agg(
            客户数='count',
            流失数='sum',
            流失率='mean'
        ).round({'流失率': 3})
        churn_stats['未流失数'] = churn_stats['客户数'] - churn_stats['流失数']
        churn_stats['流失率'] = (churn_stats['流失数'] / churn_stats['客户数']).round(4)
        churn_stats = churn_stats[['客户数', '未流失数', '流失数', '流失率']]
        charge_churn_result[col] = churn_stats

        print(churn_stats)

        # 卡方检验
        contingency = pd.crosstab(train_df[bin_col], train_df['churn_num'])
        chi2, p, _, _ = chi2_contingency(contingency)
        significant = "显著" if p < 0.05 else "不显著"
        print(f"{col}：卡方值={chi2:.2f}，p值={p:.4f}（{significant}）")
        print(f"结论：{col}的费用水平与客户流失存在{significant}关联\n")

    # 可视化分箱结果
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    axes = axes.flatten()

    for i, col in enumerate(charge_cols):
        bin_col = f'{col}_bin'
        stats = charge_churn_result[col]
        sns.barplot(x=bin_col, y='churn_num', hue=bin_col, data=train_df,
                    ax=axes[i], palette='Set2', legend=False)
        axes[i].set_title(f'{col} 分箱与流失率关系', fontsize=12)
        axes[i].set_xlabel('费用分箱', fontsize=10)
        axes[i].set_ylabel('流失率', fontsize=10)
        axes[i].tick_params(axis='x', rotation=15)

        # 标注流失率和样本量
        for j, (idx, row) in enumerate(stats.iterrows()):
            axes[i].annotate(
                f'流失率：{row["流失率"]:.1%}\n样本：{row["客户数"]}',
                xy=(j, row["流失率"]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=8
            )

    plt.tight_layout()
    plt.show()

    # 计算总费用
    train_df["total_charge"] = train_df[charge_cols].sum(axis=1)

    # 查看总费用分布与流失，量化流失客户与非流失客户的总费用差异
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(
        data=train_df,
        x='total_charge',
        hue='churn',
        multiple='stack',
        kde=True,
        ax=axes[0]  # 指定子图
    )
    axes[0].set_title('总费用分布与流失关系', fontsize=14, pad=15)
    axes[0].set_xlabel('总费用（元）', fontsize=12)
    axes[0].set_ylabel('客户数', fontsize=12)
    axes[0].tick_params(labelsize=10)

    # 流失与非流失客户的总费用箱线图 ，用统计量量化流失客户与非流失客户的总费用差异。
    colors = ['#4E79A7', '#E15759']  # 非流失-蓝，流失-红
    sns.boxplot(
        x='churn',
        y='total_charge',
        hue='churn',
        data=train_df,
        palette=colors,
        width=0.6,
        linewidth=2,
        fliersize=5,
        boxprops=dict(alpha=0.8),
        medianprops=dict(color='#F28E2C', linewidth=3),
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title('流失与非流失客户的总费用分布对比', fontsize=14, pad=15)
    axes[1].set_xlabel('是否流失', fontsize=12)
    axes[1].set_ylabel('总费用（元）', fontsize=12)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['非流失', '流失'], fontsize=10)
    axes[1].tick_params(axis='y', labelsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.3, color='gray')

    # 箱线图标注中位数
    for i, churn_status in enumerate(['no', 'yes']):
        median = train_df[train_df['churn'] == churn_status]['total_charge'].median()
        axes[1].annotate(
            f'中位数：{median:.1f}元',
            xy=(i, median),
            xytext=(0, 30),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#999', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='#666')
        )

    plt.tight_layout()
    plt.show()

    # 对总费用进行分箱并计算流失率
    train_df['total_charge_bin'] = pd.qcut(
        train_df['total_charge'],
        q=4,
        labels=['总费用_极低', '总费用_中低', '总费用_中高', '总费用_极高']
    )

    # 计算各分箱的流失率
    total_charge_churn = train_df.groupby('total_charge_bin', observed=True)['churn_num'].agg(
        客户数='count',
        流失率='mean'
    ).round(5)
    total_charge_churn['流失率'] = (total_charge_churn['流失率'] * 100).round(2)  # 转为百分比

    # 可视化总费用分箱与流失率的关系
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x='total_charge_bin',
        y='流失率',
        hue='total_charge_bin',
        data=total_charge_churn.reset_index(),
        palette='Set3',
        legend=False
    )
    plt.title('总费用分箱与流失率关系（验证U型模式）')
    plt.xlabel('总费用分箱')
    plt.ylabel('流失率（%）')

    # 标注流失率和客户数
    for i, (_, row) in enumerate(total_charge_churn.iterrows()):
        plt.annotate(
            f'流失率：{row["流失率"]:.2f}%\n客户数：{int(row["客户数"])}',
            (i, row["流失率"]),
            ha='center',
            va='bottom',
            fontsize=9
        )
    plt.tight_layout()
    plt.show()

    # 总费用分位数阈值
    low_threshold = train_df['total_charge'].quantile(0.25)
    high_threshold = train_df['total_charge'].quantile(0.75)
    print(f"低费用阈值（25%分位数）：{low_threshold:.2f}元")
    print(f"高费用阈值（75%分位数）：{high_threshold:.2f}元")

    # 定义客户类型
    train_df['charge_level'] = pd.cut(
        train_df['total_charge'],
        bins=[-np.inf, low_threshold, high_threshold, np.inf],
        labels=['低费用客户', '中等费用客户', '高费用客户']
    )

    # 计算数量和比例
    order = ['低费用客户', '中等费用客户', '高费用客户']
    charge_counts = train_df['charge_level'].value_counts().reindex(order)
    charge_ratio = (charge_counts / len(train_df) * 100).round(2)

    # 各类型客户数量及比例
    for level in order:
        print(f"{level}：{charge_counts[level]}人，占比{charge_ratio[level]}%")

    # 可视化
    plt.figure(figsize=(10, 4))
    ax = sns.countplot(
        x='charge_level',
        hue='charge_level',  # 与x变量相同
        data=train_df,
        order=order,
        palette=['#66c2a5', '#8da0cb', '#fc8d62'],
        legend=False  # 关闭图例（避免重复）
    )
    plt.title('高低费用客户分布')
    plt.xlabel('客户类型')
    plt.ylabel('客户数量')

    # 标注比例
    for i, level in enumerate(order):
        count = charge_counts[level]
        ax.text(i, count + 5, f'{charge_ratio[level]}%', ha='center', color='darkred')

    plt.tight_layout()
    plt.show()

    return train_df

def analyze_call_frequency(train_df): # 通话次数分析
    # 统计通话次数
    call_cols = ['total_day_calls', 'total_eve_calls', 'total_night_calls', 'total_intl_calls']
    print("各时段通话次数统计描述：")
    print(train_df[call_cols].describe().round(2))

    # 通话次数分布可视化
    plt.figure(figsize=(12, 5))
    for i, col in enumerate(call_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(train_df[col], kde=True, bins=30)
        plt.title(f'{col}的分布')
        plt.xlabel('通话次数')

    plt.tight_layout()
    plt.show()

    # 计算每个时段通话次数的流失率（按原始次数分组）
    def calculate_call_churn(df, col):
        """按原始通话次数分组，计算流失率和样本量"""
        call_churn = df.groupby(col).agg(
            总客户数=('churn_num', 'count'),
            流失客户数=('churn_num', 'sum')
        ).reset_index()
        call_churn['流失率(%)'] = (call_churn['流失客户数'] / call_churn['总客户数'] * 100).round(2)
        # 过滤样本量太少的组（避免极端值干扰，如样本量<10）
        call_churn = call_churn[call_churn['总客户数'] >= 10].sort_values(by=col)
        return call_churn

    # 计算4个时段的流失率
    call_churn_dict = {}
    for col in call_cols:
        call_churn_dict[col] = calculate_call_churn(train_df, col)
        print(f"{col} 流失率:")
        print(call_churn_dict[col].head(5))

    # 可视化通话次数与流失率的变化趋势
    plt.figure(figsize=(16, 8))
    for i, col in enumerate(call_cols, 1):
        plt.subplot(2, 2, i)
        data = call_churn_dict[col]

        # 绘制流失率折线图
        sns.lineplot(x=col, y='流失率(%)', data=data, marker='o', color='grey', linewidth=2)
        # 辅助：用橙色点表示样本量大小（点越大样本量越大）
        plt.scatter(
            x=data[col],
            y=data['流失率(%)'],
            s=data['总客户数'] / 2,  # 样本量缩放后作为点大小
            alpha=0.3,
            color='Orange'
        )

        plt.title(f'{col} 通话次数与流失率', fontsize=12)
        plt.xlabel('通话次数', fontsize=10)
        plt.ylabel('流失率(%)', fontsize=10)
        plt.grid(alpha=0.3)
        plt.title(f'{col} 通话次数与流失率', fontsize=12)
        plt.xlabel('通话次数', fontsize=10)
        plt.ylabel('流失率(%)', fontsize=10)
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 通话次数分箱
    # 1. 日间通话次数（4箱，等频）
    train_df['total_day_calls_bin'] = pd.qcut(
        train_df['total_day_calls'],
        q=4,  # 按四分位数分箱
        labels=['day_call_低', 'day_call_中低', 'day_call_中高', 'day_call_高'],
        duplicates='drop'  # 处理重复分位数（避免报错）
    )
    # 2. 晚间通话次数（4箱，等频）
    train_df['total_eve_calls_bin'] = pd.qcut(
        train_df['total_eve_calls'],
        q=4,
        labels=['eve_call_低', 'eve_call_中低', 'eve_call_中高', 'eve_call_高'],
        duplicates='drop'
    )
    # 3. 夜间通话次数（4箱，等频）
    train_df['total_night_calls_bin'] = pd.qcut(
        train_df['total_night_calls'],
        q=4,
        labels=['night_call_低', 'night_call_中低', 'night_call_中高', 'night_call_高'],
        duplicates='drop'
    )
    # 4. 国际通话次数（4箱，等频）
    train_df['total_intl_calls_bin'] = pd.qcut(
        train_df['total_intl_calls'],
        q=4,
        labels=['intl_call_低', 'intl_call_中低', 'intl_call_中高', 'intl_call_高'],
        duplicates='drop'
    )
    # 定义所有分箱后的字段名
    bin_cols = ['total_day_calls_bin', 'total_eve_calls_bin', 'total_night_calls_bin', 'total_intl_calls_bin']

    # 计算每个分箱的流失率、样本量等统计量
    def get_bin_churn_stats(df, bin_col):
        # 按分箱字段分组，计算核心指标
        stats = df.groupby(bin_col, observed=False)['churn_num'].agg(
            样本量='count',
            流失数='sum',
            流失率=lambda x: round(x.mean() * 100, 2)  # 转为百分比，保留2位小数
        ).reset_index()
        # 按分箱顺序排序（低→高）
        stats = stats.sort_values(by=bin_col)
        return stats

    # 计算分箱字段的流失率
    bin_churn_stats = {}
    for col in bin_cols:
        bin_churn_stats[col] = get_bin_churn_stats(train_df, col)

        print(f"分箱流失率:")
        print(bin_churn_stats[col])

    # 可视化分箱后流失率
    plt.figure(figsize=(12, 6))
    # 循环绘制每个分箱字段的流失率柱状图
    for i, col in enumerate(bin_cols, 1):
        plt.subplot(2, 2, i)  # 定位子图位置
        stats = bin_churn_stats[col]

        # 绘制流失率柱状图
        sns.barplot(
            data=stats,
            x=col,
            y='流失率',
            hue=col,
            palette='coolwarm',  # 颜色渐变（低流失率偏蓝，高流失率偏红）
            edgecolor='black',
            legend=False  # 柱形边框，更清晰
        )
        plt.xlabel('分箱区间', fontsize=10)
        plt.ylabel('流失率（%）', fontsize=10)
        plt.xticks(rotation=30, fontsize=9)  # 旋转标签，避免重叠

        # 标注：在柱形上方显示流失率和样本量
        for idx, row in stats.iterrows():
            plt.text(
                x=idx,  # 柱形x轴位置
                y=row['流失率'] + 0.5,  # 文本y轴位置（柱顶上方）
                s=f"流失率：{row['流失率']}%\n样本：{row['样本量']}",  # 显示内容
                ha='center',  # 水平居中
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='gray', pad=2, boxstyle='round,pad=0.3')  # 文本框美化
            )

    plt.tight_layout()
    plt.show()

    return train_df

def analyze_voice_mail_plan(train_df): # 语音信箱分析
    # 按是否开通语音信箱分组，计算客户总数、流失数、流失率
    vmail_churn = train_df.groupby('voice_mail_plan')['churn_num'].agg(['count', 'sum', 'mean'])
    vmail_churn.columns = ['客户总数', '流失客户数', '流失率']
    vmail_churn['流失率'] = vmail_churn['流失率'].round(3)
    vmail_churn['流失率(%)'] = vmail_churn['流失率'] * 100

    print("按是否开通语音信箱分组的流失率：")
    print(vmail_churn)

    # 按语音邮件数量分箱（0=未用，1-10=低频，11-30=中频，31+=高频）
    train_df['vmail_msg_bin'] = pd.cut(
        train_df['number_vmail_messages'],
        bins=[-1, 0, 10, 30, float('inf')],  # 用inf覆盖所有高频值
        labels=['未使用(0)', '低频(1-10)', '中频(11-30)', '高频(31+)']
    )

    # 按分箱分组，计算流失率
    vmail_msg_churn = train_df.groupby('vmail_msg_bin', observed=True)['churn_num'].agg(['count', 'sum', 'mean'])
    vmail_msg_churn.columns = ['客户总数', '流失客户数', '流失率']
    vmail_msg_churn['流失率'] = vmail_msg_churn['流失率'].round(3)
    vmail_msg_churn['流失率(%)'] = vmail_msg_churn['流失率'] * 100

    print("按语音邮件使用量分箱的流失率:")
    print(vmail_msg_churn)

    return train_df

def analyze_account_length(train_df): # 账户时长分析
    # 账户时长描述性统计
    account_stats = train_df["account_length"].describe()
    print("账户时长（月）的描述性统计：")
    print(account_stats)

    # 月份转年数
    train_df["account_years"] = (train_df["account_length"] / 12).round(2)
    years_stats = train_df["account_years"].describe()
    print("账户时长（年）的描述性统计：")
    print(years_stats)

    # 分析账户时长与流失率关系
    # 定义年数区间
    max_years = train_df["account_years"].max()
    bins = [0, 1, 2, 3, 5, 10, max_years]
    labels = ["0-1年", "1-2年", "2-3年", "3-5年", "5-10年", f"10年以上（≤{max_years}年）"]

    # 新增年数分组列
    train_df["account_years_bin"] = pd.cut(
        train_df["account_years"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # 计算各区间流失率
    churn_by_years = train_df.groupby("account_years_bin", observed=True)["churn_num"].agg(
        流失率=lambda x: round(x.mean() * 100, 2) if not x.isna().all() else 0,
        客户数量=("count")
    ).reset_index()

    print("各账户时长（年）区间的流失率：")
    print(churn_by_years)

    # 可视化用户账户时长分布情况与不同时间区间的流失率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # 1. 账户时长（年）的分布
    sns.histplot(
        data=train_df,
        x="account_years",
        kde=True,
        bins=15,
        color="Orange",
        ax=ax1
    )
    ax1.set_title("客户账户时长分布（年）", fontsize=15)
    ax1.set_xlabel("账户时长（年）", fontsize=12)
    ax1.set_ylabel("客户数量", fontsize=12)

    # 2. 各年数区间的流失率对比
    sns.barplot(
        data=churn_by_years,
        x="account_years_bin",
        y="流失率",
        hue="account_years_bin",
        palette="viridis",
        legend=False,
        ax=ax2
    )

    for i, row in enumerate(churn_by_years.itertuples()):
        ax2.text(i, row.流失率 + 0.5, f"{row.流失率}%", ha="center", fontsize=11)
    ax2.set_title("不同账户时长（年）区间的客户流失率", fontsize=15)
    ax2.set_xlabel("账户时长区间", fontsize=12)
    ax2.set_ylabel("流失率（%）", fontsize=12)
    ax2.tick_params(axis='x', rotation=15)  # 旋转x轴标签
    ax2.set_ylim(0, churn_by_years["流失率"].max() + 5)

    plt.tight_layout()
    plt.show()

    # 流失与未流失客户的年数分布对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    sns.boxplot(
        data=train_df,
        x="churn",
        y="account_years",
        hue="churn",  # 同样，为boxplot添加hue以匹配palette用法（可选，不强制）
        palette="coolwarm",
        legend=False,
        ax=ax1
    )
    ax1.set_title("流失与未流失客户的账户时长（年）分布对比", fontsize=15)
    ax1.set_xlabel("是否流失", fontsize=12)
    ax1.set_ylabel("账户时长（年）", fontsize=12)

    # 年数与流失率的趋势关系
    churn_trend_years = train_df.groupby("account_years")["churn_num"].mean().reset_index()
    churn_trend_years.columns = ["账户时长（年）", "流失率"]

    sns.lineplot(
        data=churn_trend_years,
        x="账户时长（年）",
        y="流失率",
        marker="s",
        markersize=6,
        color="#9b59b6",
        ax=ax2
    )
    ax2.set_title("账户时长（年）与流失率的趋势关系", fontsize=15)
    ax2.set_xlabel("账户时长（年）", fontsize=12)
    ax2.set_ylabel("流失率", fontsize=12)
    ax2.set_ylim(0, 0.5)

    plt.tight_layout()
    plt.show()

    return train_df

def analyze_bivariate_relationships(train_df,
                                    row_col,
                                    col_col,
                                    target_col='churn_num',
                                    include_count=False,
                                    plot_heatmap=True,
                                    figsize=(10, 6),
                                    cmap='YlOrRd'):

    # 输入校验（避免缺少列报错）
    required_cols = [row_col, col_col, target_col]
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列：{', '.join(missing_cols)}")

    # 数据清洗：删除关键列的缺失值
    train_df_clean = train_df.dropna(subset=required_cols).copy()

    # 核心统计逻辑（仅计算一次）
    if include_count:
        cross_stats = train_df_clean.groupby([row_col, col_col], observed=True)[target_col].agg(
            总用户数='count',
            流失率=lambda x: round(x.mean() * 100, 1) if x.notna().sum() > 0 else 0.0
        ).reset_index()
        # 热力图透视表
        pivot_df = cross_stats.pivot_table(
            index=row_col, columns=col_col, values='流失率', observed=True, fill_value=0.0
        )
    else:
        # 仅统计流失率，展开为矩阵格式
        cross_stats = train_df_clean.groupby([row_col, col_col], observed=True)[target_col].mean().unstack(
            fill_value=0.0
        )
        cross_stats = (cross_stats.round(3) * 100).fillna(0.0)
        pivot_df = cross_stats

    # 格式化输出
    print(f"{row_col} × {col_col}交叉分析结果:")

    if include_count:
        print(cross_stats[['总用户数', '流失率']].to_string(
            formatters={"流失率": "{:.1f}%", "总用户数": "{:.0f}"}, index=False
        ))
    else:
        print(cross_stats.to_string(float_format=lambda x: f"{x:.1f}%"))

    if plot_heatmap:
        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot_df,
            annot=True,  # 显示流失率数值
            cmap=cmap,  # 颜色映射（默认红黄色系）
            fmt='.1f',
            cbar_kws={'label': '流失率(%)'},  # 颜色条标签
            linewidths=0.5  # 格子边框，增强可读性
        )
        plt.title(f'{row_col} × {col_col} 流失率热力图', fontsize=12)
        plt.xlabel(col_col, fontsize=10)
        plt.ylabel(row_col, fontsize=10)
        plt.tight_layout()
        plt.show()

    return cross_stats, train_df_clean

def analyze_three_way_interactions(train_df_clean): # 三维交叉分析
    # 校验必要列（基于清洗后的数据集）
    required_cols = [
        'international_plan', 'cs_call_bin', 'voice_mail_plan',
        'total_day_minutes_bin', 'total_day_charge_bin',
        'total_eve_minutes_bin', 'total_eve_charge_bin',
        'total_night_minutes_bin', 'total_night_charge_bin',
        'total_intl_minutes_bin', 'total_intl_charge_bin',
        'total_day_calls_bin', 'churn_num']

    # 检查缺失列
    missing_cols = [col for col in required_cols if col not in train_df_clean.columns]
    if missing_cols:
        raise ValueError(f"数据集缺少必要列（请确保分箱函数已正确执行）：{', '.join(missing_cols)}")

    print("模块1：国际套餐 × 客服呼叫次数 × 语音邮件套餐 三维交叉分析")

    # 分组维度
    group_cols = ['international_plan', 'cs_call_bin', 'voice_mail_plan']
    grouped_stats = train_df_clean.groupby(group_cols, observed=True).agg(
        总用户数=('churn', 'count'),
        流失用户数=('churn_num', 'sum')
    ).reset_index()

    # 计算流失率
    grouped_stats['流失率(%)'] = (grouped_stats['流失用户数'] / grouped_stats['总用户数'] * 100).round(1)
    grouped_stats = grouped_stats.sort_values(by=group_cols) # 按分组维度排序
    print("\n三维交叉统计结果")
    print(grouped_stats.to_string(index=False))

    # 可视化：热力图（行=国际套餐+语音邮件，列=客服呼叫次数）
    heatmap_data = grouped_stats.pivot_table(
        index=['international_plan', 'voice_mail_plan'],
        columns='cs_call_bin',
        values='流失率(%)',
        observed=False
    )

    plt.figure(figsize=(10, 5))
    sns.heatmap(
        heatmap_data,
        annot=True,  # 标注数值
        cmap='RdYlBu_r',  # 红→黄→蓝（红=高流失，蓝=低流失）
        fmt='.1f',  # 数值格式
        linewidths=0.5,  # 格子边框
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={'label': '流失率(%)'}
    )
    plt.title('客户流失率热力图（国际套餐+客服呼叫次数+语音邮件）', fontsize=10, fontweight='bold')
    plt.xlabel('客服呼叫次数分段', fontsize=10)
    plt.ylabel('国际套餐+语音邮件套餐', fontsize=10)

    plt.tight_layout()
    plt.show()

    print("模块2:通话时段+通话时长+通话费用 三维交叉分析（含风险对比）")

    # 定义“时长-费用”对应对（确保时段一致）
    duration_charge_pairs = [
        ('total_day_minutes', 'total_day_charge'),  # 日间：时长×费用
        ('total_eve_minutes', 'total_eve_charge'),  # 晚间：时长×费用
        ('total_night_minutes', 'total_night_charge'),  # 夜间：时长×费用
        ('total_intl_minutes', 'total_intl_charge')  # 国际：时长×费用
    ]

    # 存储各时段交叉结果的字典
    cross_duration_charge = {}

    # 批量处理每个时段
    for duration_col, charge_col in duration_charge_pairs:
        # 获取分箱列名（复用之前生成的分箱列：xxx_bin）
        duration_bin_col = f'{duration_col}_bin'
        charge_bin_col = f'{charge_col}_bin'

        # 按时长分箱+费用分箱分组统计
        cross_stats = train_df_clean.groupby(
            [duration_bin_col, charge_bin_col], observed=True
        )['churn_num'].agg(
            客户数='count',
            流失数='sum',
            流失率=lambda x: round(x.mean(), 4)  # 保留4位小数，后续转百分比
        ).reset_index()

        # 提取时段名称（day/eve/night/intl），作为字典key
        period = duration_col.split('_')[1]
        cross_duration_charge[period] = cross_stats

        # 打印该时段的关键对比结果
        print(f"\n{period}时段：时长+费用 交叉流失率")

        # 筛选高时长、高费用、高时长+高费用三组
        high_duration = cross_stats[cross_stats[duration_bin_col].str.contains('高')]
        high_charge = cross_stats[cross_stats[charge_bin_col].str.contains('高')]
        high_both = cross_stats[
            cross_stats[duration_bin_col].str.contains('高') &
            cross_stats[charge_bin_col].str.contains('高')
            ]

        # 计算核心指标（处理无数据的边界情况）
        high_dur_risk = high_duration['流失率'].mean() * 100 if not high_duration.empty else 0.0
        high_charge_risk = high_charge['流失率'].mean() * 100 if not high_charge.empty else 0.0

        # 输出对比结果
        print(f"1. 高时长组平均流失率：{high_dur_risk:.1f}%")
        print(f"2. 高费用组平均流失率：{high_charge_risk:.1f}%")

        if not high_both.empty:
            combined_risk = high_both['流失率'].iloc[0] * 100
            combined_count = high_both['客户数'].iloc[0]
            single_max_risk = max(high_dur_risk, high_charge_risk)
            risk_diff = combined_risk - single_max_risk  # 组合风险 - 单一组最高风险
            print(f"3. 高时长+高费用组流失率：{combined_risk:.1f}%")
            print(f"4. 双重风险增幅：{risk_diff:.1f}%")
            print(f"5. 高组合样本数：{combined_count}（样本量≥10才具统计意义）")
        else:
            print("3. 无高时长+高费用组合数据")

    # 可视化热力图（以日间为例，其他时段逻辑一致）
    period = 'day'  #### 可改为'eve'/'night'/'intl'查看其他时段
    cross_data = cross_duration_charge[period].copy()
    duration_bin_col = f'total_{period}_minutes_bin'
    charge_bin_col = f'total_{period}_charge_bin'

    # 构建热力图透视表
    pivot_df = cross_data.pivot_table(
        index=duration_bin_col,
        columns=charge_bin_col,
        values='流失率',
        observed=True,
        fill_value=0.0
    )

    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.1%',
        cmap='Reds',
        cbar_kws={'label': '流失率'},
        linewidths=0.5
    )

    # 标记「高时长+高费用」组合
    high_duration_rows = [r for r in pivot_df.index if '高' in str(r)]
    high_charge_cols = [c for c in pivot_df.columns if '高' in str(c)]

    if high_duration_rows and high_charge_cols:
        high_row = high_duration_rows[0]
        high_col = high_charge_cols[0]
        combined_risk = pivot_df.loc[high_row, high_col]
        # 计算单一组最高风险
        high_dur_data = cross_data[cross_data[duration_bin_col].str.contains('高', na=False)]
        high_charge_data = cross_data[cross_data[charge_bin_col].str.contains('高', na=False)]
        high_dur_risk = high_dur_data['流失率'].mean() if not high_dur_data.empty else 0.0
        high_charge_risk = high_charge_data['流失率'].mean() if not high_charge_data.empty else 0.0
        single_max_risk = max(high_dur_risk, high_charge_risk)
        risk_diff = combined_risk - single_max_risk

        # 生成标签文本
        if risk_diff > 0.05:  # 增幅≥5%，视为高风险
            label_text = f'高风险组合\n{combined_risk:.1%}（↑{risk_diff:.1%}）'
            marker_color = 'black'
        elif risk_diff < -0.05:  # 降幅≥5%，视为低风险
            label_text = f'低风险组合\n{combined_risk:.1%}（↓{abs(risk_diff):.1%}）'
            marker_color = 'green'
        else:  # 无显著差异
            label_text = f'无显著差异\n{combined_risk:.1%}'
            marker_color = 'gray'

        # 绘制标记点（星形，突出显示）
        plt.scatter(
            pivot_df.columns.get_loc(high_col) + 0.5,  # x坐标（列索引+0.5居中）
            pivot_df.index.get_loc(high_row) + 0.5,  # y坐标（行索引+0.5居中）
            s=200,  # 大小
            color=marker_color,
            marker='*',  # 星形标记
            label=label_text
        )

        # 图表样式调整
        plt.legend(loc='upper left')  # 图例位置
        plt.title(f'{period}时段：时长×费用 交叉流失率热力图', fontsize=12, fontweight='bold')
        plt.xlabel('费用分箱', fontsize=10)
        plt.ylabel('时长分箱', fontsize=10)
        plt.tight_layout()
        plt.show()

        # 输出全时段结论
        print("\n各时段时长×费用 交叉分析结论：")
        for period in cross_duration_charge.keys():
            data = cross_duration_charge[period]
            duration_bin_col = f'total_{period}_minutes_bin'
            charge_bin_col = f'total_{period}_charge_bin'

            # 筛选高组合（防空值）
            high_both = data[
                data[duration_bin_col].str.contains('高', na=False) &
                data[charge_bin_col].str.contains('高', na=False)
                ]

            # 计算单一组最高风险
            high_dur_data = data[data[duration_bin_col].str.contains('高', na=False)]
            high_charge_data = data[data[charge_bin_col].str.contains('高', na=False)]
            high_dur_risk = high_dur_data['流失率'].mean() if not high_dur_data.empty else 0.0
            high_charge_risk = high_charge_data['流失率'].mean() if not high_charge_data.empty else 0.0
            single_max_risk = max(high_dur_risk, high_charge_risk)

            # 生成结论文本
            if not high_both.empty:
                combined_risk = high_both['流失率'].iloc[0]
                risk_diff = combined_risk - single_max_risk
                if risk_diff > 0.05:
                    desc = f"显著高于单一组（高{risk_diff:.1%}），属于双重风险因子"
                elif risk_diff < -0.05:
                    desc = f"显著低于单一组（低{abs(risk_diff):.1%}），风险抵消"
                else:
                    desc = "与单一组无显著差异，无叠加效应"
                print(f"- {period}时段：高组合流失率{combined_risk:.1%}，{desc}")
            else:
                print(f"- {period}时段：无高时长+高费用组合数据，无法分析叠加效应")

    print("模块3：日间通话次数 × 国际套餐 × 客服呼叫次数 三维交叉分析")

    # 分组维度（日间通话次数分箱列+国际套餐+客服呼叫次数）
    call_bin_col = 'total_day_calls_bin'  # 主函数已生成的分箱列
    group_cols = ['international_plan', 'cs_call_bin', call_bin_col]

    # 分组统计（筛选样本量≥10的组，保证统计意义）
    call_cross_stats = train_df_clean.groupby(group_cols, observed=True).agg(
        总用户数=('churn_num', 'count'),
        流失用户数=('churn_num', 'sum')
    ).reset_index()

    # 计算流失率（百分比）
    call_cross_stats['流失率(%)'] = (call_cross_stats['流失用户数'] / call_cross_stats['总用户数'] * 100).round(1)
    # 筛选有效组（样本量≥10）+ 排序
    call_cross_stats = call_cross_stats[call_cross_stats['总用户数'] >= 10].sort_values(by=group_cols)

    # 打印统计结果
    print("\n三维交叉统计结果（仅显示样本量≥10的组）")
    print(call_cross_stats.to_string(index=False))

    # 可视化：热力图（行=国际套餐+通话次数，列=客服呼叫次数）
    heatmap_data = call_cross_stats.pivot_table(
        index=['international_plan', call_bin_col],
        columns='cs_call_bin',
        values='流失率(%)',
        observed=True,
        fill_value=0.0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,  # 显示流失率数值
        cmap='RdYlGn_r',  # 红→黄→绿（高流失→低流失）
        fmt='.1f',  # 1位小数
        linewidths=0.5,  # 格子边框
        cbar_kws={'label': '流失率(%)'})

    plt.title('日间通话次数 × 国际套餐 × 客服呼叫次数 流失率热力图', fontsize=12, fontweight='bold')
    plt.xlabel('客服呼叫次数分段', fontsize=10)
    plt.ylabel('国际套餐 + 日间通话次数分箱', fontsize=10)

    plt.tight_layout()
    plt.show()

    return train_df_clean

if __name__ == '__main__':

    train_df = load_data("train.csv")
    train_df = data_quality_check(train_df)
    train_df, churn_rate_no, churn_rate_yes = analyze_international_plan_churn(train_df)
    train_df = plot_international_plan_churn(train_df, churn_rate_no, churn_rate_yes)
    train_df = analyze_number_customer_service_calls(train_df)
    train_df = analyze_call_duration(train_df)
    train_df = analyze_call_charge(train_df)
    train_df = analyze_call_frequency(train_df)
    train_df = analyze_voice_mail_plan(train_df)
    train_df = analyze_account_length(train_df)

    cross_analysis_train_df = pd.read_csv("train.csv")
    cross_analysis_train_df['churn_num'] = cross_analysis_train_df['churn'].str.lower().map({'no':0, 'yes':1}).fillna(0)
    cross_analysis_train_df, bin_boundaries = create_bin_features(df=cross_analysis_train_df,is_train=True)

    cross_stats1, train_df_clean = analyze_bivariate_relationships(train_df=cross_analysis_train_df,
        row_col='cs_call_bin',col_col='vmail_msg_bin',target_col='churn_num',include_count=False)
    cross_stats2, train_df_clean = analyze_bivariate_relationships(train_df=train_df_clean,
        row_col='international_plan',col_col='vmail_msg_bin',target_col='churn_num',include_count=False)
    cross_stats3, train_df_clean = analyze_bivariate_relationships(train_df=train_df_clean,
        row_col='account_years_bin',col_col='vmail_msg_bin',target_col='churn_num',include_count=False)
    cross_stats4, train_df_clean = analyze_bivariate_relationships(train_df=train_df_clean,
        row_col='total_day_calls_bin',col_col='total_day_charge_bin',target_col='churn_num',include_count=False)

    train_df_clean = analyze_three_way_interactions(train_df_clean)