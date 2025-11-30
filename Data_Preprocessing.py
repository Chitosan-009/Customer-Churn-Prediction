import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import os
from matplotlib import font_manager

def load_data(file_path, encoding="utf-8", na_values=None, verbose=True):
    """通用数据加载函数（处理CSV文件，含异常处理）
        :param file_path: 数据文件路径（如"train.csv"）
        :param encoding: 文件编码（默认utf-8）
        :param na_values: 缺失值标识（如["NA", ""]）
        :param verbose: 是否输出加载日志
        :return: 加载后的DataFrame"""
    if verbose:
        print(f"正在加载数据：{file_path}")

        # 异常处理：文件不存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件：{file_path}（请检查路径是否正确）")

        # 加载数据（保留pandas原生参数，支持自定义）
    try:
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            na_values=na_values
        )
    except Exception as e:
        raise RuntimeError(f"加载数据失败：{str(e)}")
    # 输出加载信息（帮助用户确认数据维度）
    if verbose:
        print(f"数据加载成功！共 {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"特征列表：{list(df.columns)}")
        print()  # 空行分隔，日志更清晰

    return df

def data_quality_check(df,check_name="训练集", plot_outliers=True, verbose=True):
    # 字体配置
    preferred_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC', 'Noto Sans CJK SC']
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
    font_set_success = False

    for f in preferred_fonts:
        if f in available_fonts:
            plt.rcParams['font.sans-serif'] = [f]
            plt.rcParams['axes.unicode_minus'] = False
            if verbose:
                print(f"已设置字体: {f}（避免中文乱码）")
            font_set_success = True
            break

    if not font_set_success and verbose:
        print("未找到适配的中文字体，可能出现中文乱码（建议安装SimHei或Microsoft YaHei）")

    # 缺失值检查
    Missing_values = df.isnull().sum()
    if verbose:
        print(f'\n{check_name}缺失情况: {Missing_values}')
        if Missing_values.sum() == 0:
            print(f"{check_name}无缺失值")

    # 重复行检查
    duplicate_count = df.duplicated().sum()
    if verbose:
        print(f"{check_name}重复行数量: {duplicate_count}")
        if duplicate_count == 0:
            print(f"{check_name}无重复行")

    # 异常值检查
    if verbose:
        print(f"\n异常值情况（数值型特征）：")
    total_outliers = 0
    total_records = df.shape[0] * df.select_dtypes(include=['int64', 'float64']).shape[1]

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        total_outliers += outliers.shape[0]
        if verbose:
            print(f'{col} 的异常值数量：{outliers.shape[0]}')

    outliers_ratio = total_outliers / total_records if total_records > 0 else 0
    if verbose:
        print(f'所有异常值占整体数据集的比例：{outliers_ratio * 100:.2f}%')

    # 异常值可视化
    if plot_outliers:
        if verbose:
            print(f"\n异常值可视化（箱线图）：")
        sns.set(style="whitegrid")
        # 再次确认字体（防止首次设置失败）
        plt.rcParams['font.sans-serif'] = ['SimHei'] if 'SimHei' in available_fonts else plt.rcParams['font.sans-serif']

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n_cols = 5
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 4 * n_rows))

    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_pos = i % n_cols
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])

        if n_rows == 1:
            sns.boxplot(x=df[col], ax=axes[col_pos])
            axes[col_pos].set_title(f"{col} (异常值: {outliers_count})")
        else:
            sns.boxplot(x=df[col], ax=axes[row, col_pos])
            axes[row, col_pos].set_title(f"{col} (异常值: {outliers_count})")

    plt.tight_layout()
    plt.show()

    # 分类变量检查
    if verbose:
        print(f"\n分类变量情况：")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if verbose:
            print(f'\nFrequency of {col}:')
            print(df[col].value_counts())
    if verbose:
        print("对个分类型变量进行了频次统计，由统计结果可以看出无异常情况。 ")

    if verbose:
        print("\n" + "="*60 + "\n")

    return df

def processing_train(train_df):
    print("数据检查与清洗（训练集）")
    print("数据质量复检：")
    print(f"缺失值：\n{train_df.isnull().sum()[train_df.isnull().sum() > 0]}")  # 仅显示有缺失的列
    print(f"重复行数量：{train_df.duplicated().sum()}")
    print(f"数据形状：{train_df.shape}")

    # 缺失值处理
    missing_cols = ['total_night_minutes_bin', 'total_night_charge_bin']
    for col in missing_cols:
        if col in train_df.columns and train_df[col].isnull().sum() > 0:
            mode_val = train_df[col].mode()[0]
            train_df[col] = train_df[col].fillna(mode_val)
            print(f"  - 用众数 '{mode_val}' 填充了 '{col}' 的缺失值")

    # 目标变量处理
    if 'churn' in train_df.columns:
        train_df['churn_num'] = (train_df['churn'] == 'yes').astype(int)
        print("已将训练集的 'churn' 转换为数值型 'churn_num'。")
    else:
        print("训练集未找到 'churn' 列，无法创建目标变量。")

    # 再次检查
    print(f"\n处理后缺失值：\n{train_df.isnull().sum()[train_df.isnull().sum() > 0]}")
    print(f"处理后重复行数量：{train_df.duplicated().sum()}")
    print(f"处理后数据形状：{train_df.shape}")

    return train_df

def create_bin_features(df, is_train=True, bin_boundaries=None):
    """ 为DataFrame创建分箱特征。
    :param df: 待处理的DataFrame
    :param is_train: 是否为训练集。若是，计算分箱边界；若否，使用传入的边界。
    :param bin_boundaries: 字典，包含分箱边界。仅在is_train=False时使用。
    :return: 处理后的DataFrame和分箱边界字典（仅当is_train=True时）"""
    if not is_train and bin_boundaries is None:
        raise ValueError("当is_train=False（测试集分箱）时，必须传入bin_boundaries（训练集的分箱边界）！")

    df_copy = df.copy()
    boundaries = {} if is_train else bin_boundaries

    # 通话次数分箱（反映频率）
    call_cols = ['total_day_calls', 'total_eve_calls', 'total_night_calls', 'total_intl_calls']
    for col in call_cols:
        if col in df_copy.columns:
            bin_col = f'{col}_bin'
            if is_train:
                # 使用四分位数分箱
                labels = [f'{col.split("_")[1]}_call_低', f'{col.split("_")[1]}_call_中低',
                          f'{col.split("_")[1]}_call_中高', f'{col.split("_")[1]}_call_高']
                df_copy[bin_col] = pd.qcut(df_copy[col], q=4, labels=labels, duplicates='drop')
                # 存储分箱边界
                quantiles = df_copy[col].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
                boundaries[col] = quantiles
            else:
                if col in boundaries:
                    labels = [f'{col.split("_")[1]}_call_低', f'{col.split("_")[1]}_call_中低',
                              f'{col.split("_")[1]}_call_中高', f'{col.split("_")[1]}_call_高']
                    # 使用训练集的分位数进行分箱
                    df_copy[bin_col] = pd.cut(df_copy[col], bins=boundaries[col], labels=labels, include_lowest=True)

    # 费用分箱（核心成本特征，替代时长）
    charge_bin_defs = {
        'total_day_charge': ([0, 17, 26, 36, float('inf')], ['day_低(0-17)', 'day_中低(17-26)', 'day_中高(26-36)', 'day_高(36+)']),
        'total_eve_charge': ([0, 11, 17, float('inf')], ['eve_低(0-11)', 'eve_中(11-17)', 'eve_高(17+)']),
        'total_night_charge': ([0, 8, 11, float('inf')], ['night_低(0-8)', 'night_中(8-11)', 'night_高(11+)']),
        'total_intl_charge': ([0, 2.7, 4.0, 5.4, float('inf')], ['intl_低(0-2.7)', 'intl_中低(2.7-4.0)', 'intl_中高(4.0-5.4)', 'intl_高(5.4+)'])
    }
    for col, (bins, labels) in charge_bin_defs.items():
        if col in df_copy.columns:
            bin_col = f'{col}_bin'
            if is_train:
                df_copy[bin_col] = pd.cut(df_copy[col], bins=bins, labels=labels)
                boundaries[col] = bins
            else:
                if col in boundaries:
                    df_copy[bin_col] = pd.cut(df_copy[col], bins=boundaries[col], labels=labels, include_lowest=True)
    # 时长分箱
    duration_bin_defs = {
        'total_day_minutes': ([0, 143, 180, 216, float('inf')], ['day时长_低(0-143)', 'day时长_中低(143-180)', 'day时长_中高(180-216)', 'day时长_高(216+)']),
        'total_eve_minutes': ([0, 166, 234, float('inf')], ['eve时长_低(0-166)', 'eve时长_中(166-234)', 'eve时长_高(234+)']),
        'total_night_minutes': ([0, 167, 235, float('inf')], ['night时长_低(0-167)', 'night时长_中(167-235)', 'night时长_高(235+)']),
        'total_intl_minutes': ([0, 8.5, 10.3, 12.0, float('inf')], ['intl时长_低(0-8.5)', 'intl时长_中低(8.5-10.3)', 'intl时长_中高(10.3-12.0)', 'intl时长_高(12.0+)'])
    }
    for col, (bins, labels) in duration_bin_defs.items():
        if col in df_copy.columns:  # 确保原始时长字段存在（训练集和测试集都需要保留到分箱后再删除）
            bin_col = f'{col}_bin'
            if is_train:
                # 训练集按边界分箱，并保存边界
                df_copy[bin_col] = pd.cut(df_copy[col], bins=bins, labels=labels, include_lowest=True)
                boundaries[col] = bins  # 存储时长字段的分箱边界，供测试集使用
            else:
      # 测试集使用训练集的边界分箱
                if col in boundaries:
                    df_copy[bin_col] = pd.cut(df_copy[col], bins=boundaries[col], labels=labels, include_lowest=True)

    # 客服呼叫次数分箱
    if 'number_customer_service_calls' in df_copy.columns:
        col = 'number_customer_service_calls'
        bin_col = 'cs_call_bin'
        bins = [-1, 0, 3, float('inf')]
        labels = ['0次呼叫', '1-3次呼叫', '4+次呼叫']
        if is_train:
            df_copy[bin_col] = pd.cut(df_copy[col], bins=bins, labels=labels)
            boundaries[col] = bins
        else:
            if col in boundaries:
                df_copy[bin_col] = pd.cut(df_copy[col], bins=boundaries[col], labels=labels, include_lowest=True)

    # 语音邮件数量分箱
    if 'number_vmail_messages' in df_copy.columns:
        col = 'number_vmail_messages'
        bin_col = 'vmail_msg_bin'
        bins = [-1, 0, 10, 30, float('inf')]
        labels = ['未使用(0)', '低频(1-10)', '中频(11-30)', '高频(31+)']
        if is_train:
            df_copy[bin_col] = pd.cut(df_copy[col], bins=bins, labels=labels)
            boundaries[col] = bins
        else:
            if col in boundaries:
                df_copy[bin_col] = pd.cut(df_copy[col], bins=boundaries[col], labels=labels, include_lowest=True)

    # 账户时长分箱
    if 'account_length' in df_copy.columns:
        df_copy["account_years"] = (df_copy["account_length"] / 12).round(1)
        col = 'account_years'
        bin_col = 'account_years_bin'
        max_years = df_copy[col].max()
        bins = [0, 1, 2, 3, 5, 10, max_years]
        labels = ["0-1年", "1-2年", "2-3年", "3-5年", "5-10年", f"10年以上（≤{max_years:.1f}年）"]
        if is_train:
            df_copy[bin_col] = pd.cut(df_copy[col], bins=bins, labels=labels)
            boundaries[col] = bins
        else:
            if col in boundaries:
                df_copy[bin_col] = pd.cut(df_copy[col], bins=boundaries[col], labels=labels, include_lowest=True)

    return df_copy, boundaries

def create_derived_features(df):
    """
    为DataFrame创建衍生特征。
    :param df: 待处理的DataFrame
    :return: 处理后的DataFrame
    """
    df_copy = df.copy()

    # 1、总费用与费用占比（反映时段依赖度）
    charge_cols = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge']
    if all(col in df_copy.columns for col in charge_cols):
        df_copy["total_charge"] = df_copy[charge_cols].sum(axis=1)

        # 添加 charge_level 特征（客户消费水平定义）
        if 'total_charge' in df_copy.columns:
            # 使用分位数定义消费水平
            low_threshold = df_copy['total_charge'].quantile(0.25)
            high_threshold = df_copy['total_charge'].quantile(0.75)

            df_copy['charge_level'] = pd.cut(
                df_copy['total_charge'],
                bins=[-np.inf, low_threshold, high_threshold, np.inf],
                labels=['低费用客户', '中等费用客户', '高费用客户']
            )
            print(f"已创建 charge_level 特征，阈值：低({low_threshold:.2f})，高({high_threshold:.2f})")

        # 总费用分箱
        df_copy['total_charge_bin'] = pd.qcut(df_copy['total_charge'], q=4,
                                              labels=['总费用_极低', '总费用_中低', '总费用_中高', '总费用_极高'])

        # 处理除零：总费用为0时用0.01替代
        df_copy['total_charge_safe'] = df_copy['total_charge'].replace(0, 0.01)
        df_copy['day_charge_ratio'] = df_copy['total_day_charge'] / df_copy['total_charge_safe']
        df_copy['eve_charge_ratio'] = df_copy['total_eve_charge'] / df_copy['total_charge_safe']
        df_copy['night_charge_ratio'] = df_copy['total_night_charge'] / df_copy['total_charge_safe']
        df_copy['intl_charge_ratio'] = df_copy['total_intl_charge'] / df_copy['total_charge_safe']
        df_copy = df_copy.drop(columns=['total_charge_safe'])

    # 2、忠诚特征（费用+次数）
    if 'total_day_charge_bin' in df_copy.columns and 'total_day_calls_bin' in df_copy.columns:
        df_copy['is_day_loyal'] = (
                (df_copy['total_day_charge_bin'].str.contains('高')) &
                (df_copy['total_day_calls_bin'].str.contains('高'))
        ).astype(int)

    # 3、风险特征（客服不满+低粘性）
    if 'cs_call_bin' in df_copy.columns and 'voice_mail_plan' in df_copy.columns:
        df_copy['is_high_service_risk'] = (
                (df_copy['cs_call_bin'] == '4+次呼叫') &
                (df_copy['voice_mail_plan'] == 'no')
        ).astype(int)

    # 4、国际依赖特征（有国际套餐且国际高费用）
    if 'international_plan' in df_copy.columns and 'total_intl_charge_bin' in df_copy.columns:
        df_copy['is_intl_dependent'] = (
                (df_copy['international_plan'] == 'yes') &
                (df_copy['total_intl_charge_bin'].str.contains('高'))
        ).astype(int)

    # 5、价格敏感度特征（日间单位次数费用）
    if 'total_day_charge' in df_copy.columns and 'total_day_calls' in df_copy.columns:
        df_copy['day_calls_safe'] = df_copy['total_day_calls'].replace(0, 1)
        df_copy['day_cost_per_call'] = df_copy['total_day_charge'] / df_copy['day_calls_safe']
        df_copy = df_copy.drop(columns=['day_calls_safe'])

    # 6、长期价值特征（总费用 × 账户年限 → 核心客户标识）
    if 'total_charge' in df_copy.columns and 'account_years' in df_copy.columns:
        df_copy['long_term_value'] = df_copy['total_charge'] * df_copy['account_years']

    # 7、通话稳定性特征（变异系数：波动越大越不稳定）
    call_cols = ['total_day_calls', 'total_eve_calls', 'total_night_calls']
    if all(col in df_copy.columns for col in call_cols):
        df_copy['call_mean'] = df_copy[call_cols].mean(axis=1)
        df_copy['call_mean_safe'] = df_copy['call_mean'].replace(0, 0.01)
        df_copy['call_stability'] = df_copy[call_cols].std(axis=1) / df_copy['call_mean_safe']
        df_copy = df_copy.drop(columns=['call_mean', 'call_mean_safe'])

    return df_copy

def delete_col(train_df):
    # 删除冗余字段（训练集）
    # 分析通话时长与费用的相关性，验证是否需要删除时长
    duration_charge_cols = [
        ('total_day_minutes', 'total_day_charge'),
        ('total_eve_minutes', 'total_eve_charge'),
        ('total_night_minutes', 'total_night_charge'),
        ('total_intl_minutes', 'total_intl_charge')
    ]
    print("通话时长与对应费用的皮尔逊相关系数：")
    for dur_col, charge_col in duration_charge_cols:
        if dur_col in train_df.columns and charge_col in train_df.columns:
            corr = train_df[dur_col].corr(train_df[charge_col])
            print(f"  - {dur_col} 与 {charge_col}：{corr:.4f}")

    drop_cols = [
        'churn',
        'total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'
    ]
    # 确保只删除存在的列
    train_df = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])
    print(f"删除冗余字段后形状：{train_df.shape}")

    return train_df

def test_processing(bin_boundaries=None):
    # 加载测试集
    test_df = pd.read_csv(r"test.csv")
    print("原始测试集形状:", test_df.shape)

    # 检查缺失值
    Missing_values_test = test_df.isnull().sum()
    print(f'测试集缺失情况: {Missing_values_test[Missing_values_test > 0]}')

    # 检查重复值
    duplicate_count_test = test_df.duplicated().sum()
    print(f"测试集重复行数量: {duplicate_count_test}")

    # 目标变量处理（如果测试集有）
    if 'churn' in test_df.columns:
        test_df['churn_num'] = (test_df['churn'] == 'yes').astype(int)
        print("已将测试集的 'churn' 转换为数值型 'churn_num'。")
    else:
        print("测试集未找到 'churn' 列，跳过目标变量转换。")

    # 应用分箱和特征衍生
    # 使用从训练集学到的分箱边界
    test_df, _ = create_bin_features(test_df, is_train=False, bin_boundaries=bin_boundaries)
    test_df = create_derived_features(test_df)

    return test_df

def update_columns(test_df, train_df):

    # 删除冗余字段 (测试集)
    print("\n删除冗余字段 (测试集)")
    drop_cols_test = [
        'churn',  # 原始目标变量
        'total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'  # 时长字段
    ]
    # 确保只删除存在的列
    test_df = test_df.drop(columns=[col for col in drop_cols_test if col in test_df.columns])

    # 确保测试集没有 churn_num
    if 'churn_num' in test_df.columns:
        test_df = test_df.drop(columns=['churn_num'])
        print("已删除测试集的 'churn_num' 列")

    # 最终数据状态检查
    print(f"\n测试集在特征工程后形状: {test_df.shape}")
    print(f"训练集在特征工程后形状: {train_df.shape}")

    # 检查列名差异
    extra_cols_train = list(set(train_df.columns) - set(test_df.columns))
    extra_cols_test = list(set(test_df.columns) - set(train_df.columns))
    print(f"\n训练集特有列: {extra_cols_train}")
    print(f"测试集特有列: {extra_cols_test}")

    # 检查 charge_level 是否在两个数据集中都存在
    if 'charge_level' in train_df.columns and 'charge_level' in test_df.columns:
        print("charge_level 特征在训练集和测试集中都存在")
    elif 'charge_level' in train_df.columns:
        print("charge_level 特征只在训练集中存在")
    elif 'charge_level' in test_df.columns:
        print("charge_level 特征只在测试集中存在")
    else:
        print("charge_level 特征在两个数据集中都不存在")

    return test_df, train_df

def fit_encoders(train_df):
    """在训练集上拟合所有需要的编码器。
    :param train_df: 特征工程后的训练集（未编码）
    :return: 一个包含所有拟合好的编码器的字典"""
    encoders = {}

    # 1、目标编码: state（高基数特征）
    print("拟合 TargetEncoder for 'state'...")
    target_encoder = ce.TargetEncoder()
    # 拟合，但不转换
    target_encoder.fit(train_df['state'], train_df['churn_num'])
    encoders['state'] = target_encoder

    # 2、标签编码: 有序分类特征（_bin结尾+charge_level）
    print("拟合 LabelEncoders for categorical features...")
    label_features = [col for col in train_df.columns if col.endswith('_bin') or col == 'charge_level']
    label_encoders = {}

    for feature in label_features:
        # 确保特征是 category 类型
        train_df[feature] = train_df[feature].astype('category')
        le = LabelEncoder()
        # 在训练集的原始类别上拟合
        le.fit(train_df[feature].cat.categories)
        label_encoders[feature] = le

    encoders['label'] = label_encoders

    print("所有编码器拟合完成。")

    return encoders

def transform_data(df,train_df_encoded,encoders, is_train=True):
    """使用拟合好的编码器转换数据 训练集/测试集
    :param df: 待转换的 DataFrame (特征工程后)
    :param encoders: 从 fit_encoders 函数获得的编码器字典
    :param is_train: 是否为训练集。若是，需要处理目标变量。
    :return: 编码后的 DataFrame"""
    df_encoded = df.copy()

    # 目标编码转换: state
    print("转换 'state'...")
    df_encoded['state_encoded'] = encoders['state'].transform(df_encoded['state'])
    df_encoded = df_encoded.drop(columns=['state'])

    # 独热编码: 低基数特征
    print("转换低基数特征 (独热编码)...")
    one_hot_features = ['area_code', 'international_plan', 'voice_mail_plan']
    df_encoded = pd.get_dummies(df_encoded, columns=one_hot_features, prefix=one_hot_features, dummy_na=False)

    # 标签编码转换: 有序分类特征
    print("转换有序分类特征 (标签编码)...")
    label_features = encoders['label'].keys()

    for feature in label_features:
        # 确保特征是 category 类型
        df_encoded[feature] = df_encoded[feature].astype('category')

        # 获取训练集的合法类别
        train_categories = encoders['label'][feature].classes_

        # 处理看不见的类别和NaN
        # 找到所有不在训练集类别中或为NaN的值
        invalid_mask = ~df_encoded[feature].isin(train_categories) | df_encoded[feature].isna()

        if invalid_mask.any():
            # 在**原始训练集**中找到该特征最常见的类别标签
            # 注意：这里我们假设 df 是原始的、未编码的，所以直接取 mode
            most_common_category = df[feature].mode()[0]
            print(f"  - 在 '{feature}' 中发现 {invalid_mask.sum()} 个无效值，将用 '{most_common_category}' 填充。")

            # 用合法的类别标签填充
            df_encoded.loc[invalid_mask, feature] = most_common_category

        # 安全地设置类别
        df_encoded[feature] = df_encoded[feature].cat.set_categories(train_categories)

        # 执行标签编码
        df_encoded[feature] = encoders['label'][feature].transform(df_encoded[feature])

    # 测试集列对齐（确保与训练集列完全一致）
    if not is_train:
        train_cols=train_df_encoded.columns.tolist()
        # 补充缺失列（用0填充）
        missing_cols = set(train_cols) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0
        # 对齐列顺序
        df_encoded = df_encoded[train_cols]

    print("数据转换完成。")

    return df_encoded

def encode_data(train_df,test_df):
    # 1、执行编码流程
    # 拟合编码器（仅用训练集）
    encoders = fit_encoders(train_df)
    # 转换训练集
    train_df_encoded = transform_data(train_df,None, encoders, is_train=True)
    # 转换测试集
    test_df_encoded = transform_data(test_df,train_df_encoded, encoders, is_train=False)


    # 2、编码后数据校验
    # 查看数据集形状
    print("数据集形状：")
    print(f"训练集（编码后）：{train_df_encoded.shape} （行：{train_df_encoded.shape[0]}, 列：{train_df_encoded.shape[1]}）")
    print(f"测试集（编码后）：{test_df_encoded.shape} （行：{test_df_encoded.shape[0]}, 列：{test_df_encoded.shape[1]}）")

    # 查看字段详细信息（列名、数据类型、非空值数量）
    print("\n\n训练集（编码后）字段信息：")
    train_df_encoded.info()

    print("\n\n测试集（编码后）字段信息：")
    test_df_encoded.info()

    # 检查缺失值
    print("\n\n训练集（编码后）缺失值检查：")
    missing_train = train_df_encoded.isnull().sum()
    print(missing_train[missing_train > 0])  # 只显示有缺失值的列（正常应该为空）
    print("\n\n测试集（编码后）缺失值检查：")
    missing_test = test_df_encoded.isnull().sum()
    print(missing_test[missing_test > 0])  # 只显示有缺失值的列（正常应该为空）


    # 3、删除测试集多余的churn_num列
    if 'churn_num' in test_df_encoded.columns:
        test_df_encoded = test_df_encoded.drop(columns=['churn_num'])
        print("已成功删除测试集的 'churn_num' 列")
    else:
        print("测试集不存在 'churn_num' 列，无需删除")

    # 4、验证删除结果（查看训练集和测试集的列差异）
    print(f"\n训练集列数：{len(train_df_encoded.columns)}，包含列：{train_df_encoded.columns.tolist()[:5]}...")
    print(f"测试集列数：{len(test_df_encoded.columns)}，包含列：{test_df_encoded.columns.tolist()[:5]}...")

    # 确认目标变量仅存在于训练集
    print(f"\n训练集是否有 'churn_num'：{'churn_num' in train_df_encoded.columns}")
    print(f"测试集是否有 'churn_num'：{'churn_num' in test_df_encoded.columns}")

    return train_df,test_df,train_df_encoded,test_df_encoded

def feature_select(train_df_encoded,test_df_encoded):
    # 1、特征选择
    # 分离训练集的特征（X）和目标变量（y）
    X_train_full = train_df_encoded.drop(columns=['churn_num'])  # 所有特征
    y_train_full = train_df_encoded['churn_num']  # 目标变量（是否流失）
    X_test = test_df_encoded  # 测试集仅含特征

    # 2、用RFE+随机森林自动筛选最优特征
    # 初始化基模型（用随机森林，天然输出特征重要性）
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    # 初始化RFECV（递归特征消除+交叉验证）
    rfecv = RFECV(
        estimator=rf_selector,  # 用随机森林作为评估特征的基模型
        step=1,  # 每次删除1个最不重要的特征
        cv=3,    # 3折交叉验证
        scoring='roc_auc',  # 用ROC-AUC作为评估标准（二分类任务最优指标）
        verbose=1  # 输出筛选过程
    )
    rfecv.fit(X_train_full, y_train_full)  # 仅在训练集拟合

    # 提取最优特征结果
    selected_features = X_train_full.columns[rfecv.support_].tolist()  # 被选中的特征
    print(f"\n最优特征数：{len(selected_features)}（原特征数：{X_train_full.shape[1]}）")
    print(f"最优特征列表：\n{selected_features}")

    # 3、筛选数据（只保留最优特征）
    X_train_selected = X_train_full[selected_features]  # 筛选后训练集特征
    X_test_selected = X_test[selected_features]  # 筛选后测试集特征（与训练集一致）

    return X_train_selected, X_test_selected,X_train_full,y_train_full,selected_features

def train_data_split(X_train_selected, X_test_selected,y_train_full,selected_features):
    # 1、 拆分训练集和验证集（8:2）
    # stratify=y确保流失比例在两组中一致，避免数据分布偏差
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_selected,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    # 查看划分结果
    print(f"训练集：{X_train.shape} （样本数×特征数）")
    print(f"验证集：{X_val.shape}")
    print(f"测试集：{X_test_selected.shape}")
    print(f"训练集流失比例：{y_train.mean():.2f}")
    print(f"验证集流失比例：{y_val.mean():.2f}")  # 与训练集接近则划分合理

    # 2、数据标准化
    scaler = StandardScaler()
    scaler.fit(X_train)  # 学习训练集的均值和标准差

    # 对训练集、验证集、测试集执行相同标准化
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_selected)

    # 转成DataFrame（方便查看和后续使用）
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=selected_features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)

    print("查看训练集标准化后前3行：")
    print(X_train_scaled.head(3))

    return X_train_scaled,X_val_scaled,X_test_scaled,y_train,y_val

def save_(X_train_scaled,y_train,X_val_scaled,y_val,selected_features):
    print(type(X_train_scaled))
    print(type(y_train))
    print(type(X_val_scaled))
    print(type(y_val))
    print(type(selected_features))


if __name__ == '__main__':

    train_df = load_data("train.csv")
    train_df = data_quality_check(train_df, check_name="训练集")
    train_df = processing_train(train_df)

    train_df, bin_boundaries = create_bin_features(train_df, is_train=True)
    bin_cols = [col for col in train_df.columns if col.endswith('_bin')]
    print(f"分箱特征数量：{len(bin_cols)}个")
    print(f"分箱特征示例：{bin_cols[:5]}...")

    train_df = create_derived_features(train_df)
    derived_cols = [col for col in train_df.columns if
                    'ratio' in col or 'is_' in col or 'value' in col or
                    'stability' in col or 'cost_per' in col]
    print(f"衍生特征数量：{len(derived_cols)}个")
    print(f"训练集特征工程后形状: {train_df.shape}")

    train_df = delete_col(train_df)

    test_df = test_processing(bin_boundaries=bin_boundaries)
    test_df, train_df = update_columns(test_df, train_df)
    train_df, test_df, train_df_encoded, test_df_encoded = encode_data(train_df, test_df)

    X_train_selected, X_test_selected, X_train_full, y_train_full, selected_features = feature_select(train_df_encoded, test_df_encoded)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val = train_data_split(X_train_selected, X_test_selected, y_train_full, selected_features)

    save_(X_train_scaled, y_train, X_val_scaled, y_val, selected_features)

    print(f"最终数据维度:")
    print(f"  - 训练集: {X_train_scaled.shape}")
    print(f"  - 验证集: {X_val_scaled.shape}")
    print(f"  - 测试集: {X_test_scaled.shape}")
    print(f"  - 选中特征数: {len(selected_features)}")
    print(f"  - 训练集流失率: {y_train.mean():.3f}")
    print(f"  - 验证集流失率: {y_val.mean():.3f}")