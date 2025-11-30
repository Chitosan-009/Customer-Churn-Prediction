import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix, roc_curve)
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
from Data_Preprocessing import (load_data,processing_train, create_bin_features, data_quality_check,
                                create_derived_features, delete_col, test_processing, update_columns,
                                encode_data, feature_select, train_data_split)

warnings.filterwarnings('ignore')

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """评估单个模型，输出关键指标+并排展示混淆矩阵和ROC曲线"""
    print(f"\n{'=' * 50}")
    print(f"🔍 {model_name} 模型评估结果：")
    print(f"{'=' * 50}")

    # 计算核心指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    # 输出指标
    metrics_info = {
        "准确率（Accuracy）": f"{accuracy:.4f} → 整体预测正确率",
        "精确率（Precision）": f"{precision:.4f} → 预测流失的人中，实际流失的比例",
        "召回率（Recall）": f"{recall:.4f} → 真实流失的人中，被正确识别的比例",
        "F1-score": f"{f1:.4f} → 精确率和召回率的平衡值",
        "ROC-AUC": f"{roc_auc:.4f} → 模型区分能力"
    }

    for metric, desc in metrics_info.items():
        print(f"  - {metric}: {desc}")

    # 可视化展示
    _plot_model_evaluation(y_true, y_pred, y_prob, model_name, roc_auc)

    # 返回核心指标（用于后续对比）
    return {
        '模型名称': model_name,
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1-score': f1,
        'ROC-AUC': roc_auc,
        '验证集预测概率': y_prob
    }

def _plot_model_evaluation(y_true, y_pred, y_prob, model_name, roc_auc):
    """绘制模型评估图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['未流失（0）', '流失（1）'],
                yticklabels=['未流失（0）', '流失（1）'],
                ax=ax1)
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    ax1.set_title(f'{model_name} - 混淆矩阵', fontsize=14, pad=20)

    # 2. ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax2.plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {roc_auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7)  # 随机猜测基准线
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('假阳性率（False Positive Rate）', fontsize=12)
    ax2.set_ylabel('真阳性率（True Positive Rate）', fontsize=12)
    ax2.set_title(f'{model_name} - ROC曲线', fontsize=14, pad=20)
    ax2.legend(loc="lower right", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def explain_logistic_regression(model, model_name, feature_names):
    """解释逻辑回归模型"""
    coefs = pd.DataFrame({
        '特征名称': feature_names,
        '系数值': model.coef_[0]  # 逻辑回归系数 shape=(1, n_features)
    }).sort_values('系数值', key=abs, ascending=False)

    print("特征系数解释：")
    print("  正系数 → 特征值越大，流失概率越高")
    print("  负系数 → 特征值越大，流失概率越低")
    print("\nTop 10 重要特征：")
    print(coefs.round(4).head(10))

    # 可视化系数
    plt.figure(figsize=(12, 6))
    top_coefs = coefs.head(15)
    colors = ['red' if c > 0 else 'green' for c in top_coefs['系数值']]

    bars = plt.barh(range(len(top_coefs)), top_coefs['系数值'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_coefs)), top_coefs['特征名称'])
    plt.xlabel('系数值', fontsize=12)
    plt.title(f'{model_name} - Top 15 特征系数\n(红色=正向影响，绿色=负向影响)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, coef) in enumerate(zip(bars, top_coefs['系数值'])):
        plt.text(coef + (0.01 if coef >= 0 else -0.01), i, f'{coef:.3f}',
                 ha='left' if coef >= 0 else 'right', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

def explain_tree_model(model, model_name, feature_names):
    """解释树模型"""
    importances = pd.DataFrame({
        '特征名称': feature_names,
        '重要性得分': model.feature_importances_
    }).sort_values('重要性得分', ascending=False)

    print("特征重要性排名（得分越高，对预测越关键）：")
    print("\nTop 10 重要特征：")
    print(importances.round(4).head(10))

    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    top_importances = importances.head(10)

    bars = plt.barh(range(len(top_importances)), top_importances['重要性得分'],
                    color='orange', alpha=0.7)
    plt.yticks(range(len(top_importances)), top_importances['特征名称'])
    plt.xlabel('特征重要性得分', fontsize=12)
    plt.title(f'{model_name} - Top 10 特征重要性', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, top_importances['重要性得分'])):
        plt.text(score + 0.001, i, f'{score:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

def explain_model_independent(model, model_name, feature_names):
    """独立模型解释函数 - 单独查看模型的特征影响"""
    print(f"\n{model_name} 模型特征解释")
    print("-" * 50)

    # 逻辑回归 → 输出系数（正负向影响）
    if '逻辑回归' in model_name:
        explain_logistic_regression(model, model_name, feature_names)
    # 树模型 → 输出特征重要性
    else:
        explain_tree_model(model, model_name, feature_names)

def train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val, selected_features):
    """训练逻辑回归模型"""
    print("\n开始训练逻辑回归模型...")
    print("=" * 50)

    # 模型训练
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train_scaled, y_train)

    # 验证集预测
    y_val_pred = lr_model.predict(X_val_scaled)
    y_val_prob = lr_model.predict_proba(X_val_scaled)[:, 1]

    # 评估模型
    lr_metrics = evaluate_model(y_val, y_val_pred, y_val_prob, "逻辑回归")

    # 保存模型文件
    joblib.dump(lr_model, '逻辑回归模型.pkl')
    print("逻辑回归模型已保存为 '逻辑回归模型.pkl'")

    # 模型解释
    explain_model_independent(lr_model, "逻辑回归", selected_features)

    return lr_model, lr_metrics

def train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val, selected_features):
    """训练随机森林模型（带超参数调优）"""
    print("\n开始训练随机森林模型...")
    print("=" * 50)

    # 定义参数搜索空间
    rf_param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(5, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }

    print("🔧 正在进行随机搜索调参...")
    rf_random = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=20,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rf_random.fit(X_train_scaled, y_train)

    # 提取最优模型
    best_rf = rf_random.best_estimator_
    print(f"随机森林最优参数：{rf_random.best_params_}")
    print(f"最优交叉验证AUC：{rf_random.best_score_:.4f}")

    # 验证集预测
    y_val_pred = best_rf.predict(X_val_scaled)
    y_val_prob = best_rf.predict_proba(X_val_scaled)[:, 1]

    # 评估模型
    rf_metrics = evaluate_model(y_val, y_val_pred, y_val_prob, "调优版随机森林")

    # 保存模型文件
    joblib.dump(best_rf, '随机森林最优模型.pkl')
    print("随机森林模型已保存为 '随机森林最优模型.pkl'")

    # 模型解释
    explain_model_independent(best_rf, "随机森林", selected_features)

    return best_rf, rf_metrics

def train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, selected_features):
    """训练XGBoost模型（带超参数调优）"""
    print("\n开始训练XGBoost模型...")
    print("=" * 50)

    # 定义参数搜索空间
    xgb_param_dist = {
        'n_estimators': randint(100, 300),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }

    print("正在进行随机搜索调优...")
    xgb_random = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(random_state=42, objective='binary:logistic',
                                    eval_metric='auc', n_jobs=-1),
        param_distributions=xgb_param_dist,
        n_iter=20,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    xgb_random.fit(X_train_scaled, y_train)

    # 提取最优模型
    best_xgb = xgb_random.best_estimator_
    print(f"XGBoost最优参数：{xgb_random.best_params_}")
    print(f"最优交叉验证AUC：{xgb_random.best_score_:.4f}")

    # 验证集预测
    y_val_pred = best_xgb.predict(X_val_scaled)
    y_val_prob = best_xgb.predict_proba(X_val_scaled)[:, 1]

    # 评估模型
    xgb_metrics = evaluate_model(y_val, y_val_pred, y_val_prob, "XGBoost")

    # 保存模型文件
    joblib.dump(best_xgb, 'XGBoost最优模型.pkl')
    print("XGBoost模型已保存为 'XGBoost最优模型.pkl'")

    # 模型解释
    explain_model_independent(best_xgb, "XGBoost", selected_features)

    return best_xgb, xgb_metrics

def train_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val, selected_features):
    """训练LightGBM模型（带超参数调优）"""
    print("\n开始训练LightGBM模型...")
    print("=" * 50)

    # 定义参数搜索空间
    lgb_param_dist = {
        'n_estimators': randint(100, 300),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }

    print("正在进行随机搜索调优...")
    lgb_random = RandomizedSearchCV(
        estimator=lgb.LGBMClassifier(random_state=42, objective='binary',
                                     metric='auc', n_jobs=-1),
        param_distributions=lgb_param_dist,
        n_iter=20,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    lgb_random.fit(X_train_scaled, y_train)

    # 提取最优模型
    best_lgb = lgb_random.best_estimator_
    print(f"LightGBM最优参数：{lgb_random.best_params_}")
    print(f"最优交叉验证AUC：{lgb_random.best_score_:.4f}")

    # 验证集预测
    y_val_pred = best_lgb.predict(X_val_scaled)
    y_val_prob = best_lgb.predict_proba(X_val_scaled)[:, 1]

    # 评估模型
    lgb_metrics = evaluate_model(y_val, y_val_pred, y_val_prob, "LightGBM")

    # 保存模型文件
    joblib.dump(best_lgb, 'LightGBM最优模型.pkl')
    print("LightGBM模型已保存为 'LightGBM最优模型.pkl'")

    # 模型解释
    explain_model_independent(best_lgb, "LightGBM", selected_features)

    return best_lgb, lgb_metrics

def _plot_model_comparison(metrics_df):
    """绘制模型对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. ROC-AUC 对比
    models = metrics_df['模型名称']
    auc_scores = metrics_df['ROC-AUC']
    bars1 = ax1.bar(models, auc_scores, color='lightblue', alpha=0.8, edgecolor='navy')
    ax1.set_title('模型 ROC-AUC 对比', fontsize=14, pad=20)
    ax1.set_ylabel('ROC-AUC 得分', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    # 在柱子上添加数值
    for bar, score in zip(bars1, auc_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=10)

    # 2. F1-score 对比
    f1_scores = metrics_df['F1-score']
    bars2 = ax2.bar(models, f1_scores, color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax2.set_title('模型 F1-score 对比', fontsize=14, pad=20)
    ax2.set_ylabel('F1-score 得分', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=10)

    # 3. 精确率 vs 召回率 散点图
    ax3.scatter(metrics_df['精确率'], metrics_df['召回率'], s=100, alpha=0.7,
                c=metrics_df['ROC-AUC'], cmap='viridis')
    ax3.set_xlabel('精确率 (Precision)', fontsize=12)
    ax3.set_ylabel('召回率 (Recall)', fontsize=12)
    ax3.set_title('精确率 vs 召回率 (颜色深浅表示AUC)', fontsize=14, pad=20)
    ax3.grid(True, alpha=0.3)

    # 添加模型标签
    for i, model in enumerate(models):
        ax3.annotate(model, (metrics_df['精确率'][i], metrics_df['召回率'][i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    # 4. 综合指标雷达图（简化版）
    metrics_to_plot = ['准确率', '精确率', '召回率', 'F1-score', 'ROC-AUC']
    n_metrics = len(metrics_to_plot)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    for i, model in enumerate(models):
        values = metrics_df[metrics_to_plot].iloc[i].tolist()
        values += values[:1]  # 闭合雷达图
        ax4.plot(angles, values, 'o-', linewidth=2, label=model, markersize=4)
        ax4.fill(angles, values, alpha=0.1)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics_to_plot, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('模型综合性能雷达图', fontsize=14, pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

def _recommend_best_model(metrics_df, models_dict):
    """推荐最优模型"""
    best_model = metrics_df.iloc[0]
    second_model = metrics_df.iloc[1] if len(metrics_df) > 1 else None

    print(f"\n最优模型推荐")
    print("=" * 50)
    print(f"推荐模型：{best_model['模型名称']}")
    print(f"ROC-AUC：{best_model['ROC-AUC']:.4f}（最高）")
    print(f"F1-score：{best_model['F1-score']:.4f}")

    if second_model is not None:
        auc_improvement = best_model['ROC-AUC'] - second_model['ROC-AUC']
        print(f"相比第二名 {second_model['模型名称']}，AUC提升：{auc_improvement:.4f}")

    print(f"\n推荐理由：")
    print("  - 综合区分能力（AUC）和分类平衡能力（F1）最优")
    print("  - 在精确率和召回率之间取得良好平衡")
    print("  - 模型稳定性和泛化能力较强")

    # 保存最优模型信息
    best_model_info = {
        'model_name': best_model['模型名称'],
        'model': models_dict[best_model['模型名称']],
        'metrics': best_model.to_dict()
    }

    joblib.dump(best_model_info, '最优模型信息.pkl')
    print(f"\n最优模型信息已保存为 '最优模型信息.pkl'")

def compare_models(model_metrics, models_dict):
    """对比所有模型的性能"""
    if not model_metrics:
        print("没有可对比的模型指标")
        return

    print("\n四大模型核心指标汇总对比")
    print("=" * 80)

    # 转换为DataFrame并排序
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df = metrics_df[['模型名称', '准确率', '精确率', '召回率', 'F1-score', 'ROC-AUC']]
    metrics_df = metrics_df.round(4)
    metrics_df_sorted = metrics_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)

    # 美化输出
    print("模型性能对比表：")
    print(metrics_df_sorted.to_string(index=False))

    # 可视化对比
    _plot_model_comparison(metrics_df_sorted)

    # 输出最优模型推荐
    _recommend_best_model(metrics_df_sorted, models_dict)

def run_feature_engineering():
    """运行完整的特征工程流程"""
    print("开始特征工程流程...")
    print("=" * 50)

    try:
        # 加载数据
        train_df = load_data("train.csv")
        train_df = data_quality_check(train_df, check_name="训练集")
        train_df = processing_train(train_df)

        # 创建分箱特征
        train_df, bin_boundaries = create_bin_features(train_df, is_train=True)
        bin_cols = [col for col in train_df.columns if col.endswith('_bin')]
        print(f"分箱特征数量：{len(bin_cols)}个")

        # 创建衍生特征
        train_df = create_derived_features(train_df)
        derived_cols = [col for col in train_df.columns if
                        'ratio' in col or 'is_' in col or 'value' in col or
                        'stability' in col or 'cost_per' in col]
        print(f"衍生特征数量：{len(derived_cols)}个")
        print(f"训练集特征工程后形状: {train_df.shape}")

        # 删除冗余字段
        train_df = delete_col(train_df)

        # 处理测试集
        test_df = test_processing(bin_boundaries=bin_boundaries)
        test_df, train_df = update_columns(test_df, train_df)

        # 编码数据
        train_df, test_df, train_df_encoded, test_df_encoded = encode_data(train_df, test_df)

        # 特征选择
        X_train_selected, X_test_selected, X_train_full, y_train_full, selected_features = feature_select(
            train_df_encoded, test_df_encoded
        )

        # 数据拆分和标准化
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val = train_data_split(
            X_train_selected, X_test_selected, y_train_full, selected_features
        )

        print("特征工程完成！")
        print(f"最终数据维度:")
        print(f"  - 训练集: {X_train_scaled.shape}")
        print(f"  - 验证集: {X_val_scaled.shape}")
        print(f"  - 测试集: {X_test_scaled.shape}")
        print(f"  - 选中特征数: {len(selected_features)}")

        return {
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'selected_features': selected_features
        }

    except Exception as e:
        print(f"特征工程失败: {str(e)}")
        return None

def main():
    """主函数 - 执行完整的模型训练流程"""
    print("开始客户流失预测模型训练流程")
    print("=" * 60)

    try:
        # 运行特征工程流程
        feature_data = run_feature_engineering()
        if not feature_data:
            print("特征工程失败，终止流程！")
            return

        print("数据准备完成，开始模型训练...")

        # 提取特征工程结果
        X_train_scaled = feature_data['X_train_scaled']
        X_val_scaled = feature_data['X_val_scaled']
        X_test_scaled = feature_data['X_test_scaled']
        y_train = feature_data['y_train']
        y_val = feature_data['y_val']
        selected_features = feature_data['selected_features']

        # 存储模型和指标
        models_dict = {}
        model_metrics = []

        print("\n" + "模型训练阶段 ".ljust(50, "="))

        # 训练所有模型
        # 逻辑回归
        lr_model, lr_metrics = train_logistic_regression(
            X_train_scaled, y_train, X_val_scaled, y_val, selected_features
        )
        models_dict['逻辑回归'] = lr_model
        model_metrics.append(lr_metrics)

        # 随机森林
        rf_model, rf_metrics = train_random_forest(
            X_train_scaled, y_train, X_val_scaled, y_val, selected_features
        )
        models_dict['随机森林'] = rf_model
        model_metrics.append(rf_metrics)

        # XGBoost
        xgb_model, xgb_metrics = train_xgboost(
            X_train_scaled, y_train, X_val_scaled, y_val, selected_features
        )
        models_dict['XGBoost'] = xgb_model
        model_metrics.append(xgb_metrics)

        # LightGBM
        lgb_model, lgb_metrics = train_lightgbm(
            X_train_scaled, y_train, X_val_scaled, y_val, selected_features
        )
        models_dict['LightGBM'] = lgb_model
        model_metrics.append(lgb_metrics)

        # 模型对比
        print("\n" + "模型对比阶段 ".ljust(50, "="))
        compare_models(model_metrics, models_dict)

        print("\n模型训练流程完成！")
        print("=" * 60)

        # 输出训练总结
        print(f"训练总结：")
        print(f"  - 成功训练模型数量：{len(models_dict)}")
        print(f"  - 使用的特征数量：{len(selected_features)}")
        print(f"  - 训练集样本数：{X_train_scaled.shape[0]}")
        print(f"  - 验证集样本数：{X_val_scaled.shape[0]}")

    except Exception as e:
        print(f"模型训练过程中出现错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()