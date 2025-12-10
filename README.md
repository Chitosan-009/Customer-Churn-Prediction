# Customer Churn Prediction

##  项目背景
本项目通过机器学习技术构建高精度流失预测模型，核心价值在于精准识别高流失风险客户，提前介入挽留；挖掘流失关键驱动因素，优化产品与服务策略；降低客户获取成本，提升用户生命周期价值。

##  数据来源与内容
- 数据来源：Kaggle
- 训练数据集包含4250个样本。每个样本包含19个特征和1个布尔变量“churn”,表示样本的类别。 19个输入功能和1个目标变量是:
    
    state,                         string. 2-letter code of the US state of customer residence
    account_length,                numerical. Number of months the customer has been with the current telco provider
    area_code,                     string="area_code_AAA" where AAA = 3 digit area code.
    international_plan,            (yes/no). The customer has international plan.
    voice_mail_plan,               (yes/no). The customer has voice mail plan.
    number_vmail_messages,         numerical. Number of voice-mail messages.
    total_day_minutes,             numerical. Total minutes of day calls.
    total_day_calls,               numerical. Total number of day calls.
    total_day_charge,              numerical. Total charge of day calls.
    total_eve_minutes,             numerical. Total minutes of evening calls.
    total_eve_calls,               numerical. Total number of evening calls.
    total_eve_charge,              numerical. Total charge of evening calls.
    total_night_minutes,           numerical. Total minutes of night calls.
    total_night_calls,             numerical. Total number of night calls.
    total_night_charge,            numerical. Total charge of night calls.
    total_intl_minutes,            numerical. Total minutes of international calls.
    total_intl_calls,              numerical. Total number of international calls.
    total_intl_charge,             numerical. Total charge of international calls
    number_customer_service_calls, numerical. Number of calls to customer service
    churn, (yes/no).               Customer churn - target variable.

##  数据清洗与分析
- 检查缺失值、重复值、异常值并处理；
- 进行单变量与多变量分析，探索特征与流失率的关系；
- 对分类变量绘制分布图，对连续变量进行分箱（binning）与分组统计；
- 业务发现：
  - 客服呼叫 ≥4 次的用户流失率显著提升；
  - 开通国际套餐的用户流失率 32%（未开通用户仅 11%），需优化套餐服务；
  - 高费用区间用户流失率27%，用户价格敏感度较高
  - 开通语音邮件套餐的用户流失率仅 8%，增值服务可提升用户粘性。

##  特征工程
- 使用自定义分箱函数生成分箱特征；
- 构造衍生特征（费用占比、通话稳定性、国际使用频率等）；
- One-hot 编码与标准化；
- 训练集 / 测试集划分并保持分布一致性；
- 保存分箱规则与编码映射，保证推理阶段一致性。

##  模型构建与评估
使用多种模型进行对比与调优：
| 模型 | ROC-AUC | 优势 |
|------|----------------|------|
| Logistic Regression | 0.809000 | 可解释性强 |
| Random Forest | 0.904100 | 非线性拟合能力强 |
| XGBoost | 0.923000 | 泛化能力好，训练速度快 |
| LightGBM | 0.929100 | 最优表现 |

- 采用 5 折交叉验证与随机搜索调参；
- 输出混淆矩阵、ROC 曲线、重要特征排序；
- 保存最佳模型用于预测。

##  模型解释与业务建议
- 使用逻辑回归系数 / 树模型特征重要度分析影响因素；
- 对流失用户画像建模；
- 主要建议：
  1. 针对客服呼叫频繁客户建立自动跟进机制；
  2. 为国际套餐用户提供轻量级优惠；
  3. 优化高额账单客户服务体验；
  4. 建立流失预警评分系统。

## ⚙️ 运行方式
```bash
# 环境安装
pip install -r requirements.txt

# 运行顺序
1. 数据清洗与分析.ipynb
2. 特征工程与预处理.ipynb
3. 模型训练与解释.ipynb
