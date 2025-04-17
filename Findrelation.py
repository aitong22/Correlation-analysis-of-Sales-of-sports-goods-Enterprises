import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


data = pd.read_csv("C:\\Users\\Aitong\\Desktop\\临时文件\\长风杯\\体育用品企业销售数据集\\订单表.csv")
data['订单日期_time'] = pd.to_datetime(data['订单日期_time'])
df_2015_2016 = data[data['订单日期_time'] >= '2015-07-01']
margin_categories = ['西南区','西北区' ,'东南区', '东北区', '中部', '中国香港', '中国台湾' ,'中国澳门' ,'新加坡', '韩国']


for margin_category in margin_categories:
    filtered_data = df_2015_2016[df_2015_2016['销售大区'] == margin_category]


    filtered_purchases = filtered_data.groupby('客户ID').apply(lambda x: list(zip(x['产品ID'], x['产品名称'])))
    print(filtered_purchases)
    # 冗余
    filtered_transactions = filtered_purchases.apply(lambda x: [item for item in x])
    # print(filtered_transactions)

    te = TransactionEncoder()
    filtered_te_ary = te.fit(filtered_transactions).transform(filtered_transactions)
    filtered_df_trans = pd.DataFrame(filtered_te_ary, columns=te.columns_)
    # 生成频繁项集
    filtered_frequent_itemsets = apriori(filtered_df_trans, min_support=0.01, use_colnames=True)
    print(f"\n{margin_category} 频繁项集：")
    print(filtered_frequent_itemsets)
    filtered_frequent_itemsets_sorted =filtered_frequent_itemsets.sort_values(by=["support"], ascending=[False])

    # 生成关联规则
    filtered_rules = association_rules(filtered_frequent_itemsets, metric="confidence", min_threshold=0.5)
    filtered_rules_sorted = filtered_rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
    print(f"\n{margin_category} 关联规则：")
    print(filtered_rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    rules_filename = f"{margin_category}_association_rules.csv"
    filtered_rules_sorted.head(1000)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv(rules_filename,index=False,encoding='gbk')
    result_filename2 =f"{margin_category}_frequent_itemsets.csv"
    filtered_frequent_itemsets_sorted.head(1000).to_csv(result_filename2,index =False,encoding='gbk')

