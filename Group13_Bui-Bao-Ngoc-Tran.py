print("\n-- 3.1.3. Đọc dữ liệu từ HDFS --")

df = spark.read\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .csv("hdfs://localhost:9000//bigdata/group13/product_sales_dataset_final.csv")

print("\n-- Kết quả đọc dữ liệu ban đầu từ HDFS --")
df.printSchema()
df.show(10)
print("Total rows:", df.count())

print("\n-- 2.2. Xem trước 10 dòng dữ liệu đầu tiên --")

df.show(10, truncate=False)


print("\n-- 2.4. Kiểm tra quy mô dữ liệu và mức độ phù hợp với yêu cầu dự án --")

n_rows = df.count()
n_cols = len(df.columns)
print(f"Số dòng: {n_rows:,}")
print(f"Số cột: {n_cols}")
print(f"Điều kiện lớn hơn 100000 dòng: {n_rows > 100000}")
print(f"Điều kiện lớn hơn 10 cột: {n_cols > 10}")


print("\n-- 2.3. Chuẩn hóa tên cột về dạng lower_snake_case --")

import re

def to_snake_case(name: str) -> str:
    name = name.strip()                       # remove leading/trailing spaces
    name = re.sub(r"\s+", "_", name)          # spaces -> underscore
    return name.lower()

rename_map = {c: to_snake_case(c) for c in df.columns}
df = df.toDF(*[rename_map[c] for c in df.columns])

print("Tên các cột đã làm sạch:")
print(df.columns)


print("\n-- Đăng ký bảng tạm sales_raw phục vụ truy vấn Spark SQL --")

df.createOrReplaceTempView("sales_raw")
spark.sql("SELECT COUNT(*) AS row_count FROM sales_raw").show()


print("\n-- 2.5. Kết quả kiểm tra schema sau chuẩn hóa tên cột --")

df.printSchema()


print("\n-- 2.5. Kết quả kiểm tra giá trị thiếu --")

null_counts = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
])
null_counts.show(truncate=False)


print("\n-- 2.5. Kết quả kiểm tra dòng trùng lặp và order_id trùng lặp --")

total_rows = df.count()
distinct_rows = df.distinct().count()
print(f"Tổng số dòng: {total_rows}")
print(f"Tổng số dòng khác biệt: {distinct_rows}")
print(f"Tổng số dòng bị lặp: {total_rows - distinct_rows}")

total_order_ids = df.select("order_id").count()
distinct_order_ids = df.select("order_id").distinct().count()
print(f"Tổng số dòng order_id: {total_order_ids}")
print(f"Tổng số dòng order_id khác biệt: {distinct_order_ids}")
print(f"Tổng số dòng order_id bị lặp: {total_order_ids - distinct_order_ids}")


print("\n-- 2.6. Kết quả chuyển đổi kiểu dữ liệu và tạo biến thời gian --")

df = (
    df
    .withColumn("order_id", F.col("order_id").cast("int"))
    .withColumn("quantity", F.col("quantity").cast("int"))
    .withColumn("unit_price", F.col("unit_price").cast("double"))
    .withColumn("revenue", F.col("revenue").cast("double"))
    .withColumn("profit", F.col("profit").cast("double"))
    .withColumn("order_date_parsed", F.to_date(F.col("order_date"), "MM-dd-yy"))
    .withColumn("order_year", F.year("order_date_parsed"))
    .withColumn("order_month", F.month("order_date_parsed"))
    .withColumn("order_year_month", F.date_format("order_date_parsed", "yyyy-MM"))
)

df.select(
    "order_id", "order_date", "order_date_parsed",
    "order_year", "order_month", "order_year_month"
).show(10, truncate=False)

print("Các hàng chứa ngày tháng chưa được xử lý:", df.filter(F.col("order_date_parsed").isNull()).count())


print("\n-- 2.5. Kết quả kiểm tra khoảng giá trị của các biến định lượng --")

df.select(
    F.min("quantity").alias("min_quantity"), F.max("quantity").alias("max_quantity"),
    F.min("unit_price").alias("min_unit_price"), F.max("unit_price").alias("max_unit_price"),
    F.min("revenue").alias("min_revenue"), F.max("revenue").alias("max_revenue"),
    F.min("profit").alias("min_profit"), F.max("profit").alias("max_profit"),
).show()


print("\n-- 2.5. Kết quả kiểm tra logic kinh doanh giữa doanh thu, lợi nhuận và biên lợi nhuận --")

# Revenue should equal quantity * unit_price (sanity check; dataset has no discount column)
df_check = df.withColumn("calc_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
mismatch_count = df_check.filter(F.abs(F.col("calc_revenue") - F.col("revenue")) > 0.01).count()
print(f"Dòng có revenue != quantity * unit_price: {mismatch_count}")


# Profit margin sanity check: profit should be less than revenue, and margin should be in a plausible range (0-100%)
df_margin = df.withColumn("profit_margin_pct", F.round(F.col("profit") / F.col("revenue") * 100, 2))
df_margin.select(
    F.min("profit_margin_pct").alias("min_margin_pct"),
    F.max("profit_margin_pct").alias("max_margin_pct"),
    F.sum((F.col("profit") > F.col("revenue")).cast("int")).alias("profit_exceeds_revenue_rows"),
).show()


print("\n-- 2.5. Kết quả kiểm tra tính nhất quán giữa sản phẩm và danh mục --")

product_category_counts = (
    df.groupBy("product_name")
    .agg(F.countDistinct("category").alias("distinct_categories"))
    .filter(F.col("distinct_categories") > 1)
)
product_category_counts.show(truncate=False)

df.filter(
    F.col("product_name").isin("KitchenAid Mixer", "Instant Pot")
).select(
    "product_name", "category", "sub_category"
).distinct().show(truncate=False)


print("\n-- 4.2. Kết quả phân phối dữ liệu theo danh mục, vùng và tiểu danh mục --")

df.groupBy("category").count().orderBy(F.desc("count")).show()
df.groupBy("region").count().orderBy(F.desc("count")).show()
df.groupBy("sub_category").count().orderBy(F.desc("count")).show(20)


print("\n-- 2.5. Kết quả tạo df_clean và bảng tạm sales_clean --")

rows_before = df.count()

df_clean = (
    df
    .dropDuplicates()
    .dropna(subset=[
        "order_id", "order_date_parsed", "category", "sub_category", "product_name",
        "region", "state", "quantity", "unit_price", "revenue", "profit"
    ])
    .filter(F.col("quantity") > 0)
    .filter(F.col("unit_price") > 0)
    .filter(F.col("revenue") >= 0)
    .filter(F.col("profit") >= 0)
    .withColumn("profit_margin_pct", F.round(F.col("profit") / F.col("revenue") * 100, 2))
)

rows_after = df_clean.count()
removed = rows_before - rows_after
removal_pct = round(removed / rows_before * 100, 2)

print(f"Rows before cleaning: {rows_before:,}")
print(f"Rows after cleaning:  {rows_after:,}")
print(f"Rows removed:         {removed:,}")
print(f"Removal percentage:   {removal_pct}%")

df_clean.createOrReplaceTempView("sales_clean")
spark.sql("SELECT COUNT(*) AS row_count FROM sales_clean").show()

# Keep a DataFrame alias for Spark MLlib sections
sales_clean_df = df_clean


print("\n-- 5.1. Query 1: Xếp hạng doanh thu, lợi nhuận và biên lợi nhuận theo danh mục --")

query1 = spark.sql("""
    WITH category_agg AS (
        SELECT
            category,
            COUNT(*) AS order_count,
            SUM(quantity) AS total_quantity,
            SUM(revenue) AS total_revenue_raw,
            SUM(profit) AS total_profit_raw,
            SUM(profit) / SUM(revenue) * 100 AS profit_margin_raw
        FROM sales_clean
        GROUP BY category
    ),
    category_ranked AS (
        SELECT
            category,
            order_count,
            total_quantity,
            total_revenue_raw,
            total_profit_raw,
            profit_margin_raw,
            total_revenue_raw / SUM(total_revenue_raw) OVER () * 100 AS revenue_share_raw,
            total_profit_raw / SUM(total_profit_raw) OVER () * 100 AS profit_share_raw,
            RANK() OVER (ORDER BY total_revenue_raw DESC) AS revenue_rank,
            RANK() OVER (ORDER BY total_profit_raw DESC) AS profit_rank,
            RANK() OVER (ORDER BY profit_margin_raw DESC) AS margin_rank
        FROM category_agg
    )
    SELECT
        category,
        order_count,
        total_quantity,
        ROUND(total_revenue_raw, 2) AS total_revenue,
        ROUND(revenue_share_raw, 2) AS revenue_share_pct,
        ROUND(total_profit_raw, 2) AS total_profit,
        ROUND(profit_share_raw, 2) AS profit_share_pct,
        ROUND(profit_margin_raw, 2) AS profit_margin_pct,
        revenue_rank,
        profit_rank,
        margin_rank
    FROM category_ranked
    ORDER BY revenue_rank
""")

query1.show(truncate=False)


print("\n-- 5.6. Query 6: Hiệu quả kinh doanh theo phân khúc giá --")

query6 = spark.sql("""
    SELECT
        CASE
            WHEN unit_price < 150 THEN 'Budget (<150)'
            WHEN unit_price < 550 THEN 'Mid-Range (150-550)'
            ELSE 'Premium (550+)'
        END AS price_tier,
        COUNT(*) AS order_count,
        SUM(quantity) AS total_quantity,
        ROUND(SUM(revenue), 2) AS total_revenue,
        ROUND(SUM(profit), 2) AS total_profit,
        ROUND(SUM(profit) / SUM(revenue) * 100, 2) AS profit_margin_pct
    FROM sales_clean
    GROUP BY
        CASE
            WHEN unit_price < 150 THEN 'Budget (<150)'
            WHEN unit_price < 550 THEN 'Mid-Range (150-550)'
            ELSE 'Premium (550+)'
        END
    ORDER BY total_revenue DESC
""")
query6.show()


print("\n-- 5.8. Query 8: Xu hướng doanh thu, lợi nhuận và tăng trưởng theo tháng --")

query8 = spark.sql("""
    WITH monthly AS (
        SELECT
            order_year_month,
            ROUND(SUM(revenue), 2) AS total_revenue,
            ROUND(SUM(profit), 2) AS total_profit
        FROM sales_clean
        GROUP BY order_year_month
    )
    SELECT
        order_year_month,
        total_revenue,
        total_profit,
        LAG(total_revenue) OVER (ORDER BY order_year_month) AS prev_month_revenue,
        ROUND(
            (total_revenue - LAG(total_revenue) OVER (ORDER BY order_year_month))
            / LAG(total_revenue) OVER (ORDER BY order_year_month) * 100, 2
        ) AS mom_growth_pct
    FROM monthly
    ORDER BY order_year_month
""")
query8.show(24)


print("\n-- 5.9. Query 9: Tổ hợp vùng và danh mục có giá trị cao nhất --")

query9 = spark.sql("""
    SELECT
        region,
        category,
        ROUND(SUM(revenue), 2) AS total_revenue,
        ROUND(SUM(profit), 2) AS total_profit,
        SUM(quantity) AS total_quantity,
        RANK() OVER (PARTITION BY region ORDER BY SUM(revenue) DESC) AS rank_in_region
    FROM sales_clean
    GROUP BY region, category
    ORDER BY region, rank_in_region
""")
query9.show(20)


print("\n-- 6.1. Spark ML: Mã hóa biến phân loại bằng StringIndexer và OneHotEncoder --")

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# StringIndexer converts each categorical column into a numeric index
category_indexer = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep")
sub_category_indexer = StringIndexer(inputCol="sub_category", outputCol="sub_category_idx", handleInvalid="keep")
region_indexer = StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep")

# OneHotEncoder converts each numeric index into a one-hot vector
encoder = OneHotEncoder(
    inputCols=["category_idx", "sub_category_idx", "region_idx"],
    outputCols=["category_vec", "sub_category_vec", "region_vec"]
)


print("\n-- 6.1. Spark ML: Tạo vector đặc trưng và chia tập train/test --")

assembler = VectorAssembler(
    inputCols=["quantity", "unit_price", "revenue", "category_vec", "sub_category_vec", "region_vec"],
    outputCol="features"
)

train_df, test_df = sales_clean_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training rows: {train_df.count()}")
print(f"Test rows:     {test_df.count()}")


print("\n-- 6.1. Spark ML: Huấn luyện Linear Regression và Random Forest Regressor --")

lr = LinearRegression(featuresCol="features", labelCol="profit")
rf = RandomForestRegressor(featuresCol="features", labelCol="profit", numTrees=100, seed=42)

pipeline_lr = Pipeline(stages=[category_indexer, sub_category_indexer, region_indexer, encoder, assembler, lr])
pipeline_rf = Pipeline(stages=[category_indexer, sub_category_indexer, region_indexer, encoder, assembler, rf])

model_lr = pipeline_lr.fit(train_df)
model_rf = pipeline_rf.fit(train_df)

preds_lr = model_lr.transform(test_df)
preds_rf = model_rf.transform(test_df)


print("\n-- 6.1. Kết quả đánh giá mô hình dự đoán lợi nhuận --")

evaluator_rmse = RegressionEvaluator(labelCol="profit", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="profit", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="profit", predictionCol="prediction", metricName="r2")

for name, preds in [("LinearRegression", preds_lr), ("RandomForestRegressor", preds_rf)]:
    rmse = evaluator_rmse.evaluate(preds)
    mae = evaluator_mae.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)
    print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")


print("\n-- 6.1. Kết quả mức độ quan trọng của biến trong Random Forest --")

rf_model = model_rf.stages[-1]
importances = rf_model.featureImportances.toArray()

# Map encoded feature positions back to readable names using the fitted
# StringIndexer / OneHotEncoder stages from the pipeline
encoder_model = model_rf.stages[3]
category_labels = model_rf.stages[0].labels
sub_category_labels = model_rf.stages[1].labels
region_labels = model_rf.stages[2].labels

# Get feature names from the assembled features metadata
# This method is more robust as it reflects the actual features passed to the model
feature_attrs_dict = preds_rf.schema["features"].metadata["ml_attr"]["attrs"]

feature_names = []
for attr_type in feature_attrs_dict:
    for attr in feature_attrs_dict[attr_type]:
        feature_names.append(attr["name"])

import pandas as pd
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print(importance_df.head(10).to_string(index=False))


