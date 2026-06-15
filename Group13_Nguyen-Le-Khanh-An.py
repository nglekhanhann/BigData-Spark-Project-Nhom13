#Phân tích dữ liệu bán hàng quy mô lớn bằng Apache Hadoop và Apache Spark: Ứng dụng Spark SQL và Spark ML trong đánh giá hiệu quả kinh doanh

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

print("\n-- 3.4. Kiểm tra PySpark và khởi tạo SparkSession --")

spark = (
    SparkSession.builder
    .appName("group13")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

print("\n-- Thông tin phiên bản Spark --")
print("Spark version:", spark.version)
spark.sparkContext.setLogLevel("WARN")


print("\n-- 4.1. Kết quả thống kê mô tả các biến định lượng --")

df_clean.select("quantity", "unit_price", "revenue", "profit", "profit_margin_pct").summary(
    "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
).show()


print("\n-- 4.2. Kết quả phân phối các biến phân loại --")

for col in ["category", "sub_category", "region", "state"]:
    print(f"--- {col} ---")
    df_clean.groupBy(col).count().orderBy(F.desc("count")).show(10, truncate=False)


print("\n-- 4.3. Kết quả tổng quan doanh thu, lợi nhuận và quy mô bán hàng --")

df_clean.agg(
    F.sum("revenue").alias("total_revenue"),
    F.sum("profit").alias("total_profit"),
    F.round(F.sum("profit") / F.sum("revenue") * 100, 2).alias("overall_margin_pct"),
    F.count("*").alias("total_orders"),
    F.sum("quantity").alias("total_units_sold"),
).show()


print("\n-- 4.4. Kết quả tổng quan hiệu quả kinh doanh theo vùng --")

region_overview = (
    df_clean.groupBy("region")
    .agg(
        F.sum("revenue").alias("total_revenue"),
        F.sum("profit").alias("total_profit"),
        F.count("*").alias("order_count"),
    )
    .withColumn("profit_margin_pct", F.round(F.col("total_profit") / F.col("total_revenue") * 100, 2))
    .orderBy(F.desc("total_revenue"))
)
region_overview.show()


print("\n-- 4.5. Kết quả tổng quan hiệu quả kinh doanh theo danh mục sản phẩm --")

category_overview = (
    df_clean.groupBy("category")
    .agg(
        F.sum("revenue").alias("total_revenue"),
        F.sum("profit").alias("total_profit"),
        F.count("*").alias("order_count"),
    )
    .withColumn("profit_margin_pct", F.round(F.col("total_profit") / F.col("total_revenue") * 100, 2))
    .orderBy(F.desc("total_revenue"))
)
category_overview.show()


print("\n-- 4.6. Kết quả phân tích hiệu quả theo phân khúc giá --")

df_priced = df_clean.withColumn(
    "price_tier",
    F.when(F.col("unit_price") < 150, "Budget (<150)")
     .when(F.col("unit_price") < 550, "Mid-Range (150-550)")
     .otherwise("Premium (550+)")
)

price_tier_overview = (
    df_priced.groupBy("price_tier")
    .agg(
        F.count("*").alias("order_count"),
        F.sum("quantity").alias("total_quantity"),
        F.sum("revenue").alias("total_revenue"),
        F.sum("profit").alias("total_profit"),
    )
    .withColumn("profit_margin_pct", F.round(F.col("total_profit") / F.col("total_revenue") * 100, 2))
    .orderBy(F.desc("total_revenue"))
)
price_tier_overview.show()


print("\n-- 4.7. Kết quả xu hướng doanh thu và lợi nhuận theo tháng --")

monthly_overview = (
    df_clean.groupBy("order_year_month")
    .agg(F.sum("revenue").alias("total_revenue"), F.sum("profit").alias("total_profit"))
    .orderBy("order_year_month")
)
monthly_overview.show(24)


print("\n-- 4.8. Biểu đồ 1: Tổng doanh thu theo danh mục sản phẩm --")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs("eda_charts", exist_ok=True)

cat_rev_pd = category_overview.select("category", "total_revenue").toPandas() \
    .sort_values("total_revenue", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(cat_rev_pd["category"], cat_rev_pd["total_revenue"])
ax.set_title("Tổng doanh thu theo phân loại sản phẩm")
ax.set_ylabel("Tổng doanh thu (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("eda_charts/01_revenue_by_category.png")
plt.show()
plt.close()


print("\n-- 4.8. Biểu đồ 2: Tổng lợi nhuận theo danh mục sản phẩm --")

cat_profit_pd = category_overview.select("category", "total_profit").toPandas() \
    .sort_values("total_profit", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(cat_profit_pd["category"], cat_profit_pd["total_profit"])
ax.set_title("Tổng lợi nhuận theo danh mục sản phẩm")
ax.set_ylabel("Tổng lợi nhuận (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("eda_charts/02_profit_by_category.png")
plt.show()
plt.close()


print("\n-- 4.8. Biểu đồ 3: Tổng doanh thu theo vùng --")

region_rev_pd = region_overview.select("region", "total_revenue").toPandas() \
    .sort_values("total_revenue", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(region_rev_pd["region"], region_rev_pd["total_revenue"])
ax.set_title("Tổng doanh thu theo vùng")
ax.set_ylabel("Tổng doanh thu (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig("eda_charts/03_revenue_by_region.png")
plt.show()
plt.close()


print("\n-- 4.8. Biểu đồ 4: Xu hướng doanh thu theo tháng --")

monthly_pd = monthly_overview.select("order_year_month", "total_revenue").toPandas() \
    .sort_values("order_year_month")

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(monthly_pd["order_year_month"], monthly_pd["total_revenue"], marker="o", linewidth=2)
ax.set_title("Xu hướng doanh thu theo thời gian (tháng) (2023-2024)")
ax.set_ylabel("Tổng doanh thu (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.xticks(rotation=60, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("eda_charts/04_monthly_revenue_trend.png")
plt.show()
plt.close()


print("\n-- 4.8. Biểu đồ 5: Top 10 sản phẩm theo doanh thu --")

top10_pd = (
    df_clean.groupBy("product_name")
    .agg(F.sum("revenue").alias("total_revenue"))
    .orderBy(F.desc("total_revenue"))
    .limit(10)
    .toPandas()
    .sort_values("total_revenue")
)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(top10_pd["product_name"], top10_pd["total_revenue"])
ax.set_title("Top 10 sản phẩm theo doanh thu")
ax.set_xlabel("Tổng doanh thu (USD)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig("eda_charts/05_top10_products_revenue.png")
plt.show()
plt.close()


print("\n-- 4.8. Biểu đồ 6: Biên lợi nhuận theo danh mục sản phẩm --")

cat_margin_pd = category_overview.select("category", "profit_margin_pct").toPandas() \
    .sort_values("profit_margin_pct", ascending=False)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(cat_margin_pd["category"], cat_margin_pd["profit_margin_pct"])
ax.set_title("Lợi nhuận biên (%) theo phân loại sản phẩm")
ax.set_ylabel("Lợi nhuận biên (%)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("eda_charts/06_avg_margin_by_category.png")
plt.show()
plt.close()


print("\n-- 4.8. Biểu đồ 7: Doanh thu theo phân khúc giá --")

tier_order = ["Budget (<150)", "Mid-Range (150-550)", "Premium (550+)"]
tier_pd = price_tier_overview.select("price_tier", "total_revenue").toPandas()
tier_pd["price_tier"] = pd.Categorical(tier_pd["price_tier"], categories=tier_order, ordered=True)
tier_pd = tier_pd.sort_values("price_tier")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(tier_pd["price_tier"], tier_pd["total_revenue"])
ax.set_title("Tổng doanh thu theo phân khúc giá")
ax.set_ylabel("Tổng doanh thu (USD)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig("eda_charts/07_revenue_by_price_tier.png")
plt.show()
plt.close()


print("\n-- 4.8. Biểu đồ 8: Sản lượng bán theo vùng --")

region_qty_pd = (
    df_clean.groupBy("region")
    .agg(F.sum("quantity").alias("total_quantity"))
    .orderBy(F.desc("total_quantity"))
    .toPandas()
)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(region_qty_pd["region"], region_qty_pd["total_quantity"])
ax.set_title("Tổng sản lượng bán được theo vùng")
ax.set_ylabel("Tổng sản lượng (đơn vị)")
plt.tight_layout()
plt.savefig("eda_charts/08_quantity_by_region.png")
plt.show()
plt.close()


print("\n-- 6.2. K-Means: Tạo vector đặc trưng và chuẩn hóa dữ liệu --")

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

cluster_features = ["revenue", "profit", "quantity", "unit_price", "profit_margin_pct"]

cluster_assembler = VectorAssembler(inputCols=cluster_features, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

assembled_df = cluster_assembler.transform(sales_clean_df)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)


print("\n-- 6.2. K-Means: Kết quả Elbow Method với K từ 2 đến 8 --")

inertias = {}
for k in range(2, 9):
    kmeans = KMeans(featuresCol="features", k=k, seed=42)
    model = kmeans.fit(scaled_df)
    inertias[k] = model.summary.trainingCost
    print(f"k={k}, WCSS={inertias[k]:.2f}")


print("\n-- 6.2. K-Means: Kết quả huấn luyện mô hình cuối cùng với K = 4 --")

k_final = 4
kmeans_final = KMeans(featuresCol="features", k=k_final, seed=42)
model_final = kmeans_final.fit(scaled_df)
clustered_df = model_final.transform(scaled_df)
clustered_df.createOrReplaceTempView("sales_clustered")

clustered_df.groupBy("prediction").count().orderBy("prediction").show()


print("\n-- 6.2. K-Means: Kết quả hồ sơ cụm --")

cluster_profile = spark.sql("""
    SELECT
        prediction AS cluster,
        COUNT(*) AS order_count,
        ROUND(AVG(revenue), 2) AS avg_revenue,
        ROUND(AVG(profit), 2) AS avg_profit,
        ROUND(AVG(quantity), 2) AS avg_quantity,
        ROUND(AVG(unit_price), 2) AS avg_unit_price,
        ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct
    FROM sales_clustered
    GROUP BY prediction
    ORDER BY avg_revenue DESC
""")
cluster_profile.show()

spark.sql("""
    SELECT prediction AS cluster, category, COUNT(*) AS cnt
    FROM sales_clustered
    GROUP BY prediction, category
    ORDER BY prediction, cnt DESC
""").show(20)


print("\n-- 6.2. K-Means: Trực quan hóa phân cụm --")

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs("eda_charts", exist_ok=True)

# Use transaction-level numeric features
cluster_features = ["revenue", "profit", "quantity", "unit_price", "profit_margin_pct"]

cluster_input_df = df_clean.dropna(subset=cluster_features)

assembler = VectorAssembler(
    inputCols=cluster_features,
    outputCol="features_raw"
)

assembled_df = assembler.transform(cluster_input_df)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

print("Rows used for clustering:", scaled_df.count())
scaled_df.select(cluster_features + ["features"]).show(5, truncate=False)


print("\n-- 6.2. K-Means: Biểu đồ Elbow Method --")

# %%
inertias = {}

for k in range(2, 9):
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="prediction",
        k=k,
        seed=42
    )
    model = kmeans.fit(scaled_df)
    inertias[k] = model.summary.trainingCost
    print(f"k={k}, WCSS={inertias[k]:.2f}")

elbow_pd = pd.DataFrame({
    "k": list(inertias.keys()),
    "wcss": list(inertias.values())
})

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(elbow_pd["k"], elbow_pd["wcss"], marker="o")
ax.set_title("Elbow Method for K-Means Clustering")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("WCSS")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eda_charts/09_kmeans_elbow_chart.png", dpi=150)
plt.show()
plt.close()


print("\n-- 6.2. K-Means: Kết quả huấn luyện K = 4 và bảng hồ sơ cụm --")

# %%
k_final = 4

kmeans_final = KMeans(
    featuresCol="features",
    predictionCol="prediction",
    k=k_final,
    seed=42
)

model_final = kmeans_final.fit(scaled_df)
clustered_df = model_final.transform(scaled_df)

clustered_df.createOrReplaceTempView("sales_clustered")

cluster_profile = (
    clustered_df
    .groupBy("prediction")
    .agg(
        F.count("*").alias("order_count"),
        F.round(F.avg("revenue"), 2).alias("avg_revenue"),
        F.round(F.avg("profit"), 2).alias("avg_profit"),
        F.round(F.avg("quantity"), 2).alias("avg_quantity"),
        F.round(F.avg("unit_price"), 2).alias("avg_unit_price"),
        F.round(F.avg("profit_margin_pct"), 2).alias("avg_profit_margin_pct")
    )
    .orderBy("prediction")
)

cluster_profile.show(truncate=False)


print("\n-- 6.2. K-Means: Kết quả bổ sung danh mục chiếm ưu thế cho từng cụm --")

from pyspark.sql.window import Window

cluster_category_counts = (
    clustered_df
    .groupBy("prediction", "category")
    .agg(F.count("*").alias("category_count"))
)

w = Window.partitionBy("prediction").orderBy(F.desc("category_count"))

dominant_category = (
    cluster_category_counts
    .withColumn("rank", F.row_number().over(w))
    .filter(F.col("rank") == 1)
    .select(
        F.col("prediction"),
        F.col("category").alias("dominant_category"),
        F.col("category_count").alias("dominant_category_count")
    )
)

cluster_profile_with_category = (
    cluster_profile
    .join(dominant_category, cluster_profile["prediction"] == dominant_category["prediction"], how="left")
    .orderBy(cluster_profile["prediction"])
)

cluster_profile_with_category.show(truncate=False)


print("\n-- 6.2. K-Means: Biểu đồ PCA trực quan hóa cụm --")

# %%
from pyspark.ml.feature import PCA
from pyspark.ml.functions import vector_to_array

pca = PCA(
    k=2,
    inputCol="features",
    outputCol="pca_features"
)

pca_model = pca.fit(clustered_df)
pca_df = pca_model.transform(clustered_df)

pca_plot_df = (
    pca_df
    .withColumn("pca_array", vector_to_array("pca_features"))
    .select(
        F.col("pca_array")[0].alias("pc1"),
        F.col("pca_array")[1].alias("pc2"),
        F.col("prediction").alias("cluster"),
        "revenue",
        "profit",
        "profit_margin_pct"
    )
)


pca_plot_pd = (
    pca_plot_df
    .sample(withReplacement=False, fraction=0.05, seed=42)
    .toPandas()
)

if len(pca_plot_pd) > 10000:
    pca_plot_pd = pca_plot_pd.sample(n=10000, random_state=42)

fig, ax = plt.subplots(figsize=(7, 5))

scatter = ax.scatter(
    pca_plot_pd["pc1"],
    pca_plot_pd["pc2"],
    c=pca_plot_pd["cluster"],
    alpha=0.45,
    s=12
)

ax.set_title("K-Means Transaction Clusters Visualized with PCA")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.grid(True, alpha=0.3)

legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)

plt.tight_layout()
plt.savefig("eda_charts/13_kmeans_pca_cluster_scatter.png", dpi=150)
plt.show()
plt.close()

print("Explained variance ratio:")
print(pca_model.explainedVariance)
