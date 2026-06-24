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

print("\n-- 3.1.3. Đọc dữ liệu từ HDFS --")

df = spark.read\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .csv("hdfs://localhost:9000//bigdata/group13/product_sales_dataset_final.csv")

print("\n-- Kết quả đọc dữ liệu ban đầu từ HDFS --")
df.printSchema()
df.show(10)
print("Total rows:", df.count())

print("\n-- Thông tin phiên bản Spark --")
print("Spark version:", spark.version)
spark.sparkContext.setLogLevel("WARN")


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
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
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


df_check = df.withColumn("calc_revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))
mismatch_count = df_check.filter(F.abs(F.col("calc_revenue") - F.col("revenue")) > 0.01).count()
print(f"Dòng có revenue != quantity * unit_price: {mismatch_count}")



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

sales_clean_df = df_clean


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


print("\n-- 5.2. Query 2: Đánh giá sức mạnh thị trường theo vùng --")

query2 = spark.sql("""
    WITH region_agg AS (
        SELECT
            region,
            COUNT(*) AS order_count,
            SUM(quantity) AS total_quantity,
            SUM(revenue) AS total_revenue_raw,
            SUM(profit) AS total_profit_raw,
            SUM(profit) / SUM(revenue) * 100 AS profit_margin_raw,
            SUM(revenue) / COUNT(*) AS avg_order_value_raw,
            SUM(profit) / COUNT(*) AS profit_per_order_raw
        FROM sales_clean
        GROUP BY region
    ),
    region_ranked AS (
        SELECT
            region,
            order_count,
            total_quantity,
            total_revenue_raw,
            total_profit_raw,
            profit_margin_raw,
            avg_order_value_raw,
            profit_per_order_raw,
            total_revenue_raw / SUM(total_revenue_raw) OVER () * 100 AS revenue_share_raw,
            total_profit_raw / SUM(total_profit_raw) OVER () * 100 AS profit_share_raw,
            RANK() OVER (ORDER BY total_revenue_raw DESC) AS revenue_rank,
            RANK() OVER (ORDER BY total_profit_raw DESC) AS profit_rank,
            RANK() OVER (ORDER BY profit_margin_raw DESC) AS margin_rank
        FROM region_agg
    )
    SELECT
        region,
        order_count,
        total_quantity,
        ROUND(total_revenue_raw, 2) AS total_revenue,
        ROUND(revenue_share_raw, 2) AS revenue_share_pct,
        ROUND(total_profit_raw, 2) AS total_profit,
        ROUND(profit_share_raw, 2) AS profit_share_pct,
        ROUND(profit_margin_raw, 2) AS profit_margin_pct,
        ROUND(avg_order_value_raw, 2) AS avg_order_value,
        ROUND(profit_per_order_raw, 2) AS profit_per_order,
        revenue_rank,
        profit_rank,
        margin_rank
    FROM region_ranked
    ORDER BY revenue_rank
""")

query2.show(truncate=False)


print("\n-- 5.3. Query 3: Top 10 sản phẩm theo doanh thu và lợi nhuận --")

query3 = spark.sql("""
    WITH product_agg AS (
        SELECT
            product_name,
            category,
            SUM(quantity) AS total_quantity,
            SUM(revenue) AS total_revenue_raw,
            SUM(profit) AS total_profit_raw,
            SUM(profit) / SUM(revenue) * 100 AS profit_margin_raw
        FROM sales_clean
        GROUP BY product_name, category
    ),
    product_ranked AS (
        SELECT
            product_name,
            category,
            total_quantity,
            total_revenue_raw,
            total_profit_raw,
            profit_margin_raw,
            total_revenue_raw / SUM(total_revenue_raw) OVER () * 100 AS revenue_share_raw,
            total_profit_raw / SUM(total_profit_raw) OVER () * 100 AS profit_share_raw,
            RANK() OVER (ORDER BY total_revenue_raw DESC) AS revenue_rank,
            RANK() OVER (ORDER BY total_profit_raw DESC) AS profit_rank,
            RANK() OVER (ORDER BY profit_margin_raw DESC) AS margin_rank
        FROM product_agg
    )
    SELECT
        product_name,
        category,
        total_quantity,
        ROUND(total_revenue_raw, 2) AS total_revenue,
        ROUND(revenue_share_raw, 2) AS revenue_share_pct,
        ROUND(total_profit_raw, 2) AS total_profit,
        ROUND(profit_share_raw, 2) AS profit_share_pct,
        ROUND(profit_margin_raw, 2) AS profit_margin_pct,
        revenue_rank,
        profit_rank,
        margin_rank
    FROM product_ranked
    ORDER BY revenue_rank
    LIMIT 10
""")

query3.show(10, truncate=False)


print("\n-- 5.4. Query 4: Sản phẩm có sản lượng cao nhưng biên lợi nhuận thấp --")

query4 = spark.sql("""
    WITH product_agg AS (
        SELECT
            product_name,
            category,
            SUM(quantity) AS total_quantity,
            SUM(revenue) AS total_revenue_raw,
            SUM(profit) AS total_profit_raw,
            SUM(profit) / SUM(revenue) * 100 AS profit_margin_raw
        FROM sales_clean
        GROUP BY product_name, category
    ),
    benchmarks AS (
        SELECT
            AVG(total_quantity) AS avg_qty,
            AVG(profit_margin_raw) AS avg_margin
        FROM product_agg
    )
    SELECT
        p.product_name,
        p.category,
        p.total_quantity,
        ROUND(b.avg_qty, 2) AS avg_quantity_benchmark,
        ROUND(p.total_quantity - b.avg_qty, 2) AS quantity_above_benchmark,
        ROUND(p.total_revenue_raw, 2) AS total_revenue,
        ROUND(p.total_profit_raw, 2) AS total_profit,
        ROUND(p.profit_margin_raw, 2) AS profit_margin_pct,
        ROUND(b.avg_margin, 2) AS avg_margin_benchmark,
        ROUND(p.profit_margin_raw - b.avg_margin, 2) AS margin_gap_pct,
        CASE
            WHEN p.total_quantity > b.avg_qty
                 AND p.profit_margin_raw < b.avg_margin
                 AND p.total_revenue_raw >= 5000000
            THEN 'High revenue but margin pressure'
            WHEN p.total_quantity > b.avg_qty
                 AND p.profit_margin_raw < b.avg_margin
            THEN 'High volume but low margin'
            ELSE 'Monitor'
        END AS diagnosis
    FROM product_agg p
    CROSS JOIN benchmarks b
    WHERE p.total_quantity > b.avg_qty
      AND p.profit_margin_raw < b.avg_margin
    ORDER BY p.total_quantity DESC
""")

query4.show(truncate=False)


print("\n-- 5.5. Query 5: Xếp hạng biên lợi nhuận theo tiểu danh mục --")

query5 = spark.sql("""
    SELECT
        sub_category,
        category,
        ROUND(SUM(revenue), 2) AS total_revenue,
        ROUND(SUM(profit), 2) AS total_profit,
        ROUND(SUM(profit) / SUM(revenue) * 100, 2) AS profit_margin_pct,
        RANK() OVER (ORDER BY SUM(profit) / SUM(revenue) DESC) AS margin_rank
    FROM sales_clean
    GROUP BY sub_category, category
    ORDER BY profit_margin_pct DESC
    LIMIT 10
""")
query5.show(10, truncate=False)


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


print("\n-- 5.7. Query 7: Tiểu danh mục có tỷ trọng doanh thu cao nhưng biên lợi nhuận dưới trung bình --")


query7 = spark.sql("""
    WITH subcategory_agg AS (
        SELECT
            category,
            sub_category,
            COUNT(*) AS order_count,
            SUM(quantity) AS total_quantity,
            SUM(revenue) AS total_revenue_raw,
            SUM(profit) AS total_profit_raw,
            SUM(profit) / SUM(revenue) * 100 AS profit_margin_raw
        FROM sales_clean
        GROUP BY category, sub_category
    ),
    subcategory_with_share AS (
        SELECT
            category,
            sub_category,
            order_count,
            total_quantity,
            total_revenue_raw,
            total_profit_raw,
            profit_margin_raw,
            total_revenue_raw / SUM(total_revenue_raw) OVER () * 100 AS revenue_share_raw
        FROM subcategory_agg
    ),
    benchmarks AS (
        SELECT
            AVG(revenue_share_raw) AS avg_revenue_share,
            SUM(total_profit_raw) / SUM(total_revenue_raw) * 100 AS overall_margin
        FROM subcategory_with_share
    )
    SELECT
        s.category,
        s.sub_category,
        s.order_count,
        s.total_quantity,
        ROUND(s.total_revenue_raw, 2) AS total_revenue,
        ROUND(s.revenue_share_raw, 2) AS revenue_share_pct,
        ROUND(s.total_profit_raw, 2) AS total_profit,
        ROUND(s.profit_margin_raw, 2) AS profit_margin_pct,
        ROUND(b.avg_revenue_share, 2) AS avg_revenue_share_benchmark,
        ROUND(b.overall_margin, 2) AS overall_margin_benchmark,
        ROUND(s.profit_margin_raw - b.overall_margin, 2) AS margin_gap_pct,
        CASE
            WHEN s.revenue_share_raw > b.avg_revenue_share
                 AND s.profit_margin_raw < b.overall_margin
            THEN 'High revenue share but below-average margin'
            ELSE 'Monitor'
        END AS diagnosis
    FROM subcategory_with_share s
    CROSS JOIN benchmarks b
    WHERE s.revenue_share_raw > b.avg_revenue_share
      AND s.profit_margin_raw < b.overall_margin
    ORDER BY s.revenue_share_raw DESC, margin_gap_pct ASC
""")
query7.show(truncate=False)

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


print("\n-- 5.10. Query 10: Phân khúc giá trị theo bang --")

# QUERY 10: State-Level Value Segmentation using NTILE

query10 = spark.sql("""
    WITH state_agg AS (
        SELECT
            state,
            region,
            COUNT(*) AS order_count,
            SUM(quantity) AS total_quantity,
            SUM(revenue) AS total_revenue_raw,
            SUM(profit) AS total_profit_raw,
            SUM(profit) / SUM(revenue) * 100 AS profit_margin_raw,
            SUM(revenue) / COUNT(*) AS avg_order_value_raw,
            SUM(profit) / COUNT(*) AS profit_per_order_raw
        FROM sales_clean
        GROUP BY state, region
    ),
    state_segmented AS (
        SELECT
            *,
            NTILE(3) OVER (ORDER BY total_revenue_raw DESC) AS revenue_tier
        FROM state_agg
    )
    SELECT
        state,
        region,
        order_count,
        total_quantity,
        ROUND(total_revenue_raw, 2) AS total_revenue,
        ROUND(total_profit_raw, 2) AS total_profit,
        ROUND(profit_margin_raw, 2) AS profit_margin_pct,
        ROUND(avg_order_value_raw, 2) AS avg_order_value,
        ROUND(profit_per_order_raw, 2) AS profit_per_order,
        CASE
            WHEN revenue_tier = 1 THEN 'High-Value'
            WHEN revenue_tier = 2 THEN 'Mid-Value'
            ELSE 'Emerging'
        END AS value_segment
    FROM state_segmented
    ORDER BY total_revenue_raw DESC
""")

query10.show(truncate=False)


print("\n-- 5.11. Query 11: Sản phẩm ưu tiên tăng trưởng --")

query11 = spark.sql("""
    WITH product_agg AS (
        SELECT
            product_name,
            category,
            ROUND(SUM(revenue), 2) AS total_revenue,
            ROUND(SUM(profit), 2) AS total_profit,
            SUM(quantity) AS total_quantity,
            ROUND(SUM(profit) / SUM(revenue) * 100, 2) AS profit_margin_pct
        FROM sales_clean
        GROUP BY product_name, category
    ),
    quartiles AS (
        SELECT
            *,
            NTILE(4) OVER (ORDER BY total_revenue DESC) AS revenue_quartile,
            NTILE(4) OVER (ORDER BY profit_margin_pct DESC) AS margin_quartile
        FROM product_agg
    )
    SELECT
        product_name,
        category,
        total_revenue,
        total_profit,
        total_quantity,
        profit_margin_pct,
        revenue_quartile,
        margin_quartile,
        CASE
            WHEN revenue_quartile = 1 AND margin_quartile <= 2 THEN 'Priority: Scale Up'
            WHEN margin_quartile = 1 AND revenue_quartile >= 2 THEN 'Priority: Promote'
            ELSE 'Monitor'
        END AS priority_action
    FROM quartiles
    WHERE revenue_quartile = 1 OR margin_quartile = 1
    ORDER BY total_revenue DESC
""")
query11.show(20, truncate=False)


print("\n-- 5.12. Query 12: Sản phẩm cần hành động điều chỉnh --")

query12 = spark.sql("""
    WITH product_region_agg AS (
        SELECT
            product_name,
            category,
            region,
            SUM(quantity) AS total_quantity,
            SUM(revenue) AS total_revenue_raw,
            SUM(profit) AS total_profit_raw,
            SUM(profit) / SUM(revenue) * 100 AS profit_margin_raw
        FROM sales_clean
        GROUP BY product_name, category, region
    ),
    benchmarks AS (
        SELECT
            AVG(total_revenue_raw) AS avg_revenue_per_group,
            SUM(total_profit_raw) / SUM(total_revenue_raw) * 100 AS overall_margin
        FROM product_region_agg
    ),
    corrective_candidates AS (
        SELECT
            p.product_name,
            p.category,
            p.region,
            p.total_quantity,
            p.total_revenue_raw,
            p.total_profit_raw,
            p.profit_margin_raw,
            b.avg_revenue_per_group,
            b.overall_margin,
            p.total_revenue_raw - b.avg_revenue_per_group AS revenue_gap_raw,
            p.profit_margin_raw - b.overall_margin AS margin_gap_raw
        FROM product_region_agg p
        CROSS JOIN benchmarks b
        WHERE p.total_revenue_raw >= b.avg_revenue_per_group
          AND p.profit_margin_raw < b.overall_margin
    )
    SELECT
        product_name,
        category,
        region,
        total_quantity,
        ROUND(total_revenue_raw, 2) AS total_revenue,
        ROUND(avg_revenue_per_group, 2) AS avg_revenue_benchmark,
        ROUND(revenue_gap_raw, 2) AS revenue_gap,
        ROUND(total_profit_raw, 2) AS total_profit,
        ROUND(profit_margin_raw, 2) AS profit_margin_pct,
        ROUND(overall_margin, 2) AS overall_margin_benchmark,
        ROUND(margin_gap_raw, 2) AS margin_gap_pct,
        CASE
            WHEN total_revenue_raw >= avg_revenue_per_group
                 AND profit_margin_raw < overall_margin
                 AND margin_gap_raw <= -10
            THEN 'Urgent: improve margin'
            WHEN total_revenue_raw >= avg_revenue_per_group
                 AND profit_margin_raw < overall_margin
            THEN 'Corrective action: review pricing/cost'
            ELSE 'Monitor'
        END AS recommended_action
    FROM corrective_candidates
    ORDER BY total_revenue_raw DESC, margin_gap_raw ASC
""")

query12.show(truncate=False)


print("\n-- 6.1. Spark ML: Mã hóa biến phân loại bằng StringIndexer và OneHotEncoder --")

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline


category_indexer = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep")
sub_category_indexer = StringIndexer(inputCol="sub_category", outputCol="sub_category_idx", handleInvalid="keep")
region_indexer = StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep")


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

encoder_model = model_rf.stages[3]
category_labels = model_rf.stages[0].labels
sub_category_labels = model_rf.stages[1].labels
region_labels = model_rf.stages[2].labels


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
