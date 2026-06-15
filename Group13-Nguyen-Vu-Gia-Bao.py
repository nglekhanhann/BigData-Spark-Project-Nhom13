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


print("\n-- 5.7. Query 7: Tiểu danh mục có tỷ trọng doanh thu cao nhưng biên lợi nhuận dưới trung bình --")

# QUERY 7: High-Revenue-Share Sub-Categories with Below-Average Margin

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


