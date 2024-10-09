WITH eth_l0_interactors AS (
    SELECT DISTINCT sender_wallet AS addr
    FROM external.layerzero.fact_transactions_snapshot
    WHERE source_chain = 'Ethereum'
    QUALIFY RANK() OVER (ORDER BY sender_wallet) BETWEEN 300000 AND 400000
),
node_metrics AS (
	SELECT ei.addr,
	COALESCE(COUNT(distinct t1.from_address),0) as in_degree,
	COALESCE(COUNT(distinct t2.to_address),0) as out_degree
	FROM eth_l0_interactors ei 
	LEFT JOIN  ethereum.core.fact_transactions t1 ON ei.addr = t1.to_address 
	LEFT JOIN  ethereum.core.fact_transactions t2 ON ei.addr = t2.from_address
    GROUP BY ei.addr
),
neighborhood AS (
    -- Find all transactions involving addresses in Set A and their neighbors (directed edges)
    SELECT DISTINCT
        t1.from_address AS addr
    FROM ethereum.core.fact_transactions t1
    WHERE t1.block_timestamp <= TO_TIMESTAMP_NTZ('2024-05-01 00:00:00')
    AND t1.to_address IN (SELECT addr FROM eth_l0_interactors)

    UNION

    SELECT DISTINCT
        t1.to_address AS addr
    FROM ethereum.core.fact_transactions t1
    WHERE t1.block_timestamp <= TO_TIMESTAMP_NTZ('2024-05-01 00:00:00')
    AND t1.from_address IN (SELECT addr FROM eth_l0_interactors)

    UNION

    SELECT addr FROM eth_l0_interactors
),
weights AS (
    SELECT 
        t.from_address,
        t.to_address, 
        SUM(t.value + t.tx_fee) AS edge_weight
    FROM 
        ethereum.core.fact_transactions t 
    JOIN 
        neighborhood from_addrs
    JOIN 
        neighborhood to_addrs 
    ON 
        from_addrs.addr = t.from_address AND to_addrs.addr = t.to_address
    WHERE 
        t.block_timestamp <= TO_TIMESTAMP_NTZ('2024-05-01 00:00:00')
    GROUP BY 
        t.from_address, t.to_address
),
global_max_weight AS (
    SELECT MAX(edge_weight) AS max_weight
    FROM weights
),
triangles AS (
    SELECT 
        ei.addr as l0_interactor,  -- the interacting address
        edge1.to_address AS addr2,
        edge2.to_address AS addr3,
        edge1.edge_weight / gmw.max_weight AS weight1,
        edge2.edge_weight / gmw.max_weight AS weight2,
        edge3.edge_weight / gmw.max_weight AS weight3
    FROM 
        eth_l0_interactors ei
    JOIN 
        weights edge1 ON edge1.from_address = ei.addr AND edge1.to_address != ei.addr
    JOIN 
        weights edge2 ON edge1.to_address = edge2.from_address AND edge2.to_address != ei.addr AND edge2.to_address != edge2.from_address
    JOIN
        weights edge3 ON edge2.to_address = edge3.from_address AND edge3.to_address = ei.addr
    CROSS JOIN
        global_max_weight gmw
)
    SELECT 
        l0_interactor AS addr,
        COALESCE(SUM(POW(weight1 * weight2 * weight3, 1/3)),0) AS weight_sum, 
        COALESCE(COUNT(t.*),0) as n_triangles,
        COALESCE(COUNT(DISTINCT addr2),0) * COALESCE(COUNT(DISTINCT addr3),0) AS possible_triangles,
        CASE
          WHEN COALESCE(COUNT(DISTINCT addr2),0) * COALESCE(COUNT(DISTINCT addr3),0) = 0 THEN 0
          ELSE COALESCE(SUM(POW(weight1 * weight2 * weight3, 1/3)), 0) / (COALESCE(COUNT(DISTINCT addr2),0) * COALESCE(COUNT(DISTINCT addr3),0))
        END AS clustering_coeff,
        ROW_NUMBER() OVER (ORDER BY l0_interactor) AS rank
    FROM 
        eth_l0_interactors ei LEFT JOIN triangles t on ei.addr = t.l0_interactor
    GROUP BY l0_interactor
