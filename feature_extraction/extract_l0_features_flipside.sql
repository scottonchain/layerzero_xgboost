WITH addresses AS (
  SELECT
    sender_wallet AS addr,
    -- Number of ETH - LayerZero interactions
    COUNT(DISTINCT source_transaction_hash) AS n_eth_interactions
  FROM
    external.layerzero.fact_transactions_snapshot
  WHERE
    source_chain = 'Ethereum'
  GROUP BY
    1
),
layerzero_data AS (
  SELECT
    a.addr,
    a.n_eth_interactions,
    COUNT(DISTINCT b.source_chain) AS n_source_chains_w_interactions,
    COUNT(DISTINCT b.destination_chain) AS n_dest_chains_w_interactions,
    COUNT(DISTINCT b.destination_chain) / COUNT(DISTINCT b.source_chain) AS n_avg_dest_chain_per_source_chain,
    COUNT(DISTINCT b.project) / COUNT(DISTINCT b.source_chain) AS n_avg_project_per_source_chain,
    MAX(COALESCE(b.stargate_swap_usd, 0)) AS max_stargate_swap,
    AVG(COALESCE(b.stargate_swap_usd, 0)) AS avg_stargate_swap,
    COUNT(DISTINCT b.source_transaction_hash) AS n_cross_chain_interactions
  FROM
    addresses a
    JOIN external.layerzero.fact_transactions_snapshot b ON a.addr = b.sender_wallet
  GROUP BY
    1,
    2
),
ethereum_tx_data AS (
  SELECT
    from_address AS addr,
    COUNT(*) AS num_transactions,
    MIN(
      DATEDIFF(
        SECOND,
        block_timestamp,
        TO_TIMESTAMP_NTZ('2024-05-01 00:00:00')
      )
    ) AS earliest_transaction_time_out,
    MAX(
      DATEDIFF(
        SECOND,
        block_timestamp,
        TO_TIMESTAMP_NTZ('2024-05-01 00:00:00')
      )
    ) AS latest_transaction_time_out,
    MAX(value) AS max_tx_value,
    MIN(value) AS min_tx_value,
    SUM(value) AS total_tx_value,
    AVG(value) AS avg_tx_value,
    MAX(tx_fee) AS max_tx_fee,
    MIN(tx_fee) AS min_tx_fee,
    COUNT(DISTINCT to_address) AS out_degree
  FROM
    ethereum.core.fact_transactions a
    JOIN addresses b on a.from_address = b.addr
  WHERE
    block_timestamp <= TO_TIMESTAMP_NTZ('2024-05-01 00:00:00')
  GROUP BY
    from_address
),
ethereum_in_tx_data AS (
  -- Inbound transaction metrics
  SELECT
    to_address AS addr,
    COUNT(*) AS num_transactions_in,
    MIN(
      DATEDIFF(
        SECOND,
        block_timestamp,
        TO_TIMESTAMP_NTZ('2024-05-01')
      )
    ) AS earliest_transaction_time_in,
    MAX(
      DATEDIFF(
        SECOND,
        block_timestamp,
        TO_TIMESTAMP_NTZ('2024-05-01')
      )
    ) AS latest_transaction_time_in,
    MAX(value) AS max_tx_value_in,
    MIN(value) AS min_tx_value_in,
    SUM(value) AS total_tx_value_in,
    AVG(value) AS avg_tx_value_in,
    COUNT(DISTINCT from_address) AS indegree
  FROM
    ethereum.core.fact_transactions a
    JOIN addresses b ON a.to_address = b.addr
  WHERE
    block_timestamp < TO_TIMESTAMP_NTZ('2024-05-01 00:00:00')
  GROUP BY
    to_address
)
SELECT
  l.addr,
  l.n_eth_interactions,
  l.n_source_chains_w_interactions,
  l.n_dest_chains_w_interactions,
  l.n_avg_dest_chain_per_source_chain,
  l.n_avg_project_per_source_chain,
  l.n_cross_chain_interactions,
  l.max_stargate_swap,
  l.avg_stargate_swap,
  e.num_transactions,
  e.earliest_transaction_time_out,
  e.latest_transaction_time_out,
  e.max_tx_value,
  e.min_tx_value,
  e.total_tx_value,
  e.avg_tx_value,
  e.max_tx_fee,
  e.min_tx_fee,
  e.out_degree,
  (
    COALESCE(e.latest_transaction_time_out - e.earliest_transaction_time_out,0)
  ) / 86400 AS time_span_day_out,
  CASE
    WHEN (
      COALESCE(e.latest_transaction_time_out - e.earliest_transaction_time_out,0)
    ) = 0 THEN 0
    ELSE e.out_degree / (
      (
        e.latest_transaction_time_out - e.earliest_transaction_time_out
      ) / 86400
    )
  END AS out_degree_per_day_out,
  CASE
    WHEN (
      COALESCE (e.latest_transaction_time_out - e.earliest_transaction_time_out, 0)
    ) = 0 THEN 0
    ELSE e.total_tx_value / (
      (
        e.latest_transaction_time_out - e.earliest_transaction_time_out
      ) / 86400
    )
  END AS tx_value_per_day_out,
  COALESCE(ei.num_transactions_in, 0),
  COALESCE(ei.earliest_transaction_time_in,0),
  COALESCE(ei.latest_transaction_time_in,0),
  COALESCE(ei.max_tx_value_in,0),
  COALESCE(ei.min_tx_value_in,0),
  COALESCE(ei.total_tx_value_in,0),
  COALESCE(ei.avg_tx_value_in,0),
  ei.indegree,
  (
    COALESCE(ei.latest_transaction_time_in - ei.earliest_transaction_time_in, 0)
  ) / 86400 AS time_span_day_in,
  CASE
    WHEN (
      COALESCE(ei.latest_transaction_time_in - ei.earliest_transaction_time_in, 0)
    ) = 0 THEN 0
    ELSE ei.indegree / (
      (
        COALESCE(ei.latest_transaction_time_in - ei.earliest_transaction_time_in, 0)
      ) / 86400
    )
  END AS indegree_per_day_in,
  CASE
    WHEN (
      COALESCE(ei.latest_transaction_time_in - ei.earliest_transaction_time_in, 0)
    ) = 0 THEN 0
    ELSE ei.total_tx_value_in / (
      (
        COALESCE(ei.latest_transaction_time_in - ei.earliest_transaction_time_in, 0)
      ) / 86400
    )
  END AS tx_value_per_day_in
FROM
  layerzero_data l
  LEFT JOIN ethereum_tx_data e ON l.addr = e.addr 
  LEFT JOIN ethereum_in_tx_data ei on l.addr = ei.addr
  -- CHUNK DATA FOR FLIPSIDE DOWNLOAD
  QUALIFY RANK() OVER (
    ORDER BY
      l.addr ASC NULLS LAST
  ) BETWEEN 400000
  AND 500000
ORDER BY
  l.addr;
