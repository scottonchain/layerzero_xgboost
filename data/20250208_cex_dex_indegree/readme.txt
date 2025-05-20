WITH UniqueLabels AS (
    SELECT DISTINCT address, label_type
    FROM ethereum.core.dim_labels
    WHERE label_type = 'cex' OR label_type='dex'
),
FilteredTransfers AS (
    SELECT ent.from_address, ent.to_address
    FROM ethereum.core.ez_native_transfers ent
    WHERE ent.block_timestamp <= '2024-05-01'
),
AggregatedData AS (
    SELECT
        ent.to_address,
        ul.label_type,
        COUNT(DISTINCT ul.address) AS unique_addresses
    FROM FilteredTransfers ent
    JOIN ethereum.core.dim_labels ul ON ent.from_address = ul.address
    GROUP BY ent.to_address, ul.label_type
)


SELECT
    fts.sender_wallet AS to_address,
    MAX(CASE WHEN ad.label_type = 'cex' THEN ad.unique_addresses ELSE 0 END) AS cex_in_count,
    MAX(CASE WHEN ad.label_type = 'dex' THEN ad.unique_addresses ELSE 0 END) AS dex_in_count
FROM
    external.layerzero.fact_transactions_snapshot fts
LEFT JOIN
    AggregatedData ad ON fts.sender_wallet = ad.to_address
WHERE source_chain = 'Ethereum'
GROUP BY
    fts.sender_wallet
ORDER BY
    fts.sender_wallet
LIMIT 100000
OFFSET 400000;  -- Change this value in increments of 10000 for each subsequent batch: 0, 10000, 20000, ...
