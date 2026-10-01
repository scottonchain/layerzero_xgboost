# Dune Spellbook entity labels

Public address lists from Dune's open-source dbt repository,
https://github.com/duneanalytics/spellbook, extracted at pinned commits on 2026-10-01.
These are the sources of the Dune tables `addresses_ethereum.cex` (hildobby's CEX list),
`addresses_ethereum.dex`, and the Ethereum bridge labels.

- **current**: the pipeline's primary label set.
- **presnapshot**: the last commit before the LayerZero snapshot (2024-05-01), used only in the
  label-vintage sensitivity check. Rows in the CEX list also carry `added_date`.

The earlier file `data/20241013_hildobby_cex_evms/` equals the CEX list at commit `01ade3df4`
(2024-09-04), address for address.

| File | Rows | Commit | Path in spellbook |
|---|---|---|---|
| `cex_evms_addresses_current_9f61b0dd2.csv` | 4,957 | `9f61b0dd2` (2026-01-28) | `dbt_subprojects/hourly_spellbook/models/_sector/cex/addresses/chains/cex_evms_addresses.sql` |
| `cex_evms_addresses_presnapshot_839264841.csv` | 2,189 | `839264841` (2024-04-23) | `models/cex/cex_evms_addresses.sql` |
| `dex_ethereum_addresses_current_d2ae74352.csv` | 73 | `d2ae74352` (2024-07-26) | `dbt_subprojects/dex/models/addresses/ethereum/dex_ethereum_addresses.sql` |
| `dex_ethereum_addresses_presnapshot_4eef8dc7e.csv` | 73 | `4eef8dc7e` (2023-11-21) | `models/dex/ethereum/dex_ethereum_addresses.sql` |
| `bridges_ethereum_current_9f61b0dd2.csv` | 136 | `9f61b0dd2` (2026-01-28) | `dbt_subprojects/daily_spellbook/models/_sector/labels/addresses/bridge/identifiers/labels_bridges_ethereum.sql` |
| `bridges_ethereum_presnapshot_ad08fa0cc.csv` | 136 | `ad08fa0cc` (2024-02-06) | `models/labels/addresses/bridge/identifiers/labels_bridges_ethereum.sql` |
