# Data

This file describes the input data in `data/`: what each file is, where it came from, when it was
retrieved, its size and columns, its Git LFS object, which notebooks read it, and its known issues.
Row counts exclude the header row. Facts were checked against the files, the readmes in each folder,
[`sybil_pipeline.py`](sybil_pipeline.py) and the notebooks at commit `3bbec2c` (notebooks `16`–`22` and `99` added afterwards). Where something is
not documented in the repository, this file says so.

The CSV files in `data/` are Git LFS objects (`data/.gitattributes`), except the two folders added in
2026, which are plain git files (each has its own `.gitattributes`). Run `git lfs pull` after cloning.
`data/` is part of the provenance check (`sp.provenance`): `10_tie_out` rejects results produced
before any change to it.

"Pipeline notebooks" below means the notebooks that build the feature table, either with
`sp.build_master_df` or step by step with `sp.data_paths` (`00`): `00` to `05`, `07` to `09`, and
`11` to `20` and `22`. `06`, `10` and `99` read no file in `data/`; they read saved predictions and results.

## Contents

| Folder | What | Rows | Used by |
|---|---|---|---|
| [`20240915_final_sybil_list/`](#20240915_final_sybil_list) | Sybil addresses accepted through LayerZero's community bounty (the labels) | 151,784 | Pipeline notebooks; also read directly by `14`, `15`, `16`, `22` |
| [`20241013_hildobby_cex_evms/`](#20241013_hildobby_cex_evms) | hildobby's list of known EVM CEX addresses | 2,431 | `07` (pre-snapshot label vintage) |
| [`20241104_layer0_sybil_features/`](#20241104_layer0_sybil_features) | LayerZero and Ethereum transaction features per interactor (5 files) | 434,788 (434,786 after cleaning) | Pipeline notebooks |
| [`20241114_gas_provision/`](#20241114_gas_provision) | Gas provision network: first ETH transfer into each address | 604,864 (archive: 758,633) | Pipeline notebooks; `07` directly. Archive: none |
| [`20241117_tree_features/`](#20241117_tree_features) | Original precomputed provider and tree features (regression check only) | 434,111 | `00` only |
| [`20241214_labeled_addresses/`](#20241214_labeled_addresses) | Known entity addresses compiled in 2024 | 9,054,104 | Pipeline notebooks (main file only) |
| [`20250208_cex_dex_indegree/`](#20250208_cex_dex_indegree) | Distinct CEX and DEX addresses that sent ETH to each interactor (5 files) | 434,788 | Pipeline notebooks |
| [`20260128_dune_spellbook_labels/`](#20260128_dune_spellbook_labels) | Dune Spellbook CEX, DEX and bridge lists at pinned commits | current: 4,957 / 73 / 136; presnapshot: 2,189 / 73 / 136 | Pipeline notebooks (current); `07` (presnapshot) |
| [`20260930_etherscan_service_labels/`](#20260930_etherscan_service_labels) | Etherscan name tags for funders of 50+ interactors (labeling rule, step 2) | 36 (lookups: 65) | Pipeline notebooks (`service_labels.csv`) |

---

<a id="20240915_final_sybil_list"></a>
## `20240915_final_sybil_list/`

**Contents.** `fcfs_list.csv`: one row per Sybil address, with the reward addresses, the
Commonwealth forum report and GitHub issue it was reported in, the time it was flagged and its ZRO
allocation. `readme.md` describes LayerZero's detection process and the columns.

**Source.** LayerZero's Sybil list for the June 2024 airdrop. The readme names the LayerZero
Foundation and links [layerzero.network](https://layerzero.network/) and a
[Dune dashboard of airdrop claims](https://dune.com/oladee/layerzero-airdrop-claim-stats); it does
not give the URL the file was downloaded from.

**Retrieval date.** 2024-09-15, from the folder prefix (the README calls it the "Sybil list
snapshot 2024-09-15"). The readme gives no date.

| File | Rows | Columns | Storage |
|---|---|---|---|
| `fcfs_list.csv` | 151,784 | `Address`, `Reward Address1`, `Reward Address2`, `Commonwealth Report link`, `Github issue`, `Timestamp`, `Allocation` | LFS `94347fb723f7a0acf437043f12a81b80129812469df05e7607416b1f82349b65`, 27,765,183 bytes |
| `readme.md` | – | – | plain git |

Every address is unique. `Reward Address2` is empty on 146,098 rows and `Github issue` on 112,340.
`Timestamp` is UTC and runs from 2024-05-18 02:00:24 to 2024-05-30 00:00:00. `Allocation` is the ZRO
amount originally allocated to the address (readme).

**Used by.** `sp.add_labels` sets `sybil = 1` for every interactor whose lower-cased address is in
`Address`, in every pipeline notebook. `07` uses the folder date as the list date
(`SYBIL_LIST_DATE = '2024-09-15'`). `14` and `15` read the file directly to group Sybils by bounty
report, GitHub issue and reporter; `16` reads it to check its overlap with LayerZero's initial list.

**Known issues.**
- This is LayerZero's list of Sybil addresses accepted through the community bounty (reports of
  18–31 May 2024), not LayerZero's full Sybil list. It does not contain LayerZero's initial list (803,093
  addresses, published 18 May 2024 in the deleted `LayerZero-Labs/sybil-report` repository; archived by the
  [Wayback Machine](https://web.archive.org/web/*/https://github.com/LayerZero-Labs/sybil-report/raw/main/*) on 24 May 2024), which is in [`review_support/initial_list/`](review_support/initial_list/README.md)
  outside `data/`: the two lists share no address. Consistent with this, every row has a Commonwealth report link and every `Timestamp` falls between 18 and
  30 May 2024. The README (input data table) still describes the file as "LayerZero Foundation's
  final Sybil list", and the folder's `readme.md` describes LayerZero's whole detection process
  (Louvain clustering, Nansen and Chaos Labs, self-reporting, the bounty) rather than the scope of
  this file. Both should be updated. The bounty window (18–31 May) is as stated by the authors; it
  is not documented elsewhere in the repository.
- The readme calls the file `fcfs_file.csv`; the file is `fcfs_list.csv`. Its column names differ
  slightly in case from the readme (`Commonwealth Report link`, `Github issue`).
- The meaning of "fcfs" is not documented.

**Terms of use.** Terms not documented in this repository — to be confirmed by the authors.

---

<a id="20241013_hildobby_cex_evms"></a>
## `20241013_hildobby_cex_evms/`

**Contents.** `All_Known_EVM_CEX_Addresses.2024-10-13.csv`: centralized-exchange addresses on EVM
chains, with exchange name, a distinct wallet name, who added the entry and when. `readme.txt`
holds only the source URL.

**Source.** Dune query [3237025](https://dune.com/queries/3237025) (`readme.txt`). According to
[`20260128_dune_spellbook_labels/readme.md`](data/20260128_dune_spellbook_labels/readme.md), the
file equals the Spellbook CEX list at commit `01ade3df4` (2024-09-04), address for address.

**Retrieval date.** 2024-10-13, from the folder prefix and the file name. The file was committed to
this repository on 2026-06-26 (`ff793e2`).

| File | Rows | Columns | Storage |
|---|---|---|---|
| `All_Known_EVM_CEX_Addresses.2024-10-13.csv` | 2,431 | `address`, `cex_name`, `distinct_name`, `added_by`, `added_date` | LFS `7f046a83c5f0a2930508e4e3804b9f3484a02f8ef5f815b6712ffd4d161719b1`, 206,801 bytes |
| `readme.txt` | – | – | plain git |

`added_date` runs from 2022-08-28 to 2024-08-13. The file has no newline after the last row.

**Used by.** Only the `presnapshot` label vintage (`sp.stream_labeled_anchors(..., vintage='presnapshot')`,
called by `07`): addresses in this list with `added_date` after 2024-05-01 are removed from the 2024
labeled-address file. The list is also one of the inputs of the 2024 labeled-address file (its build
script in `20241214_labeled_addresses/readme.txt`); all 2,431 addresses are in that file. The
`legacy/` notebook reads it.

**Terms of use.** Terms not documented in this repository — to be confirmed by the authors.

---

<a id="20241104_layer0_sybil_features"></a>
## `20241104_layer0_sybil_features/`

**Contents.** Five chunks of one query result, one row per address that sent a LayerZero transaction
from Ethereum: LayerZero activity (chains, projects, contracts, Stargate swaps, native drops, timing),
the same restricted to transactions into Ethereum (`*_TO_ETH_*`), and outbound and inbound Ethereum
transaction statistics. `readme.txt` holds the query.

**Source.** Flipside, query in [`readme.txt`](data/20241104_layer0_sybil_features/readme.txt). It reads
`external.layerzero.fact_transactions_snapshot` (LayerZero's snapshot table) and
`ethereum.core.fact_transactions`, with Ethereum transactions cut at 2024-05-01 00:00:00 (outbound
`<=`, inbound `<`). The query was run in chunks of 100,000 addresses by rank (the `QUALIFY RANK()`
clause; the readme shows the last chunk).

**Retrieval date.** 2024-11-04, from the folder prefix. The readme gives no date.

| File | Rows | Storage |
|---|---|---|
| `l0_features_0_100000.csv` | 100,000 | LFS `53b6baa79f7382232bce4964e38d69dbbcaad837b5a4b275f722fd397d9c9431`, 44,412,467 bytes |
| `l0_features_100000_200000.csv` | 100,000 | LFS `774dcdfc7421197971328871338ef4517b608013b81d7a86da0071eed2a12450`, 44,414,738 bytes |
| `l0_features_200000_300000.csv` | 100,000 | LFS `7d96d5462d5044716f81ff626b992686731789800cef2b7615b9996911e7114c`, 44,412,830 bytes |
| `l0_features_300000_400000.csv` | 100,000 | LFS `38a5b1e4c899459616a8aabfff22dba8d035f4257a60874af4c7aa95bff67eb9`, 44,417,071 bytes |
| `l0_features_400000_500000.csv` | 34,788 | LFS `b64b0aaaa0ee193b676609e4b5607800282478b9a11acbd156d2db6b685ec729`, 15,468,898 bytes |
| `readme.txt` | – | plain git |

Total 434,788 rows, 434,787 distinct addresses. `sp.load_l0` removes the burn address, the empty
rows and the duplicate, leaving 434,786 interactors.

Columns (57, the same in every file): `ADDR`, `N_ETH_INTERACTIONS`, `N_L0_SOURCE_CHAINS`,
`N_L0_DEST_CHAINS`, `N_L0_DEST_CHAIN_PER_SOURCE_CHAIN`, `N_L0_PROJECTS`,
`N_L0_PROJECT_PER_SOURCE_CHAIN`, `N_L0_SOURCE_CONTRACTS`, `N_L0_DEST_CONTRACTS`,
`L0_MAX_STARGATE_SWAP`, `L0_AVG_STARGATE_SWAP`, `L0_MIN_STARGATE_SWAP`, `L0_MAX_NATIVE_DROP_USD`,
`L0_AVG_NATIVE_DROP_USD`, `N_L0_TXS`, `EARLIEST_L0_TX_TIME`, `LATEST_L0_TX_TIME`, `L0_TX_TIME_SPAN`,
`N_L0_TO_ETH_SOURCE_CHAINS`, `N_L0_TO_ETH_PROJECTS`, `N_L0_TO_ETH_PROJECT_PER_SOURCE_CHAIN`,
`N_L0_TO_ETH_SOURCE_CONTRACTS`, `N_L0_TO_ETH_DEST_CONTRACTS`, `L0_TO_ETH_MAX_STARGATE_SWAP`,
`L0_TO_ETH_AVG_STARGATE_SWAP`, `L0_TO_ETH_MIN_STARGATE_SWAP`, `L0_TO_ETH_MAX_NATIVE_DROP_USD`,
`L0_TO_ETH_AVG_NATIVE_DROP_USD`, `N_L0_TO_ETH_TXS`, `EARLIEST_L0_TO_ETH_TX_TIME`,
`LATEST_L0_TO_ETH_TX_TIME`, `L0_TO_ETH_TX_TIME_SPAN`, `NUM_TRANSACTIONS`, `EARLIEST_TX_BLOCK_OUT`,
`LATEST_TX_BLOCK_OUT`, `TX_BLOCK_SPAN`, `MAX_TX_VALUE_OUT`, `MIN_TX_VALUE_OUT`, `TOTAL_TX_VALUE_OUT`,
`AVG_TX_VALUE_OUT`, `MAX_TX_FEE_OUT`, `MIN_TX_FEE_OUT`, `OUT_DEGREE`, `TIME_SPAN_OUT`,
`OUT_DEGREE_PER_BLOCK_OUT`, `TX_VALUE_PER_BLOCK_OUT`, `NUM_TRANSACTIONS_IN`, `EARLIEST_TX_BLOCK_IN`,
`LATEST_TX_BLOCK_IN`, `MAX_TX_VALUE_IN`, `MIN_TX_VALUE_IN`, `TOTAL_TX_VALUE_IN`, `AVG_TX_VALUE_IN`,
`IN_DEGREE`, `TIME_SPAN_IN`, `INDEGREE_PER_BLOCK_IN`, `TX_VALUE_PER_BLOCK_IN`. The pipeline
lower-cases the names and drops `in_degree` and `out_degree`.

**Used by.** `sp.load_l0`, in every pipeline notebook. The Etherscan build script
(`20260930_etherscan_service_labels/build_etherscan_lookups.py`) also reads it.

**Known issues.**
- Units of the timing features. `EARLIEST_L0_TX_TIME`, `LATEST_L0_TX_TIME` and their `*_TO_ETH_*`
  versions are Unix epoch seconds (UTC): the query computes them with `date_part(epoch_second, ...)`,
  and the values run from 1570912430 (2019-10-12 20:33:50 UTC) to 1714607998 (2024-05-01 23:59:58 UTC).
  `L0_TX_TIME_SPAN` and `L0_TO_ETH_TX_TIME_SPAN` are the difference of the two, in seconds, not days
  (maximum 143,185,447 s, about 1,657 days). The `*_BLOCK_*` and `TIME_SPAN_IN` / `TIME_SPAN_OUT`
  columns are in blocks.
- Each file ends with a stray UTF-8 byte-order mark after the last line. pandas reads it as one extra
  row with an empty-looking address and no values (so `len(pd.read_csv(...))` is 100,001, not
  100,000); `sp.load_l0` drops it as an all-empty row.
- `0x3aecba06e531a982cfdba16f0589ece5dc200fa9` is the last row of `l0_features_0_100000.csv` and the
  first row of `l0_features_100000_200000.csv` (chunk boundary). `sp.load_l0` keeps the first.
- The burn address `0x000…000` is the first row of the first file; `sp.load_l0` removes it.
- The README input table gives "434,793 raw": that count includes the five byte-order-mark rows.

**Terms of use.** Terms not documented in this repository — to be confirmed by the authors (Flipside).

---

<a id="20241114_gas_provision"></a>
## `20241114_gas_provision/`

**Contents.** `20241114_1633_layer0_provision_network_000000000000.csv`: the gas provision network,
one row per activated address, with the address that sent it its first ETH (the gas provider), the
amount, time, block and transaction. It covers the LayerZero interactors and, recursively, their
providers. `archive/20241114_layerzero_provisions_000000000000.csv` has the same columns and more rows.
`readme.txt` holds the queries.

**Source.** Google BigQuery, two queries in [`readme.txt`](data/20241114_gas_provision/readme.txt):
the first takes, for every address, the first transfer with value > 0 from the public dataset
`bigquery-public-data.crypto_ethereum.traces` (keeping only addresses whose first block has a single
such transfer); the second walks recursively from the LayerZero interactors to their providers, up
to 500 levels. Both read tables in the authors' project `octan-infra.layerzero_interactors`
(`layerzero_interactors`, `cex_dex_labels`) that are not in this repository; the tables were created
with a 14-day expiry. The query has no date filter. The query that produced the archive file is not
documented.

**Retrieval date.** 2024-11-14, from the folder prefix (the main file name also has `1633`, presumably
16:33 on that day; not documented).

| File | Rows | Columns | Storage |
|---|---|---|---|
| `20241114_1633_layer0_provision_network_000000000000.csv` | 604,864 | `activated_address`, `gas_provider`, `gas_provision_amount`, `first_gas_provision_time`, `block_number`, `tx_hash` | LFS `aa9aeed0995c8a5995b4ccdc8c1da87d5f4df4926e3dbd4109ee81108484ebeb`, 124,287,435 bytes |
| `archive/20241114_layerzero_provisions_000000000000.csv` | 758,633 | same | LFS `74da359185863b2228bd548779e3994d239ee2606ad9bdffdf8b1b6829bddc15`, 156,167,687 bytes |
| `readme.txt` | – | – | plain git |

`activated_address` is unique in both files. `gas_provision_amount` is in wei (the pipeline divides by
1e18). `first_gas_provision_time` is a UTC timestamp string; in the main file it runs to
2024-10-31 21:52:11 UTC. 203 rows have no `gas_provider`, of which 117 are block 0 rows dated
1970-01-01.

**Used by.** Main file: `sp.load_provision`, in every pipeline notebook; `07` also loads it directly;
the Etherscan build script reads it. Archive: not read by any notebook or by `sybil_pipeline.py`. It
is a superset of the main file: it contains every row of the main file unchanged (same provider and
transaction) plus 153,769 more activated addresses, and it covers 434,785 of the 434,786 interactors,
against 434,118 in the main file. Its purpose is not documented.

**Known issues.**
- Edges at or after `sybil_pipeline.SNAPSHOT_END` (2024-05-02 00:00 UTC) are dropped by
  `sp.load_provision` before any provider, chain or tree feature is computed: 296 rows of the main
  file. The latest edge kept is 2024-05-01 23:56:47 UTC. Rows with no provider and the burn address
  are dropped too.
- The source tables in `octan-infra` are not public and have expired, so the network cannot be
  rebuilt from the queries alone.

**Terms of use.** Terms not documented in this repository — to be confirmed by the authors (BigQuery
public dataset `crypto_ethereum`).

---

<a id="20241117_tree_features"></a>
## `20241117_tree_features/`

**Contents.** `20241117_graph_and_tree_features.csv`: the original precomputed provider and gas
provision tree features per interactor, as used in the 2025 paper. `20241117 Gas Provision
Featurization.ipynb`: the notebook that wrote it (no saved description cells).

**Source.** The featurization notebook in this folder. It reads the gas provision folder, plus label
lists and an interactor list from local Windows paths outside this repository
(`20241011_l0_address_labels`, `20241013_hildobby_cex_evms`, `20241114_bigquery_contract_list`,
`20241112_dune_contract_list`, `20241112_dawsbot_labels`, `20241112_brianleect_labels`,
`20241028_layerzero_interactors`).

**Retrieval date.** Built 2024-11-17 (folder prefix).

| File | Rows | Columns | Storage |
|---|---|---|---|
| `20241117_graph_and_tree_features.csv` | 434,111 | 33: `addr`, `provider_fan_out`, `provider_total_gas_provision_amount`, `provider_avg_gas_provision_amount`, `provider_max_gas_provision_amount`, `provider_min_gas_provision_amount`, `provider_is_star_like_attack`, `provider_is_labeled`, `tree_size`, `total_gas`, `max_depth`, `branching_factor`, `balance_factor`, `leaf_provision_proportion`, `avg_depth`, `depth_variance`, `leaf_to_internal_ratio`, `avg_leaf_gas`, `breadth_factor`, `breadth_to_depth_ratio`, `gini_coefficient`, `gas_distribution_entropy`, `gas_distribution_skewness`, `leaf_gas_distribution_entropy`, `leaf_gas_distribution_skewness`, `sparsity`, `depth`, `longest_chain_ratio`, `star_like_ratio`, `depth_weighted_avg_gas`, `chain_length`, `gas_provision_amount`, `gas_provision_block_number` | LFS `defb252b55d239297b94c4d9aa3b37db129a86e93d6e1266374d9ca52047478e`, 107,589,964 bytes |
| `20241117 Gas Provision Featurization.ipynb` | – | – | plain git |

**Used by.** `00` only, through `sp.compare_to_precomputed`. The pipeline does not use these values: it
recomputes the features from the filtered network (`sp.provision_features`). The featurization
notebook is not run by any notebook; it is kept as the source of the original file and of the port.

**Known issues.**
- Regression check only (README, "Why recompute the tree features?"; `CLAUDE.md`;
  [`docs/REVISION_LEAKAGE.md`](docs/REVISION_LEAKAGE.md), A7). The file was built from the
  unfiltered network, so it includes edges after the snapshot, and from inputs outside the repository.
- `0x3aecba06e531a982cfdba16f0589ece5dc200fa9` appears twice (434,110 distinct addresses);
  `sp.compare_to_precomputed` drops the duplicate. See A10 in `docs/REVISION_LEAKAGE.md`.
- `gini_coefficient` is identically 0 up to rounding (formula error, `docs/REVISION_LEAKAGE.md`); it is
  skipped in the comparison.

**Terms of use.** Derived by the authors from the gas provision network and label lists above. Terms
not documented in this repository — to be confirmed by the authors.

---

<a id="20241214_labeled_addresses"></a>
## `20241214_labeled_addresses/`

**Contents.** `20241214_labeled_addresses.csv`: 9,054,104 known entity addresses (exchanges,
contracts, named accounts), one lower-case address per line, sorted, no header, CRLF line endings.
`20241214_labeled_addresses_part_0.csv` to `_part_18.csv`: the same addresses split into 19 files of
500,000 lines (the last 54,104), in a different order. `additional_addresses.csv`: one address. `readme.txt`:
the build script.

**Source.** The script in [`readme.txt`](data/20241214_labeled_addresses/readme.txt) takes the union of
six lists, read from local paths outside this repository except the hildobby list:
`20241011_l0_address_labels` (Flipside LayerZero address labels, per `docs/REVISION_LEAKAGE.md`),
`20241013_hildobby_cex_evms`, `20241114_bigquery_contract_list`, `20241112_dune_contract_list`,
`20241112_dawsbot_labels` and `20241112_brianleect_labels`, then adds 44 addresses by hand
(`labeled_addresses.add(...)`). The script writes `20241213_consolidated_addresses.csv`; the file in
this folder has a different name, and how the part files and `additional_addresses.csv` were produced
is not documented.

**Retrieval date.** 2024-12-14, from the folder prefix (the script's output name says 2024-12-13). The
source lists are dated 2024-10-11 to 2024-11-14 in their folder names; the README says "Label lists
compiled Oct–Dec 2024".

| File | Rows | Columns | Storage |
|---|---|---|---|
| `20241214_labeled_addresses.csv` | 9,054,104 (`wc -l`) | none (one address per line) | LFS `77b8566d3d57b436e88121dcbb9282abff85ad9f0871fbbfda30e08761ddebe4`, 398,380,576 bytes |
| `20241214_labeled_addresses_part_0.csv` | 500,000 | none | LFS `513f07e8020f82674b898c0dfb33c55af55941b46728325deb7dacf25dd70d9a`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_1.csv` | 500,000 | none | LFS `6a3ab6f17edb6f8de7d55ecc3454c5834672f9a0e7191579c6f25ac415cb52e2`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_2.csv` | 500,000 | none | LFS `2ae0c09d6275d42aed700a5c19af2377110fffce6a9e22fd8967dae30db2f3b5`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_3.csv` | 500,000 | none | LFS `451d623933518020b02215daf03a676b0cb43747339d2663cb1d913f7c880134`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_4.csv` | 500,000 | none | LFS `48f1f99c9c70d3f5ceda43878edd9b5d52d4c2f108ebdae05d49c2a44f0b92c4`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_5.csv` | 500,000 | none | LFS `638e0df290269ff8d00ceb77a5fee4b3b1807ba2d75a6ab341ae76d287f538e1`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_6.csv` | 500,000 | none | LFS `174ff83b410f331b161a2e2f8fb27ec792506b7f943b86be7a7e98670c59134f`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_7.csv` | 500,000 | none | LFS `97f927f72708f593eb0efc682141b5744e3f805f11b5081fdbf5e75144abce9f`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_8.csv` | 500,000 | none | LFS `ead9be5fa99cd39bdf50519c23d4dfdb88ea81e6397dc862574f55dd41d86850`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_9.csv` | 500,000 | none | LFS `ba253b5cc702382003e3d612e9bc634fc53a10c2c6e8297f9d43f41e2c16e895`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_10.csv` | 500,000 | none | LFS `a43092373ffc3557cf66e7893bac4c27a4d6ed9ecea0801f0ab656ec53355fe2`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_11.csv` | 500,000 | none | LFS `554c33432ff012477fbe8f93cde0bde27f0f0c9985cf0474e53b1d61759c9b40`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_12.csv` | 500,000 | none | LFS `dd2ee00633a3fd672614db6ac8af0580072e92731f47680b50a76710da91b3b8`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_13.csv` | 500,000 | none | LFS `c0006c5abde88609c1342484199d98f2c25e1f56223c5d169ce69823a9bef6d4`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_14.csv` | 500,000 | none | LFS `b424bc2707c8e4717fdb4631ed86ac8d9d284f9a22dd55c7e7b6593e26617e85`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_15.csv` | 500,000 | none | LFS `c8030383ce259371b11d3a6bcff3af16b762f1ba8776921ac6d1c951ff903e55`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_16.csv` | 500,000 | none | LFS `e87613e642988c1c44ae9d50a4b843dc033a04e9f8d18665095706255c9707e7`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_17.csv` | 500,000 | none | LFS `e1a1e0a37eb5b693c0f7f74d8108fd9a1c69ceb57cfb2306a8989ea6331cfe3a`, 22,000,000 bytes |
| `20241214_labeled_addresses_part_18.csv` | 54,104 | none | LFS `18a277e7834ad43768f72ea92e39d9eaa22a592bde273547ef46580a076438b7`, 2,380,576 bytes |
| `additional_addresses.csv` | 1 (no header, no final newline) | none | LFS `0f81122461927345dde2211e9dbf4c2e24e9af5d5b758c4a0f4eab796b0f0cf3`, 42 bytes |
| `readme.txt` | – | – | plain git |

The 19 part files together hold exactly the lines of the main file (same SHA-256 after sorting).
All 44 hand-added addresses and all 2,431 hildobby addresses are in the main file.

**Used by.** Main file and `readme.txt`: `sp.stream_labeled_anchors`, in every pipeline notebook. The
main file is streamed line by line and only addresses in the provision network are kept (README, RAM
note); do not load it whole. The part files and `additional_addresses.csv` are not read by any
notebook or by `sybil_pipeline.py`. The `legacy/` notebook (2025 paper) loads every `.csv` in this
folder, so in the original analysis the labeled set also included the address in
`additional_addresses.csv` (`0x9241f27daffd0bb1df4f2a022584dd6c77843e64`, which is a LayerZero
interactor and the provider of 9 addresses in the network). No readme says why the part files or this
address were added.

**Known issues.**
- Hand-added addresses are excluded by the labeling rule. `sp.hand_added_addresses` parses the 44
  addresses added with `labeled_addresses.add('0x…')` in `readme.txt`, and `sp.stream_labeled_anchors`
  skips them when reading this file (in both label vintages). They are labeled only if a Spellbook list
  or the Etherscan step labels them independently; per `docs/REVISION_LEAKAGE.md` (A12), 5 of the 24
  that are not in the hildobby list remain labeled this way. The 20 that are also in the hildobby list
  are skipped here too; all 20 are in the current Spellbook CEX list (19 in the presnapshot list),
  so they stay labeled through it.
- Five of the six source lists are not in the repository, so the file cannot be rebuilt.
- The address in `additional_addresses.csv` is not labeled in the current pipeline but was in the 2025
  analysis (see above).

**Terms of use.** Terms not documented in this repository — to be confirmed by the authors (Flipside
labels, hildobby list, BigQuery and Dune contract lists, dawsbot and brianleect label sets).

---

<a id="20250208_cex_dex_indegree"></a>
## `20250208_cex_dex_indegree/`

**Contents.** Five chunks of one query result: for each LayerZero interactor, the number of distinct
addresses labeled `cex` and `dex` that sent it native ETH. `readme.txt` holds the query.

**Source.** Flipside, query in [`readme.txt`](data/20250208_cex_dex_indegree/readme.txt): labels from
`ethereum.core.dim_labels` (`label_type` `cex` or `dex`), transfers from
`ethereum.core.ez_native_transfers` with `block_timestamp <= '2024-05-01'`, interactors from
`external.layerzero.fact_transactions_snapshot` (`source_chain = 'Ethereum'`). Run with
`LIMIT 100000` and `OFFSET` 0 to 400000 (the readme shows the last).

**Retrieval date.** 2025-02-08, from the folder prefix. The readme gives no date.

| File | Rows | Columns | Storage |
|---|---|---|---|
| `cex_dex_features_in_0.csv` | 100,000 | `TO_ADDRESS`, `CEX_IN_COUNT`, `DEX_IN_COUNT` | LFS `5588932ed9f0ec33614c5f034d74f45415de8eb5e75b24b16a94c0e9afb93f58`, 4,807,430 bytes |
| `cex_dex_features_in_100000.csv` | 100,000 | same | LFS `450fc8a2fab757c4cedc52f1c6ec97504f778194c8d7bd8475d54814790b49a5`, 4,807,675 bytes |
| `cex_dex_features_in_200000.csv` | 100,000 | same | LFS `bcefb6a2ab01ee5ca4bf1ac32c81eef381c037f627ba63f481c8c0d00b950e7f`, 4,807,546 bytes |
| `cex_dex_features_in_300000.csv` | 100,000 | same | LFS `714de5e9d8a67c3e32db70158a2847f4caf5f01bb97693fbb2a038fe035e264b`, 4,807,502 bytes |
| `cex_dex_features_in_400000.csv` | 34,788 | same | LFS `ee4e35fce506cbf4f0630b968db7c4796809c1f232220832e418902008c131d2`, 1,672,429 bytes |
| `readme.txt` | – | – | plain git |

Total 434,788 rows: 434,787 distinct addresses, the same set as the L0 feature files, plus one row
whose address is the string `"null"` (last data row of the last file).

**Used by.** `sp.merge_cex_dex`, in every pipeline notebook (left join on address; missing counts are 0).

**Known issues.**
- As in the L0 feature files, each file ends with a stray UTF-8 byte-order mark that pandas reads as
  one extra row (100,001 in `len(pd.read_csv(...))`). After the merge it matches no interactor.
- The label source (`dim_labels`) is Flipside's current table at query time (2025-02-08), not a
  pre-snapshot version; only the transfers are cut at 2024-05-01.

**Terms of use.** Terms not documented in this repository — to be confirmed by the authors (Flipside).

---

<a id="20260128_dune_spellbook_labels"></a>
## `20260128_dune_spellbook_labels/`

**Contents.** CEX, DEX and bridge address lists for Ethereum, each at two pinned commits of Dune's
Spellbook: `current` (the pipeline's primary label set) and `presnapshot` (the last commit before the
LayerZero snapshot of 2024-05-01, used only in the label-vintage sensitivity check). `readme.md` gives
the commit and path of each file.

**Source.** [github.com/duneanalytics/spellbook](https://github.com/duneanalytics/spellbook), at the
commits and paths listed in [`readme.md`](data/20260128_dune_spellbook_labels/readme.md). These are the
sources of the Dune tables `addresses_ethereum.cex`, `addresses_ethereum.dex` and the Ethereum bridge
labels (readme).

**Retrieval date.** Extracted on 2026-10-01 (readme; committed the same day, `46c38c4`). The folder
prefix 2026-01-28 is the date of the newest pinned commit (`9f61b0dd2`), not the retrieval date.

| File | Rows | Commit (date) | Storage |
|---|---|---|---|
| `cex_evms_addresses_current_9f61b0dd2.csv` | 4,957 | `9f61b0dd2` (2026-01-28) | plain git (not LFS) |
| `cex_evms_addresses_presnapshot_839264841.csv` | 2,189 | `839264841` (2024-04-23) | plain git (not LFS) |
| `dex_ethereum_addresses_current_d2ae74352.csv` | 73 | `d2ae74352` (2024-07-26) | plain git (not LFS) |
| `dex_ethereum_addresses_presnapshot_4eef8dc7e.csv` | 73 | `4eef8dc7e` (2023-11-21) | plain git (not LFS) |
| `bridges_ethereum_current_9f61b0dd2.csv` | 136 | `9f61b0dd2` (2026-01-28) | plain git (not LFS) |
| `bridges_ethereum_presnapshot_ad08fa0cc.csv` | 136 | `ad08fa0cc` (2024-02-06) | plain git (not LFS) |
| `readme.md`, `.gitattributes` | – | – | plain git |

Columns (all files): `address`, `name`, `detail`, `added_date`. `added_date` is empty in both DEX files;
in the CEX files it runs from 2022-08-28 to 2026-01-02 (current) and to 2024-04-20 (presnapshot); in
the bridge files from 2022-09-22 to 2023-12-10. The current and presnapshot DEX files are
byte-identical, as are the two bridge files; only the CEX list differs between vintages.

**Used by.** `sp.SPELLBOOK['current']`: every pipeline notebook, through `sp.stream_labeled_anchors`.
`sp.SPELLBOOK['presnapshot']`: `07` only (`label_vintage='presnapshot'`), where entries with
`added_date` after 2024-05-01 are also dropped. `07` also reads the current CEX list directly for
`added_date`.

**Terms of use.** The readme describes these as "public address lists from Dune's open-source dbt
repository" but states no license or terms. Terms not documented in this repository — to be confirmed
by the authors.

---

<a id="20260930_etherscan_service_labels"></a>
## `20260930_etherscan_service_labels/`

**Contents.** Step 2 of the labeling rule. `etherscan_lookups.csv`: all 65 addresses looked up on
etherscan.io during the review, with the page URL, the date read, the name tag and badges, an entity
type, the number of interactors each funds and the rule's decision. `service_labels.csv`: the 36 rows
in the rule's scope (funders of at least `sp.ETHERSCAN_MIN_FANOUT` = 50 interactors not covered by the
public lists), 10 of them labeled. `build_etherscan_lookups.py` generates both files and asserts that
every address in scope was looked up. Column meanings are in
[`readme.md`](data/20260930_etherscan_service_labels/readme.md).

**Source.** Etherscan name tags read by the authors from each address's page (`etherscan_url`),
transcribed in `build_etherscan_lookups.py`; `interactors_funded`, `covered_by_public_list` and
`in_rule_scope` are computed by the script from the repository data. Run from the repo root:
`python data/20260930_etherscan_service_labels/build_etherscan_lookups.py`.

**Retrieval date.** Pages read on 2026-09-30 (54 rows) and 2026-10-01 (11 rows), per `checked_on`. The
folder prefix is the first date. Tags can change after `checked_on`.

| File | Rows | Columns | Storage |
|---|---|---|---|
| `service_labels.csv` | 36 (10 labeled) | `address`, `etherscan_url`, `checked_on`, `lookup_reason`, `address_kind`, `name_tag`, `tag_badges`, `entity_type`, `interactors_funded`, `covered_by_public_list`, `in_rule_scope`, `labeled`, `decision_reason` | plain git (not LFS) |
| `etherscan_lookups.csv` | 65 (14 with `labeled = True`, of which 4 are already covered by a public list) | same | plain git (not LFS) |
| `build_etherscan_lookups.py`, `readme.md`, `.gitattributes` | – | – | plain git |

**Used by.** `service_labels.csv`: `sp.stream_labeled_anchors` (current vintage only), in every
pipeline notebook. `etherscan_lookups.csv` is not read by any notebook; it is read by
`review_support/blockscout_crosscheck/blockscout_crosscheck.py`.

**Terms of use.** Derived from Etherscan pages. Terms not documented in this repository — to be
confirmed by the authors.

---

## Open questions for the authors

1. **Terms of use** for each third-party source, and whether redistributing the files in this
   repository is permitted:
   - LayerZero's Sybil list (`fcfs_list.csv`) and the source URL it was downloaded from;
   - hildobby's CEX list (Dune query 3237025) and the Dune Spellbook lists (no license file is quoted
     in `20260128_dune_spellbook_labels/readme.md`);
   - Flipside query results (`20241104_layer0_sybil_features`, `20250208_cex_dex_indegree`, and the
     Flipside LayerZero labels inside `20241214_labeled_addresses`);
   - Google BigQuery public dataset `crypto_ethereum` (`20241114_gas_provision`), and the BigQuery and
     Dune contract lists, dawsbot and brianleect label sets inside `20241214_labeled_addresses`;
   - Etherscan name tags (`20260930_etherscan_service_labels`).
2. **Sybil list scope.** Confirm the wording for `fcfs_list.csv` (bounty-accepted list, reports of
   18–31 May 2024; the file's timestamps end 2024-05-30 00:00:00 UTC; it shares no address with
   LayerZero's initial list in `review_support/initial_list/`), and whether the README's "final Sybil list" and the
   folder name should change. What does "fcfs" stand for?
3. **Archive network.** What query produced `20241114_gas_provision/archive/…`, and why is it kept?
4. **Part files and `additional_addresses.csv`.** How were they produced, and why? Should
   `0x9241f27daffd0bb1df4f2a022584dd6c77843e64`, labeled in the 2025 analysis through
   `additional_addresses.csv`, be mentioned in the paper's description of the change in labels?
5. **Labeled-address build.** The script writes `20241213_consolidated_addresses.csv`; confirm that
   `20241214_labeled_addresses.csv` is its output, and the retrieval dates of the six source lists.
6. **Retrieval times.** Confirm that the folder prefixes of the Flipside and BigQuery folders are the
   query dates, and what `1633` in the provision network file name means.
7. **Unpublished BigQuery tables.** `octan-infra.layerzero_interactors.layerzero_interactors` and
   `cex_dex_labels`, used by the provision queries, are not in the repository; should they be?
