# LayerZero's initial Sybil list

`initial_list.csv`: the initial list of Sybil addresses LayerZero published on 18 May 2024, before the
bounty-hunting phase (sources: LayerZero, Nansen and Chaos Labs, including self-reported addresses). By the
bounty rules, the bounty-accepted list the pipeline uses as labels (`data/20240915_final_sybil_list/fcfs_list.csv`)
excludes these addresses. Used only by `16_label_robustness_initial_list`; the pipeline and its labels do not read it.
It is stored here, not under `data/`, because `data/` is hash-checked by `sp.provenance`.

| | |
|---|---|
| Source | LayerZero's `LayerZero-Labs/sybil-report` repository (deleted), raw files archived by the Wayback Machine on 24 May 2024: <https://web.archive.org/web/*/https://github.com/LayerZero-Labs/sybil-report/raw/main/*> |
| File | `initial_list.csv` (Git LFS), one column `ADDRESS` |
| Size | 34,560,871 bytes |
| SHA-256 | `1a643f70056fe0c0b38d4a13468e968a2de0bd18de6bc25ed7139d9fa691a7cc` |
| Git blob | `0fc2e78854519c233e2e5f730305f83afb799ef8` |
| Rows | 803,093 addresses, no duplicates, all lowercase |
| Formats | 801,932 20-byte EVM addresses; 1,161 32-byte addresses (consistent with Aptos, a non-EVM chain LayerZero supports) |
| Overlap with `fcfs_list.csv` | 0 of 151,784 |
| Retrieved | 2026-10-06 |

## Provenance

**Cite as:** LayerZero Labs, initial Sybil list, `sybil-report` repository, 18 May 2024; archived copy, Internet
Archive Wayback Machine, captured 24 May 2024, <https://web.archive.org/web/*/https://github.com/LayerZero-Labs/sybil-report/raw/main/*>.


LayerZero published the list in `github.com/LayerZero-Labs/sybil-report`, which has since been deleted (the
GitHub API returns 404 for the repository and its forks).

**Archived original.** The Wayback Machine holds the repository's raw files as captured on 24 May 2024
(<https://web.archive.org/web/*/https://github.com/LayerZero-Labs/sybil-report/raw/main/*>). On 2026-10-06 the
authors retrieved the list from that capture and shared it (Google Drive, file id
`1GJ356Vu59_188nxBAygkky4Bv5XbKvX1`, served as `initialList.csv`). That file is byte-identical to the one committed
here (same SHA-256 and size, checked 2026-10-06). The archive rate-limited every request from the machine used for
the analysis, so the capture itself was not fetched from there.

Independent copies, found and compared on 2026-10-05 and 2026-10-06, agree:

| Copy | Name | First committed / dated | Agreement |
|---|---|---|---|
| `psrvere/layerzero-sybil-checker` (GitHub) | `initialList.csv` | 2024-05-18 13:44 UTC | identical bytes (same git blob) |
| `ibuyshite/LayerZero-Sybil` (GitHub) | `LO Report` | 2024-05-18 00:31 UTC | identical bytes |
| `tomkysar/sybil.rip` (GitHub) | `addys.csv` | 2024-05-18 02:10 UTC | identical bytes |
| `shuail0/layerzero-sybil-query` (GitHub) | `initialList.csv` | 2024-05-20 08:35 UTC | identical bytes; README: "the 800K Sybil data provided by LayerZero" |
| Google Drive zip (obtained by the authors) | `initialList.csv` inside `initialList.csv.zip` | file dated 2024-05-18 07:08 (zip time, no time zone) | identical bytes |
| Telegram copy (obtained by the authors) | `initialList.txt` | — | same 803,093 addresses in the same order; 297 written in scientific notation (`3.128…E+46`), each equal to the numeric value of the hex address, i.e. damaged by a spreadsheet |

The file committed here is byte-identical to the archived original and to the GitHub and Google Drive copies.
Details and the search log: `docs/REVISION_LEAKAGE.md`, findings 2026-10-06 (A16).

## Among the 434,786 interactors

34,545 interactors are on the list (all labeled non-Sybil; 8.29 % of the 416,575 non-Sybils): IxL 29,646, IxE 2,669,
IxI 2,206, No provider 24. None of the 18,211 positives is on it.
