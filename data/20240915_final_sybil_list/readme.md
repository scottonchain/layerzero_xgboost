# LayerZero Airdrop – Sybil Addresses Report

This file contains a list of Ethereum addresses identified as **Sybil participants** in the LayerZero airdrop. These addresses were flagged and excluded from rewards due to suspected manipulation or abuse of the airdrop eligibility criteria.

## About the LayerZero Airdrop

[LayerZero](https://layerzero.network/), a cross-chain messaging protocol, conducted a [large airdrop campaign](https://dune.com/oladee/layerzero-airdrop-claim-stats) in **June 2024**

## Sybil Detection Process

To preserve the integrity of the airdrop and prevent Sybil attackers from unfairly claiming rewards, the **LayerZero Foundation** employed a **multi-layered, rigorous detection framework**, which included:

### Technical and Analytical Methods

- **Louvain Community Detection Algorithm**: Used to identify clusters of addresses likely controlled by the same actor
- **Outsourced Expert Analysis**: Detection was supported by [Nansen](https://www.nansen.ai/) and [Chaos Labs](https://www.chaoslabs.xyz/)
- **Self-Reporting Bounty**: Users could voluntarily disclose Sybil activity for a reduced penalty
- **Crowd-Sourced Challenge**: The public was incentivized to identify Sybil clusters, enhancing coverage


## File Description

The csv file fcfs_file.csv contains details of Sybil addresses that were excluded from the airdrop. Each entry includes metadata related to the detection and community discussions.

### Columns:

| Column              | Description |
|---------------------|-------------|
| **Address**         | Detected Sybil address |
| **Reward Address1** | Primary address originally assigned to receive the reward |
| **Reward Address2** | (Optional) Secondary linked reward address |
| **Commonwealth Report Link** | Link to public discussion on the LayerZero forum |
| **GitHub Issue**    | Related GitHub issue, if applicable |
| **Timestamp**       | When the address was flagged |
| **Allocation**      | Amount of ZRO originally allocated to the address |


