# LayerZero Airdrop – Sybil Addresses Report

This file contains a list of Ethereum addresses identified as **Sybil participants** in the LayerZero airdrop. These addresses were flagged and excluded from rewards due to suspected manipulation or abuse of the airdrop eligibility criteria.

---

## 📦 About the LayerZero Airdrop

[LayerZero](https://layerzero.network/), a cross-chain messaging protocol, conducted a [large airdrop campaign](https://dune.com/oladee/layerzero-airdrop-claim-stats) in **June 2024**, distributing **85 million ZRO tokens** — valued at approximately **$425 million USD** — to early users and contributors.

### Airdrop Highlights

- **Token Amount**: 85,000,000 ZRO
- **Market Value**: ~$425M USD
- **Goal**: Reward real users and supporters of the LayerZero ecosystem
- **Eligibility**: Based on multi-chain interactions and usage of LayerZero-enabled dApps (e.g., Stargate, Radiant, etc.)

---

## 🔍 Sybil Detection Process

To preserve the integrity of the airdrop and prevent Sybil attackers from unfairly claiming rewards, the **LayerZero Foundation** employed a **multi-layered, rigorous detection framework**, which included:

### 🧠 Technical and Analytical Methods

- **Louvain Community Detection Algorithm**: Used to identify clusters of addresses likely controlled by the same actor
- **Outsourced Expert Analysis**: Detection was supported by [Nansen](https://www.nansen.ai/) and [Chaos Labs](https://www.chaoslabs.xyz/)
- **Self-Reporting Bounty**: Users could voluntarily disclose Sybil activity for a reduced penalty
- **Crowd-Sourced Challenge**: The public was incentivized to identify Sybil clusters, enhancing coverage

### 📊 Results

- **Eligible Wallets**: ~1.28 million
- **Flagged Sybils**: ~480,000 addresses
- **Duration**: The process took **several months** and incurred significant cost
- **Reliability**: While a few **false positives** are acknowledged, this list is regarded as a **high-confidence truth set** for Sybil detection in the airdrop

---

## 📄 File Description

This CSV file contains details of Sybil addresses that were excluded from the airdrop. Each entry includes metadata related to the detection and community discussions.

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

---

## 📚 Purpose of This File

This file is published to support:

- **Transparency** in the airdrop and Sybil filtering process  
- **Auditability** for community and researchers  
- **Reference** for appeals and dispute resolution  

---

## 📌 Example Entry

```text
Address:             0xba166fa456d5ba51d5ce057981f54d52995285a3  
Reward Address1:     0x5788fac9d32c2c02394ae003eed5e2513a255b4d  
Reward Address2:     (empty)  
Commonwealth Link:   https://commonwealth.im/layerzero/discussion/18933  
GitHub Issue:        (none)  
Timestamp:           2024-05-28 05:14:14+00:00  
Allocation:          2.5000065492  

🔓 License

This dataset is released for public transparency and open analysis. Use is permitted under fair attribution to LayerZero and community contributors.
🔗 Useful Links

    LayerZero Official Website

    LayerZero Airdrop Claim Stats (Dune)

    Commonwealth Forum – LayerZero

    LayerZero GitHub