import pandas as pd
import numpy as np
import os
import csv

label_dir = r"data\20241011_l0_address_labels"

# Initialize a set to hold addresses to exclude
labeled_addresses = set()

# Load each CSV file in the specified directory
for filename in os.listdir(label_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(label_dir, filename)
        # Load the CSV file
        df_labels = pd.read_csv(file_path)
        labeled_addresses.update(df_labels['ADDR'].unique())

label_dir = r"C:\Users\scott\Documents\20240903 Octan\20240903 Jupyter\inputs\20241013_hildobby_cex_evms"

# Load each CSV file in the specified directory
for filename in os.listdir(label_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(label_dir, filename)
        # Load the CSV file
        df_labels = pd.read_csv(file_path)
        labeled_addresses.update(df_labels['address'].unique())  
        
        
# ARE WE GETTING CONTRACTS AS GAS PROVIDERS?  WHY?

label_dir = r"C:\Users\scott\Documents\20240903 Octan\20240903 Jupyter\inputs\20241114_bigquery_contract_list"

# Load each CSV file in the specified directory
for filename in os.listdir(label_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(label_dir, filename)
        # Load the CSV file
        df_labels = pd.read_csv(file_path)
        labeled_addresses.update(df_labels['address'].unique())    
        
        
label_dir = r"C:\Users\scott\Documents\20240903 Octan\20240903 Jupyter\inputs\20241112_dune_contract_list"

# Load each CSV file in the specified directory
for filename in os.listdir(label_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(label_dir, filename)
        # Load the CSV file
        df_labels = pd.read_csv(file_path)
        labeled_addresses.update(df_labels['address'].unique()) 
        
label_dir = r"C:\Users\scott\Documents\20240903 Octan\20240903 Jupyter\inputs\20241112_dawsbot_labels"


for filename in os.listdir(label_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(label_dir, filename)
        # Load the CSV file
        df_labels = pd.read_csv(file_path)
        labeled_addresses.update(df_labels['address'].unique())     

#C:\Users\scott\Documents\20240903 Octan\20240903 Jupyter\inputs\20241112_brianleect_labels
label_dir = r"C:\Users\scott\Documents\20240903 Octan\20240903 Jupyter\inputs\20241112_brianleect_labels"

# Load each CSV file in the specified directory
for filename in os.listdir(label_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(label_dir, filename)
        # Load the CSV file
        df_labels = pd.read_csv(file_path)
        labeled_addresses.update(df_labels['Address'].unique())     




labeled_addresses.add('0x00000000000007736e2f9aa5630b8c812e1f3fc9')
labeled_addresses.add('0xe93685f3bba03016f02bd1828badd6195988d950')
labeled_addresses.add('0xd64791e747188b0e5061fc65b56bf20fee2e3321')
labeled_addresses.add('0x000f422887ea7d370ff31173fd3b46c8f66a5b1c')
labeled_addresses.add('0x912fd21d7a69678227fe6d08c64222db41477ba0')
labeled_addresses.add('0x230a1ac45690b9ae1176389434610b9526d2f21b')
labeled_addresses.add('0x151b381058f91cf871e7ea1ee83c45326f61e96d')
labeled_addresses.add('0x3b794929566e3ba0f25e4263e1987828b5c87161')
labeled_addresses.add('0x912fd21d7a69678227fe6d08c64222db41477ba0')
labeled_addresses.add('0x91962711a4d2e4a830b366ce7276d99001e8564b')
labeled_addresses.add('0x0385b3f162a0e001b60ecb84d3cb06199d78f666')
labeled_addresses.add('0x4fb5df81b644e3bd5ad0ba07dce2b67559c764e0')
labeled_addresses.add('0x2a038e100f8b85df21e4d44121bdbfe0c288a869')
labeled_addresses.add('0xcc9557f04633d82fb6a1741dcec96986cd8689ae')
labeled_addresses.add('0x963737c550e70ffe4d59464542a28604edb2ef9a')
labeled_addresses.add('0x2db1d8cdf1abe8c70b531a790cdf2ff38aecf652')
labeled_addresses.add('0x5e809a85aa182a9921edd10a4163745bb3e36284')
labeled_addresses.add('0xa88902d6e93922893ee77234ed1c3ba4bec90224')
labeled_addresses.add('0x1439eda7f9a911b9120e9a0dafb60eae317f7685')
labeled_addresses.add('0x1eaca1277bcdfa83e60658d8938b3d63cd3e63c1')
labeled_addresses.add('0x158353a7601c7ba9bfdea6c77e571557267cd03b')
labeled_addresses.add('0xffff3dcb664c3f69b049d121fba7b7d7273961ef')
labeled_addresses.add('0x0000000000000068f116a894984e2db1123eb395')
labeled_addresses.add('0x469503159ddf6bfd0a9ec8eba8e97a84fd3eae5b')

labeled_addresses.add('0x000005a271a610964bb42658c7ff50fee2aa055a')

labeled_addresses.add('0x3b0BC51Ab9De1e5B7B6E34E5b960285805C41736'.lower())
labeled_addresses.add('0x6B8fAC654D072d8A799F03626db7E4f4679B0E1d'.lower()) #Kraken

#0x120A270bbC009644e35F0bB6ab13f95b8199c4ad

#Shapeshift
labeled_addresses.add('0x120A270bbC009644e35F0bB6ab13f95b8199c4ad'.lower())
labeled_addresses.add('0x9e6316f44BaEeeE5d41A1070516cc5fA47BAF227'.lower())
labeled_addresses.add('0x70faa28A6B8d6829a4b1E649d26eC9a2a39ba413'.lower())
labeled_addresses.add('0x563b377A956c80d77A7c613a9343699Ad6123911'.lower())
labeled_addresses.add('0xD3273EBa07248020bf98A8B560ec1576a612102F'.lower())
labeled_addresses.add('0x3b0BC51Ab9De1e5B7B6E34E5b960285805C41736'.lower())
labeled_addresses.add('0xeed16856D551569D134530ee3967Ec79995E2051'.lower())
labeled_addresses.add('0x70217E7De3A68187905269462506f81cF344bbad'.lower())
labeled_addresses.add('0x9BcB0733C56B1D8F0c7c4310949E00485cAe4E9d'.lower())

labeled_addresses.add('0xee77aa3Fd23BbeBaf94386dD44b548e9a785ea4b'.lower())
labeled_addresses.add('0x2f155ddeFC29c414C94b801B91F55B257231825E'.lower())
labeled_addresses.add('0x3AEf01dB231c3C9fF844f7E611c63b8c36bc6A02'.lower())
labeled_addresses.add('0xdf69de4a2a58866afeBb7713e3dd10C2153fF27C'.lower())


labeled_addresses.add('0xe65A88f487F5d26469Cfd37ce7Ef763D6d9BE454'.lower())
labeled_addresses.add('0xF08BDf21373A09aB7eDD7769A402D3a22826D317'.lower())
labeled_addresses.add('0xDa1E5D4Cc9873963f788562354b55A772253b92f'.lower())
labeled_addresses.add('0xdf69de4a2a58866afeBb7713e3dd10C2153fF27C'.lower())

labeled_addresses.add('0x8d12A197cB00D4747a1fe03395095ce2A5CC6819'.lower())

labeled_addresses.add('0x84A518A35C7c361C64301b8f209C2F0edB6F608d'.lower())

#0x84A518A35C7c361C64301b8f209C2F0edB6F608d
#0x8d12A197cB00D4747a1fe03395095ce2A5CC6819

# This is a contract 0x0000000000a8fb09af944ab3baf7a9b3e1ab29d8 listed as gas provider for 0xe206e3dca498258f1b7eec1c640b5aee7bb88fd0 which is in turn a miner.
# NEED TO UNDERSTAND THIS MINER and THE CONTRACT ACTIVATION TRANSACTION -does this happen more often?
labeled_addresses.add('0x0000000000a8fb09af944ab3baf7a9b3e1ab29d8')

# Display the number of excluded addresses
print(f"Number of labeled addresses: {len(labeled_addresses)}")

output_file = (r'data\20241213_consolidated_addresses\20241213_consolidated_addresses.csv')

ad = sorted(list(labeled_addresses))

with open(output_file, mode="w", newline="\n") as file:
    #writer = csv.writer(file)
    for address in ad:
        try:
            file.write(f"{address}\n")
        except Exception as e:
            print(address, e)

print(f"Labeled addresses written to {output_file}")