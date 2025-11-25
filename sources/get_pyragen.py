from datasets import load_dataset

dataset = load_dataset("bnadimi/PyraNet-Verilog")
dataset.save_to_disk("./pyranet_verilog_local")

print(dataset)

for sample in dataset['train']:
    print(sample)
    break
