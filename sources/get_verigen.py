from datasets import load_dataset

dataset = load_dataset("shailja/Verilog_GitHub")
dataset.save_to_disk("./verigen_local")

# Ver la estructura
print(dataset)

# Acceder a los datos
for sample in dataset['train']:
    print(sample)
    break
