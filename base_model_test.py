from huggingface_hub import snapshot_download
from mlx_lm import load, generate
import os

# Configuración
model_id = "lmstudio-community/Qwen3-4B-Thinking-2507-MLX-4bit"
local_dir = "./Qwen3-4B-Thinking-2507-MLX-4bit"

# Descargar el modelo si no existe
if not os.path.exists(local_dir):
    print(f"Descargando modelo a {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("Descarga completada!")
else:
    print(f"Modelo ya existe en {local_dir}")

# Cargar y usar el modelo
print("Cargando modelo...")
model, tokenizer = load(local_dir)

messages = [{"role": "user", "content": "Say hello and then stop."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
out = generate(model, tokenizer, prompt=prompt, max_tokens=1000, verbose=True)
print(out)
