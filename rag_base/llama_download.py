from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import os
import shutil

def download_model_from_huggingface(model_name: str, output_dir: str):
    print(f"Iniciando la descarga del modelo {model_name}")
    
    try:
        # Configurar el directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Descargar el modelo
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True  # Permite reanudar descargas interrumpidas
        )
        
        # Verificar la descarga intentando cargar el modelo y el tokenizer
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        print(f"Modelo {model_name} descargado exitosamente en {output_dir}")
        return True
    except Exception as e:
        print(f"Error al descargar o cargar el modelo: {str(e)}")
        # Limpiar el directorio de descarga en caso de error
        shutil.rmtree(output_dir, ignore_errors=True)
        return False

# Ejemplo de uso
model_name = "meta-llama/Llama-2-7b-hf"  # Reemplaza con el nombre del modelo que deseas descargar
output_directory = "./downloaded_model"  # Directorio donde se guardará el modelo

success = download_model_from_huggingface(model_name, output_directory)

if success:
    print("El modelo se ha descargado correctamente y está listo para ser utilizado.")
else:
    print("Hubo un problema al descargar el modelo. Por favor, verifica los errores e intenta nuevamente.")