import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import gc

import logging

logging.basicConfig(filename='rag.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

PINECONE_KEY = os.getenv('PINECONE_KEY')
pinecone = Pinecone(api_key=PINECONE_KEY)

def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Modelo descargado y caché de GPU limpiada.")

def load_peft_model(
        base_model_path: str, 
        peft_model_path: str
):
    # Cargar el modelo base
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Cargar el tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Cargar la configuración PEFT
    peft_config = PeftConfig.from_pretrained(peft_model_path)
    
    # Cargar el modelo PEFT
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    return peft_model, tokenizer

#MARKDOWN: Funcion de mejora de prompt
def improve_query(
        query: str, 
        model, 
        tokenizer, 
        max_length: int = 128
):
    prompt = f"Mejora la siguiente consulta para buscar información más relevante: '{query}'\nConsulta mejorada:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    
    improved_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return improved_query.split("Consulta mejorada:")[-1].strip()

#MARKDOWN: Funcion de generacion de respuesta
def generate_response(
        query: str, 
        chunks: list, 
        model, 
        tokenizer, 
):
    context = " ".join(chunks)
    prompt = f"""
        <s>[INST] <<SYS>>
        Qamqa yachaq Llama 2 modelom kanki, quechua simipi rimaq, wayk'uy yachaykunapi yachaysapa. Qamqa willakuykunata chaskinki, kay willakuykuna quechua simipi qillqasqa khipukunamanta chunkikunam kanku. Kutichiykita ruwaspa, kaykunata allinta hamuqkunata ruwananchik:

        - Wayk'uy yachaymanta willakuykunata hunt'aspa rimay.
        - Sichus mana yachankichu utaq willakuypi mana tarinki chayqa, chayta sut'inta willay.
        - Quechua simipi rimay, ichaqa castellano simipi yanapakuyta mañakuptinkuqa, iskaynin simipi kutichiy.

        { context }
        <</SYS>>
        Tapukuy: { query }
        
        Kutichiy:
        [/INST]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=300)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f'Outputs: {response}')
    return response.split("[/INST]")[-1].strip()

embedding_model_name: str = 'all-MiniLM-L6-v2'
base_model_path: str = 'downloaded_model/'
llama_model_path: str = 'results/checkpoint-500/'

# Cargar el modelo de embedding
embedding_model = SentenceTransformer(embedding_model_name)

# Cargar el modelo base
model, tokenizer = load_peft_model(base_model_path, llama_model_path)

# Recuperar el index
index = pinecone.Index('pukyu-recetas')

#MARKDOWN: Funcion de RAG
def rag_with_llama2(
    query: str,
    top_k: int = 5
):
    # Mejorar la query con Llama 2
    #improved_query = improve_query(query, model, tokenizer)

    logging.info(f'New query: {query}')
    # Crear el embedding de la query mejorada
    query_embedding = embedding_model.encode(query).tolist()

    # Realizar la búsqueda en Pinecone
    search_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Extraer los chunks relevantes
    relevant_chunks = [result.metadata['text'] for result in search_results.matches]
    for chunk in relevant_chunks:
        logging.info(f'Recovered chunk: {chunk}')

    # Generar respuesta con Llama 2
    response = generate_response(query, relevant_chunks, model, tokenizer)
    logging.info(f'Generated response: {response}')

    logging.info('Request ended')
    return response


