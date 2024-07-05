import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig
import torch

load_dotenv()

PINECONE_KEY = os.getenv('PINECONE_KEY')
pinecone = Pinecone(api_key=PINECONE_KEY)

#MARKDOWN: Funcion de mejora de prompt
def improve_query(query: str, model, tokenizer, max_length: int = 128):
    prompt = f"Mejora la siguiente consulta para buscar información más relevante: '{query}'\nConsulta mejorada:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    
    improved_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return improved_query.split("Consulta mejorada:")[-1].strip()

#MARKDOWN: Funcion de generacion de respuesta
def generate_response(query: str, chunks: list, model, tokenizer, max_length: int = 512):
    context = " ".join(chunks)
    prompt = f"Basado en la siguiente información:\n\n{context}\n\nResponde a la pregunta: {query}\n\nRespuesta:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Respuesta:")[-1].strip()

#MARKDOWN: Funcion de RAG
def rag_with_llama2(
    query: str,
    index_name: str = 'pukyu-recetas', 
    embedding_model_name: str = 'all-MiniLM-L6-v2',
    llama_model_path: str = 'pretrained_model/',
    top_k: int = 5
):
    # Inicializar Pinecone y el modelo de embedding
    index = pinecone.Index(index_name)
    embedding_model = SentenceTransformer(embedding_model_name)

    # Cargar el modelo base
    base_model = 'meta-llama/Llama-2-7b-hf'
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )  
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map={"": 0}, quantization_config=bnb_config
    )

    # Cargar el modelo Llama 2 y el tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    llama_model = PeftModel.from_pretrained(model, llama_model_path)

    # Mejorar la query con Llama 2
    improved_query = improve_query(query, llama_model, tokenizer)

    # Crear el embedding de la query mejorada
    query_embedding = embedding_model.encode(improved_query).to_list()

    # Realizar la búsqueda en Pinecone
    search_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Extraer los chunks relevantes
    relevant_chunks = [result.metadata['text'] for result in search_results.matches]

    # Generar respuesta con Llama 2
    response = generate_response(improved_query, relevant_chunks, llama_model, tokenizer)

    return response
