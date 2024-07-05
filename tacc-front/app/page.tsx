// pages/index.js
'use client';
import { useEffect, useState } from 'react';
import { Input } from "@nextui-org/input";
import { Card, CardHeader, CardBody, CardFooter } from "@nextui-org/card";
import axios from 'axios';
import { toast } from 'react-toastify';
import { Spinner } from '@nextui-org/spinner';

const mockRecipe = {
  title: 'Ensalada de Frutas',
  ingredients: [
    '1 Manzana',
    '1 Plátano',
    '1 Naranja',
    '1/2 Taza de Uvas',
    '2 Cucharadas de Miel'
  ],
  instructions: 'Cortar todas las frutas en trozos pequeños, mezclar en un bol grande y añadir la miel. Mezclar bien y servir frío.'
};

export default function Home() {
  const [search, setSearch] = useState('');
  const [recipe, setRecipe] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!search) return;

    const fetchRecipe = async () => {
      setLoading(true);
      try {
        const response = await axios.post(`http://localhost:8000/?question=${search}`);
        setRecipe(response.data);
      } catch (error) {
        toast.error('Ha ocurrido un error al obtener la receta');
        //@ts-ignore
        setRecipe(mockRecipe);
        setRecipe(null);
      } finally {
        setLoading(false);
      }
    };

    fetchRecipe();
  }, [search]);


  return (
    <div className="bg-gray-900 text-gray-200 py-10 rounded-lg">
      <h1 className='text-4xl font-bold text-center mb-8 text-indigo-400'>Recomendador de Recetas</h1>
      <div className="flex justify-center mb-8">
        <Input
          isClearable
          placeholder="Buscar ingredientes receta..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full max-w-md placeholder-gray-400"
        />
      </div>
      <div className='flex justify-center'>
        {
          loading ?
          <Spinner /> 
          :
          !recipe ?
          <div className="text-gray-300">No se ha encontrado una respuesta.</div>
          :
        <Card className="w-full max-w-md bg-gray-800 shadow-lg rounded-lg p-6">
          <CardHeader className="mb-4 border-b border-gray-700">
            {
              //@ts-ignore
              <h3 className='text-xl font-bold text-indigo-400'>{recipe.title || mockRecipe.title}</h3>
              }
          </CardHeader>
          <CardBody>
            <h4 className='text-md font-semibold mb-2 text-indigo-300'>Ingredientes</h4>
            <ul className="list-disc list-inside mb-4">
              {
                //@ts-ignore
              recipe.ingredients ?
                //@ts-ignore
              recipe.ingredients?.map((ingredient, index) => (
                <li key={index} className="text-gray-300">{ingredient}</li>
              ))
            :
            mockRecipe.ingredients?.map((ingredient, index) => (
              <li key={index} className="text-gray-300">{ingredient}</li>
            ))
            }
            </ul>
            <h4 className='text-md font-semibold mb-2 text-indigo-300'>Instrucciones</h4>
            {//@ts-ignore
            <p className="text-gray-300">{recipe.instructions || mockRecipe.instructions}</p>
            }
          </CardBody>
        </Card>
        }
      </div>
    </div>
  );
}
