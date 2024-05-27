import os
import uvicorn 
from fastapi import FastAPI, Request, Query, Path, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import linear_kernel

app = FastAPI(title='Proyecto Integrador I Hecho por Michael Martinez')
# Cargar el dataset
df_recom = pd.read_parquet(r'https://github.com/bkmay1417/prueva/blob/296b29c99b3e9edfb7ad00c0728984e8f32ee37c/Dataset/recomendacion.parquet?raw=True')

@app.get("/Sistema_de_recomendacion")
async def recomendacion_juego(item_id: float = Query(default=10.0)):
    """
    10.0 = Counter-Strike
    """
    # Verificar que el item_id exista en el DataFrame
    if item_id not in df_recom['item_id'].values:
        return "El juego con el item_id proporcionado no existe."

    # Obtener el índice del juego dado su item_id
    idx = df_recom[df_recom['item_id'] == item_id].index[0]

    # Vectorizar los géneros
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_recom['genres_str'])

    # Obtenemos el vector tf-idf del item_id ingresado
    item_tfidf_vector = tfidf_matrix[idx]

    # Calcular la similitud del coseno
    cosine_sim = cosine_similarity(item_tfidf_vector, tfidf_matrix)

    # Obtener las puntuaciones de similitud del juego con todos los demás juegos
    sim_scores = list(enumerate(cosine_sim.flatten()))

    # Ordenar los juegos por puntuación de similitud (de mayor a menor)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los 5 juegos más similares (excluyendo el propio juego)
    sim_scores = sim_scores[1:6]

    # Obtener los item_id de los 5 juegos más similares
    game_indices = [i[0] for i in sim_scores]

    return df_recom['title'].iloc[game_indices].tolist()



