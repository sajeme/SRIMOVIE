from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TMDB_API_KEY = '9d1bbbbde65732d68b3d9fafccd663c3'

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)


def get_movie_info(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=es-ES'
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        genres = ' '.join([g['name'] for g in data.get('genres', [])])
        return {
            'id': movie_id,
            'title': data.get('title', ''),
            'genres': genres,
            'overview': data.get('overview', '')
        }
    else:
        print(f"Error al obtener información de la película {movie_id}: {res.status_code}")
    return None



@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/detalle.html')
def detalle():
    return send_from_directory('.', 'detalle.html')

VALORACIONES_JSON = 'valoraciones.json'

@app.route('/guardar_valoraciones', methods=['POST'])
def guardar_valoraciones():
    data = request.json
    usuario = data.get('usuario')
    valoraciones = data.get('valoraciones')  # dict: {movie_id: rating}

    if not usuario or not valoraciones:
        return jsonify({'error': 'Datos incompletos'}), 400

    # Cargar datos actuales o iniciar diccionario
    if os.path.exists(VALORACIONES_JSON):
        with open(VALORACIONES_JSON, 'r') as f:
            data_json = json.load(f)
    else:
        data_json = {}

    # Si el usuario ya existe, actualizar su info
    if usuario in data_json:
        data_json[usuario].update(valoraciones)
    else:
        data_json[usuario] = valoraciones

    # Guardar de nuevo
    with open(VALORACIONES_JSON, 'w') as f:
        json.dump(data_json, f, indent=2)

    return jsonify({'mensaje': 'Valoraciones guardadas correctamente'})

@app.route('/recomendar', methods=['GET'])
def recomendar():
    usuario = request.args.get('usuario')
    if not usuario:
        return jsonify({'error': 'Usuario no especificado'}), 400

    if not os.path.exists(VALORACIONES_JSON):
        return jsonify({'error': 'No hay valoraciones disponibles'}), 404

    with open(VALORACIONES_JSON, 'r') as f:
        data = json.load(f)

    if usuario not in data:
        return jsonify({'error': 'Usuario no encontrado'}), 404

    valoraciones = data[usuario]

    # Películas que el usuario calificó con 4 o más
    favoritas = [mid for mid, score in valoraciones.items() if score >= 4]
    if not favoritas:
        return jsonify({'error': 'No hay películas valoradas con 4 o más'}), 400

    print(f"Películas favoritas de {usuario}: {favoritas}")  # Agregado para debug

    # Obtener info de todas las películas que haya valorado el usuario
    peliculas_usuario = [get_movie_info(mid) for mid in favoritas]
    print(f"Películas del usuario: {peliculas_usuario}")  # Agregado para debug

    # También obtener info de otras películas populares de TMDB
    populares_url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=es-ES&page=1"
    populares_res = requests.get(populares_url).json()
    peliculas_catalogo = [get_movie_info(p['id']) for p in populares_res['results']]
    print(f"Películas del catálogo popular: {peliculas_catalogo}")  # Agregado para debug

    # Unimos ambas listas y quitamos duplicados
    peliculas = {p['id']: p for p in peliculas_usuario + peliculas_catalogo if p is not None}
    peliculas_df = pd.DataFrame(peliculas.values())

    # Armar columna combinada
    peliculas_df['features'] = peliculas_df['genres'] + ' ' + peliculas_df['overview']

    # Vectorizar y calcular similitud
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(peliculas_df['features'])
    similarity = cosine_similarity(tfidf_matrix)

    # Obtener índices de películas favoritas
    favoritos_idx = peliculas_df[peliculas_df['id'].isin([int(f) for f in favoritas])].index

    # Calcular promedio de similitud con las favoritas
    promedio_sim = similarity[favoritos_idx].mean(axis=0)

    # Ordenar por similitud (excluyendo las favoritas)
    peliculas_df['score'] = promedio_sim
    recomendadas = peliculas_df[~peliculas_df['id'].isin([int(f) for f in favoritas])]
    top_recomendadas = recomendadas.sort_values(by='score', ascending=False).head(5)

    print(f"Recomendaciones finales: {top_recomendadas[['id', 'title']]}")  # Agregado para debug

    # Retornar al frontend
    resultados = top_recomendadas[['id', 'title']].to_dict(orient='records')
    return jsonify(resultados)

@app.route('/recomendar_colaborativo', methods=['GET'])
def recomendar_colaborativo():
    usuario = request.args.get('usuario')
    if not usuario:
        return jsonify({'error': 'Usuario no especificado'}), 400

    if not os.path.exists(VALORACIONES_JSON):
        return jsonify({'error': 'No hay valoraciones'}), 404

    with open(VALORACIONES_JSON, 'r') as f:
        data = json.load(f)

    if usuario not in data:
        return jsonify({'error': 'Usuario no encontrado'}), 404

    # Convertir JSON a DataFrame
    df = pd.DataFrame(data).T.fillna(0)

    # Si sólo hay un usuario, no se puede recomendar por colaborativo
    if len(df) < 2:
        return jsonify({'error': 'No hay otros usuarios para comparar'}), 400

    # Similaridad entre usuarios
    sim_matrix = cosine_similarity(df)
    sim_df = pd.DataFrame(sim_matrix, index=df.index, columns=df.index)

    # Usuarios más parecidos al actual
    similares = sim_df[usuario].sort_values(ascending=False)[1:]  # quitarse a sí mismo

    # Peso de cada usuario similar
    recomendaciones = pd.Series(dtype='float64')
    for otro_usuario, similitud in similares.items():
        no_vistas = df.loc[otro_usuario][df.loc[usuario] == 0]
        recomendaciones = recomendaciones.add(no_vistas * similitud, fill_value=0)

    # Top recomendaciones
    top = recomendaciones.sort_values(ascending=False).head(5).index

    # Convertir a enteros
    peliculas_recomendadas = [int(mid) for mid in top if mid.isnumeric()]

    # Obtener títulos de TMDB
    resultado = []
    for pid in peliculas_recomendadas:
        info = get_movie_info(pid)
        if info:
            resultado.append({'id': pid, 'title': info['title']})

    return jsonify(resultado)

@app.route('/usuarios')
def usuarios():
    if not os.path.exists(VALORACIONES_JSON):
        return jsonify([])
    with open(VALORACIONES_JSON, 'r') as f:
        data = json.load(f)
    return jsonify(list(data.keys()))


#CHORO DE CHAT VALE VERGA
@app.route('/valoradas', methods=['GET'])
def valoradas():
    usuario = request.args.get('usuario')
    if not usuario:
        return jsonify({'error': 'Usuario no especificado'}), 400

    if not os.path.exists(VALORACIONES_JSON):
        return jsonify({'error': 'No hay valoraciones'}), 404

    with open(VALORACIONES_JSON, 'r') as f:
        data = json.load(f)

    if usuario not in data:
        return jsonify({'error': 'Usuario no encontrado'}), 404

    valoradas = []
    for movie_id, rating in data[usuario].items():
        info = get_movie_info(movie_id)
        if info:
            valoradas.append({
                'id': movie_id,
                'title': info['title'],
                'rating': rating
            })

    return jsonify(valoradas)

