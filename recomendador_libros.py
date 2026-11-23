"""
Sistema de recomendación de libros mediante filtrado colaborativo item-item (KNN).

Prueba técnica de acceso. Enunciado:
    1. Crear el sistema de recomendación.
    2. Validar el resultado mostrando las recomendaciones para 3 o 4 usuarios concretos.

Funcionamiento general:
    - Se limpian los tres CSV del dataset Book-Crossing (libros, puntuaciones, usuarios).
    - Se construye una matriz libro x usuario con las valoraciones explícitas (1-10).
    - Un modelo NearestNeighbors con distancia coseno encuentra, para cada libro,
      los libros valorados de forma más parecida por los mismos usuarios.
    - Para un usuario, se buscan los vecinos de los libros que puntuó con 7 o más
      y se suman las afinidades (1 - distancia). Los libros con mayor suma se recomiendan.

Los CSV deben estar en la carpeta data/ (ver README.md).
"""

import html
import random
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA_PROYECTO / "data"
ARCHIVO_HTML = CARPETA_PROYECTO / "recomendaciones.html"

# Solo se tienen en cuenta usuarios con MÁS de este número de valoraciones.
UMBRAL_VALORACIONES_USUARIO = 5

# Solo se tienen en cuenta libros con AL MENOS este número de valoraciones.
# Se probó con 200 y 50; 10 da un buen equilibrio entre resultados realistas y tiempo de cálculo.
MIN_VALORACIONES_LIBRO = 10

# Nota a partir de la cual se considera que al usuario le ha gustado un libro (un notable).
NOTA_MINIMA_GUSTO = 7

# Vecinos que se piden al modelo por cada libro (incluye al propio libro, que se descarta).
# Con menos vecinos el ranking final se empobrece.
N_VECINOS = 10

# Usuarios fijos para la validación por consola.
USUARIOS_PRUEBA = [11676, 35859, 212898, 98391]


# ---------------------------------------------------------------------------
# Carga y limpieza de datos
# ---------------------------------------------------------------------------

def cargar_datos():
    """Lee los tres CSV del dataset."""
    # dtype=str evita el aviso de tipos mezclados en Books.csv (p. ej. en el año de publicación).
    df_libros = pd.read_csv(CARPETA_DATOS / "Books.csv", dtype=str)
    # El ISBN se lee como texto para que coincida con el de Books.csv
    # (si pandas lo interpretara como número, perdería los ceros iniciales).
    df_puntuaciones = pd.read_csv(CARPETA_DATOS / "Ratings.csv", dtype={"ISBN": str})
    df_usuarios = pd.read_csv(CARPETA_DATOS / "Users.csv")
    return df_libros, df_puntuaciones, df_usuarios


def limpiar_libros(df_libros):
    """Rellena los pocos nulos de Books.csv (271.360 filas, 8 columnas).

    Nulos detectados: 2 en autor, 2 en editorial y 3 en la imagen grande.
    Son tan pocos que basta con un valor genérico en lugar de eliminar filas.
    """
    print("Nulos en libros:\n", df_libros.isnull().sum(), "\n", "-" * 50, sep="")

    df_libros["Book-Author"] = df_libros["Book-Author"].fillna("desconocido")
    df_libros["Publisher"] = df_libros["Publisher"].fillna("desconocido")
    df_libros["Image-URL-L"] = df_libros["Image-URL-L"].fillna("sin imagen")
    return df_libros


def explorar_puntuaciones(df_puntuaciones):
    """Muestra nulos y la distribución de notas de Ratings.csv (1.149.780 filas, 3 columnas).

    No hay nulos. El histograma muestra que la mayoría de valoraciones son 0,
    que en este dataset significa valoración implícita (el usuario interactuó
    con el libro pero no le puso nota).
    """
    print("Nulos en puntuaciones:\n", df_puntuaciones.isnull().sum(), "\n", "-" * 50, sep="")

    df_puntuaciones["Book-Rating"].hist(bins=10, color="skyblue", edgecolor="black")
    plt.title("Distribución de las valoraciones")
    plt.xlabel("Nota")
    plt.ylabel("Número de valoraciones")
    plt.show()


def limpiar_usuarios(df_usuarios):
    """Corrige las edades de Users.csv (278.858 filas, 3 columnas).

    Problemas detectados:
        - 110.762 nulos en la edad.
        - Más de 800 usuarios con edades imposibles (menos de 1 año o más de 100).

    Solución:
        1. Recortar (clip) por percentiles: el 5 % deja un mínimo de 16 años y el
           99,74 % un máximo plausible de 88 años.
        2. Rellenar los nulos con la moda. La media se descarta porque unos pocos
           valores extremos la desplazarían demasiado.

    Nota: la edad no interviene en el recomendador; esta limpieza forma parte del
    análisis exploratorio del dataset.
    """
    print("Nulos en usuarios:\n", df_usuarios.isnull().sum(), sep="")
    print(df_usuarios["Age"].describe())

    limite_inferior = df_usuarios["Age"].quantile(0.05)
    limite_superior = df_usuarios["Age"].quantile(0.9974)
    df_usuarios["Age"] = df_usuarios["Age"].clip(lower=limite_inferior, upper=limite_superior)

    imputer_moda = SimpleImputer(strategy="most_frequent")
    df_usuarios["Age"] = imputer_moda.fit_transform(df_usuarios[["Age"]]).ravel()

    print("Edades tras la limpieza:\n", df_usuarios["Age"].describe(), "\n", "-" * 50, sep="")
    return df_usuarios


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def construir_modelo(df_puntuaciones):
    """Filtra los datos, crea la matriz libro x usuario y entrena el KNN."""
    # 1. Usuarios activos: más de UMBRAL_VALORACIONES_USUARIO valoraciones.
    conteo_usuarios = df_puntuaciones["User-ID"].value_counts()
    usuarios_activos = conteo_usuarios[conteo_usuarios > UMBRAL_VALORACIONES_USUARIO].index
    df_filtrado = df_puntuaciones[df_puntuaciones["User-ID"].isin(usuarios_activos)]

    # 2. Libros populares entre esos usuarios: al menos MIN_VALORACIONES_LIBRO valoraciones.
    conteo_libros = df_filtrado["ISBN"].value_counts()
    libros_populares = conteo_libros[conteo_libros >= MIN_VALORACIONES_LIBRO].index
    df_filtrado = df_filtrado[df_filtrado["ISBN"].isin(libros_populares)]

    # 3. Solo valoraciones explícitas (la nota 0 es implícita y no indica gusto).
    df_filtrado = df_filtrado[df_filtrado["Book-Rating"] > 0]

    # 4. Matriz libro x usuario. Las celdas vacías (usuario que no valoró ese libro) valen 0.
    matriz = df_filtrado.pivot(index="ISBN", columns="User-ID", values="Book-Rating").fillna(0)

    # 5. Casi todas las celdas son 0, así que se guarda como matriz dispersa (CSR):
    #    solo almacena los valores distintos de cero. Sin filtrar, la tabla
    #    habría tenido unos 35.000 millones de celdas y no cabía en memoria.
    matriz_dispersa = csr_matrix(matriz.values)

    print(f"Filas originales: {df_puntuaciones.shape[0]}")
    print(f"Filas tras filtrar: {df_filtrado.shape[0]}")
    print(f"Dimensiones de la matriz (libros x usuarios): {matriz.shape}")

    # 6. Distancia coseno: compara el patrón de valoraciones de dos libros
    #    independientemente de cuántas valoraciones tenga cada uno.
    #    "brute" calcula todas las distancias; es el método compatible con coseno y matrices dispersas.
    modelo = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=N_VECINOS)
    modelo.fit(matriz_dispersa)

    return matriz, matriz_dispersa, modelo


def calcular_recomendaciones(user_id, df_puntuaciones, matriz, matriz_dispersa, modelo):
    """Devuelve una lista [(isbn, afinidad_total), ...] ordenada de mayor a menor afinidad.

    Para cada libro que el usuario puntuó con NOTA_MINIMA_GUSTO o más, se buscan sus
    libros vecinos. Cada vecino suma su afinidad (1 - distancia coseno), de modo que
    un libro parecido a varios de los favoritos del usuario acumula más puntuación.
    """
    historial = df_puntuaciones[df_puntuaciones["User-ID"] == user_id]

    # Libros que le gustaron: son el punto de partida de la búsqueda.
    isbn_favoritos = historial.loc[historial["Book-Rating"] >= NOTA_MINIMA_GUSTO, "ISBN"]

    # Libros que ya conoce (con cualquier nota): no se le deben recomendar.
    # Se usa un set porque comprobar "x in set" es mucho más rápido que en una lista.
    isbn_conocidos = set(historial["ISBN"])

    afinidades = {}
    for isbn in isbn_favoritos:
        if isbn not in matriz.index:
            continue  # el libro no superó los filtros del modelo

        indice_libro = matriz.index.get_loc(isbn)
        distancias, indices = modelo.kneighbors(matriz_dispersa[indice_libro])

        for distancia, indice_vecino in zip(distancias[0], indices[0]):
            isbn_vecino = matriz.index[indice_vecino]
            if isbn_vecino == isbn or isbn_vecino in isbn_conocidos:
                continue
            afinidades[isbn_vecino] = afinidades.get(isbn_vecino, 0) + (1 - distancia)

    return sorted(afinidades.items(), key=lambda par: par[1], reverse=True)


# ---------------------------------------------------------------------------
# Presentación de resultados
# ---------------------------------------------------------------------------

def mostrar_en_consola(user_id, recomendaciones, info_libros, num_recomendaciones=5):
    """Imprime las recomendaciones de un usuario: título, autor, editorial y afinidad."""
    print(f"\n{'=' * 50}\nRecomendaciones para el usuario {user_id}:\n")

    if not recomendaciones:
        print("No hay recomendaciones: el usuario no tiene libros valorados con nota suficiente.")
        return

    mostrados = 0
    for isbn, afinidad in recomendaciones:
        if mostrados == num_recomendaciones:
            break
        if isbn not in info_libros.index:
            continue  # ISBN presente en Ratings.csv pero no en Books.csv
        libro = info_libros.loc[isbn]
        mostrados += 1
        print(f"{mostrados}: {libro['Book-Title']} | {libro['Book-Author']} | "
              f"{libro['Publisher']} (afinidad: {afinidad:.2f})")


def bloque_html_usuario(user_id, recomendaciones, info_libros, num_recomendaciones=4):
    """Genera el bloque HTML con las portadas y datos de las recomendaciones de un usuario.

    html.escape() evita que títulos con caracteres como < o ' rompan la página.
    """
    partes = [
        "<div style='border:2px solid #cba6f7; border-radius:12px; padding:20px; margin-bottom:30px;'>",
        f"<h2 style='color:#cba6f7;'>Usuario {user_id}</h2>",
    ]

    contador = 0
    for isbn, afinidad in recomendaciones:
        if contador == num_recomendaciones:
            break
        if isbn not in info_libros.index:
            continue
        libro = info_libros.loc[isbn]
        partes.append(
            "<div style='background:#1e1e2e; padding:10px; margin:10px; border-radius:8px;'>"
            f"<img src='{html.escape(str(libro['Image-URL-M']))}' style='width:80px; float:left; margin-right:10px;'>"
            f"<p style='color:white;'>{html.escape(str(libro['Book-Title']))}</p>"
            f"<p style='color:green;'>{html.escape(str(libro['Book-Author']))}</p>"
            f"<p style='color:lightblue;'>{html.escape(str(libro['Year-Of-Publication']))} - "
            f"{html.escape(str(libro['Publisher']))}</p>"
            f"<p style='color:pink;'>afinidad: {afinidad:.2f}</p>"
            "<div style='clear:both;'></div>"
            "</div>"
        )
        contador += 1

    if contador == 0:
        partes.append("<p style='color:white;'>Sin recomendaciones para este usuario.</p>")

    partes.append("</div>")
    return "".join(partes)


def generar_html(usuarios, df_puntuaciones, matriz, matriz_dispersa, modelo, info_libros):
    """Crea un HTML con las recomendaciones de varios usuarios y lo abre en el navegador.

    La plantilla HTML se elaboró con ayuda de una herramienta de IA (Gemini), ya que
    HTML no formaba parte del temario del curso.
    """
    partes = [
        "<html><head><meta charset='utf-8'><title>Recomendaciones</title></head>",
        "<body style='background-color:#1a1b26; font-family:arial; padding:20px;'>",
        "<h1 style='color:white;'>Recomendaciones</h1>",
    ]
    for usuario in usuarios:
        recomendaciones = calcular_recomendaciones(usuario, df_puntuaciones, matriz, matriz_dispersa, modelo)
        partes.append(bloque_html_usuario(usuario, recomendaciones, info_libros))
    partes.append("</body></html>")

    ARCHIVO_HTML.write_text("".join(partes), encoding="utf-8")
    webbrowser.open(ARCHIVO_HTML.as_uri())
    print(f"\nHTML generado en {ARCHIVO_HTML}")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    df_libros, df_puntuaciones, df_usuarios = cargar_datos()

    df_libros = limpiar_libros(df_libros)
    explorar_puntuaciones(df_puntuaciones)
    limpiar_usuarios(df_usuarios)

    matriz, matriz_dispersa, modelo = construir_modelo(df_puntuaciones)

    # Tabla de consulta por ISBN para buscar los datos de cada libro de forma directa.
    info_libros = df_libros.drop_duplicates(subset="ISBN").set_index("ISBN")

    # Validación 1: usuarios fijos, resultados por consola.
    for usuario in USUARIOS_PRUEBA:
        recomendaciones = calcular_recomendaciones(usuario, df_puntuaciones, matriz, matriz_dispersa, modelo)
        mostrar_en_consola(usuario, recomendaciones, info_libros)

    # Validación 2: 4 usuarios al azar entre los 100 más activos, resultados en HTML con portadas.
    top_100 = df_puntuaciones["User-ID"].value_counts().head(100).index.tolist()
    usuarios_aleatorios = random.sample(top_100, 4)
    generar_html(usuarios_aleatorios, df_puntuaciones, matriz, matriz_dispersa, modelo, info_libros)


if __name__ == "__main__":
    main()
