import os
import string
import numpy as np
from nltk import download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Descarga de recursos necesarios de NLTK
download('punkt')
download('stopwords')

# Funciones de utilidad
def read_files_in_directory(directory):
    files_contents = []
    file_names = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    files_contents.append(file.read())
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='windows-1252') as file:
                    files_contents.append(file.read())
            file_names.append(filename)
    return file_names, files_contents


def preprocess(text):
    """ Procesa el texto aplicando normalización, eliminación de stopwords y stemming. """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def generate_vector_space_models(original_texts, suspicious_texts):
    """ Genera modelos de espacio vectorial para textos originales y sospechosos. """
    vectorizer = CountVectorizer(analyzer='word')
    # Combina todos los textos para crear un espacio vectorial común
    combined_texts = original_texts + suspicious_texts
    vectorizer.fit(combined_texts)  # Ajusta el vectorizador a todos los textos
    original_vectors = vectorizer.transform(original_texts)  # Transforma los textos originales
    suspicious_vectors = vectorizer.transform(suspicious_texts)  # Transforma los textos sospechosos
    return original_vectors, suspicious_vectors, vectorizer.get_feature_names_out()


# En el flujo principal del script:
if __name__ == "__main__":
    # Directorios de los documentos
    path_to_originals = './TextosOriginales'
    path_to_suspicious = './TextosConPlagio'
    
    # Cargar y procesar los documentos
    original_filenames, original_texts = read_files_in_directory(path_to_originals)
    suspicious_filenames, suspicious_texts = read_files_in_directory(path_to_suspicious)

    # Preprocesar todos los textos
    processed_originals = [preprocess(text) for text in original_texts]
    processed_suspicious = [preprocess(text) for text in suspicious_texts]

    # Crear modelos de espacio vectorial común
    original_vectors, suspicious_vectors, feature_names = generate_vector_space_models(processed_originals, processed_suspicious)

    # Calcular similitud de coseno
    similarities = cosine_similarity(suspicious_vectors, original_vectors)

    # Reportar resultados
    threshold = 0.2  # Umbral de similitud
    for i, filename in enumerate(suspicious_filenames):
        print("\n")
        print(f"Top coincidencias para el archivo sospechoso '{filename}':")
        file_similarities = [(original_filenames[j], similarities[i, j]) for j in range(len(original_filenames))]
        # Ordenar las similitudes y tomar el top 5
        top_5_similarities = sorted(file_similarities, key=lambda x: x[1], reverse=True)[:5]
        for original_file, sim in top_5_similarities:
            if sim > threshold:
                print(f"- {original_file} con una similitud del {sim*100:.2f}%")
        