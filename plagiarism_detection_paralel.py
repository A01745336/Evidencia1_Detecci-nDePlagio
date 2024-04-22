import os
import string
import time
import numpy as np
from nltk import download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from concurrent.futures import ProcessPoolExecutor

# Descarga de recursos necesarios de NLTK
inicio = time.time()
download('punkt')
download('stopwords')


def read_and_preprocess_file(file_path):
    """Lee y preprocesa un solo archivo."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='windows-1252') as file:
            content = file.read()
    # Preprocesar el contenido del archivo
    processed_content = preprocess(content)
    return processed_content


def parallel_process_files(directory):
    """Procesa en paralelo todos los archivos de texto en un directorio."""
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    processed_texts = []
    # Ejecuta la lectura y el preprocesamiento en paralelo
    with ProcessPoolExecutor(max_workers=6) as executor:
        processed_texts = list(executor.map(read_and_preprocess_file, file_paths, chunksize=10))
    return file_paths, processed_texts


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
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    # Combina todos los textos para crear un espacio vectorial común
    combined_texts = original_texts + suspicious_texts
    vectorizer.fit(combined_texts)  # Ajusta el vectorizador a todos los textos
    original_vectors = vectorizer.transform(original_texts)  # Transforma los textos originales
    suspicious_vectors = vectorizer.transform(suspicious_texts)  # Transforma los textos sospechosos
    return original_vectors, suspicious_vectors, vectorizer.get_feature_names_out()


def evaluate_performance(similarities, threshold, ground_truth):
    """
    Evalúa el rendimiento de la herramienta de detección de plagio.
    similarities: matriz de similitudes entre documentos sospechosos y originales.
    threshold: el umbral de similitud para considerar un documento como plagiado.
    ground_truth: dict con clave = nombre del documento sospechoso y valor = bool indicando si es plagiado.
    
    Retorna un dict con las métricas TP, FP, TN, FN.
    """
    TP = FP = TN = FN = 0
    for i, file_path in enumerate(suspicious_filenames):
        # Extrae el nombre del archivo de la ruta
        susp_filename = os.path.basename(file_path)
        is_plagiarized = ground_truth.get(susp_filename)
        # Considera el documento plagiado si alguna similitud supera el umbral.
        detected_as_plagiarized = any(sim > threshold for sim in similarities[i])
        
        if is_plagiarized and detected_as_plagiarized:
            TP += 1
        elif not is_plagiarized and detected_as_plagiarized:
            FP += 1
        elif is_plagiarized and not detected_as_plagiarized:
            FN += 1
        elif not is_plagiarized and not detected_as_plagiarized:
            TN += 1

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}


def generate_report(performance_metrics, similarities, ground_truth_labels):
    """
    Genera un informe de rendimiento basado en las métricas dadas.
    performance_metrics: dict con TP, FP, TN, FN.
    similarities: lista de valores de similitud entre documentos.
    ground_truth_labels: lista de etiquetas de verdad fundamental (0 para no plagiado, 1 para plagiado).
    """
    TP = performance_metrics["TP"]
    FP = performance_metrics["FP"]
    TN = performance_metrics["TN"]
    FN = performance_metrics["FN"]
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Calcular AUC
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, similarities)
    roc_auc = auc(fpr, tpr)

    report = (
        f"True Positives: {TP}\n"
        f"False Positives: {FP}\n"
        f"True Negatives: {TN}\n"
        f"False Negatives: {FN}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"F1 Score: {f1_score:.2f}\n"
        f"AUC (ROC): {roc_auc:.2f}\n"
    )
    print(report)
    # Puedes añadir código aquí para guardar el informe en un archivo si es necesario.



# En el flujo principal del script:
if __name__ == "__main__":
    # Directorios de los documentos
    path_to_originals = './TextosOriginales'
    path_to_suspicious = './TextosConPlagio'
    
    # Paralelizar el procesamiento de archivos originales y sospechosos
    original_filenames, processed_originals = parallel_process_files(path_to_originals)
    suspicious_filenames, processed_suspicious = parallel_process_files(path_to_suspicious)

    # Crear modelos de espacio vectorial común
    original_vectors, suspicious_vectors, feature_names = generate_vector_space_models(processed_originals, processed_suspicious)

    # Calcular similitud de coseno
    similarities = cosine_similarity(suspicious_vectors, original_vectors)

    # Reportar resultados
    threshold = 0.2  # Umbral de similitud


    # Información de ground truth para evaluación
    ground_truth = {
        'FID-01.txt': True,
        'FID-02.txt': True,
        'FID-03.txt': True,
        'FID-04.txt': True,
        'FID-05.txt': True,
        'FID-06.txt': True,
        'FID-07.txt': True,
        'FID-08.txt': True,
        'FID-09.txt': True,
        'FID-10.txt': True,
        'FID-11.txt': False,
    }

    # Evaluación del rendimiento
    performance_metrics = evaluate_performance(similarities, threshold, ground_truth)
    
    # Aplanar las similitudes y preparar las etiquetas de ground truth
    all_similarities = []
    ground_truth_labels = []

    for i, file_path in enumerate(suspicious_filenames):
        susp_filename = os.path.basename(file_path)
        for j in range(len(original_filenames)):
            all_similarities.append(similarities[i][j])
            ground_truth_labels.append(1 if ground_truth.get(susp_filename) else 0)

    # Llamada a generate_report
    generate_report(performance_metrics, all_similarities, ground_truth_labels)

    for i, filename in enumerate(suspicious_filenames):
        print("\n")
        print(f"Top coincidencias para el archivo sospechoso '{filename}':")
        file_similarities = [(original_filenames[j], similarities[i, j]) for j in range(len(original_filenames))]
        # Ordenar las similitudes y tomar el top 5
        top_5_similarities = sorted(file_similarities, key=lambda x: x[1], reverse=True)[:5]
        for original_file, sim in top_5_similarities:
            if sim > threshold:
                print(f"- {original_file} con una similitud del {sim*100:.2f}%")

    fin = time.time()
    print(f'\n El tiempo de ejecución fue de : {fin-inicio}')
        