import time

import nltk
import numpy as np
import pandas as pd
import re
import skfuzzy as fuzz
import skfuzzy.control as ctrl

from nltk import find
from nltk.sentiment import SentimentIntensityAnalyzer

pd.set_option('display.max_columns', None)


def csv_to_dataframe(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo {file_path} no existe. Por favor, verifica la ruta y vuelve a intentarlo.")

def preprocess_text(text):
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"@", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    # Elimina caracteres que no sean letras o espacios
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Elimina palabras de una sola letra
    text = ' '.join([word for word in text.split() if len(word) > 1])
    return text

def preprocess_dataframe(df):
    # Aplica la función preprocess_text a la columna 'sentence'
    df['sentence'] = df['sentence'].apply(preprocess_text)

    # Devuelve el DataFrame procesado
    return df

def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    execution_times = []

    # Aplica la función polarity_scores a cada sentencia en la columna 'sentence'
    for index, sentence in df['sentence'].items():
        start_time = time.time()
        scores = sia.polarity_scores(sentence)  # Obtención de puntajes
        end_time = time.time()
        tiempo_ejecucion = end_time - start_time

        df.at[index, 'puntaje_positivo'] = scores['pos']
        df.at[index, 'puntaje_negativo'] = scores['neg']
        execution_times.append(tiempo_ejecucion)

    df['tiempo_ejecucion'] = execution_times
    return df

def generate_fuzzy_logic():

    min_positive = df['puntaje_positivo'].min()  # Mínimo de puntaje positivo
    max_positive = df['puntaje_positivo'].max()  # Máximo de puntaje positivo
    min_negative = df['puntaje_negativo'].min()  # Mínimo de puntaje negativo
    max_negative = df['puntaje_negativo'].max()  # Máximo de puntaje negativo

    # Calcular el valor medio (mid) para positivo y negativo
    mid_positive = (min_positive + max_positive) / 2
    mid_negative = (min_negative + max_negative) / 2

    # Generar variables universales de entrada y salida
    positive = ctrl.Antecedent(np.arange(min_positive, max_positive + 0.1, 0.1), 'positive')
    negative = ctrl.Antecedent(np.arange(min_negative, max_negative + 0.1, 0.1), 'negative')
    output = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'output')

    # Generar funciones de membresía para positive
    positive['low'] = fuzz.trimf(positive.universe, [min_positive, min_positive, mid_positive])
    positive['medium'] = fuzz.trimf(positive.universe, [min_positive, mid_positive, max_positive])
    positive['high'] = fuzz.trimf(positive.universe, [mid_positive, max_positive, max_positive])

    # Generar funciones de membresía para negative
    negative['low'] = fuzz.trimf(negative.universe, [min_negative, min_negative, mid_negative])
    negative['medium'] = fuzz.trimf(negative.universe, [min_negative, mid_negative, max_negative])
    negative['high'] = fuzz.trimf(negative.universe, [mid_negative, max_negative, max_negative])

    # Generar funciones de membresía para output
    output['negative'] = fuzz.trimf(output.universe, [0, 0, 5])
    output['neutral'] = fuzz.trimf(output.universe, [0, 5, 10])
    output['positive'] = fuzz.trimf(output.universe, [5, 10, 10])

    # Reglas
    rules = [
        ctrl.Rule(positive['low'] & negative['low'], output['neutral']),
        ctrl.Rule(positive['medium'] & negative['low'], output['positive']),
        ctrl.Rule(positive['high'] & negative['low'], output['positive']),
        ctrl.Rule(positive['low'] & negative['medium'], output['negative']),
        ctrl.Rule(positive['medium'] & negative['medium'], output['neutral']),
        ctrl.Rule(positive['high'] & negative['medium'], output['positive']),
        ctrl.Rule(positive['low'] & negative['high'], output['negative']),
        ctrl.Rule(positive['medium'] & negative['high'], output['negative']),
        ctrl.Rule(positive['high'] & negative['high'], output['neutral'])
    ]

    # Sistema de control
    sentiment_ctrl = ctrl.ControlSystem(rules)
    # Inicia la simulación
    sentiment = ctrl.ControlSystemSimulation(sentiment_ctrl)

    return sentiment

def scale(resultado_inferencia):
    if resultado_inferencia < 3.3:
        return 'Negativo'
    elif 3.3 <= resultado_inferencia < 6.7:
        return 'Neutral'
    elif 6.7 <= resultado_inferencia:
        return 'Positivo'

def process_sentiments(df, sentiment):
    sentiment_output = []
    result = []

    for index, row in df.iterrows():
        sentiment.input['positive'] = row['puntaje_positivo']
        sentiment.input['negative'] = row['puntaje_negativo']
        # Realiza el proceso de inferencia difusa y defuzzificación
        sentiment.compute()  # Se hacen los cálculos de las reglas, se utiliza la interferencia Mamdani y se realiza la defuzzificación por el metodo centroide

        sentiment_output.append(sentiment.output['output'])
        result.append({
            'Oracion Original': row['sentence'],
            'Label Orignial': row['sentiment'],
            'Puntaje Positivo': row['puntaje_positivo'],
            'Puntaje Negativo': row['puntaje_negativo'],
            'Tiempo de Ejecucion': row['tiempo_ejecucion'],
            'Resultado de Inferencia': sentiment.output['output'],
        })

    df['resultado_inferencia'] = sentiment_output
    df['sentimiento'] = df['resultado_inferencia'].apply(scale)
    return df, result

def save_results(result, df):
    result_df = pd.DataFrame(result)
    result_df['Sentimiento'] = df['sentimiento']
    result_df.to_csv('data/resultados_sentimiento.csv', index=False)

def print_summary(df, result):
    sentiment_count = df['sentimiento'].value_counts()
    sentiment_count_dict = sentiment_count.to_dict()
    print(df)
    print('-----------------------------------------')
    print('Resumen de Sentimientos')
    print(sentiment_count_dict)
    average_execution_time = pd.DataFrame(result)['Tiempo de Ejecucion'].mean()
    print(f"Tiempo promedio de ejecución: {average_execution_time} segundos")

if __name__ == "__main__":
    file_path = "data/test_data.csv"
    df = csv_to_dataframe(file_path)

    try:
        find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')


    df_cleaned = preprocess_dataframe(df)
    df_sentiment = analyze_sentiment(df_cleaned)
    sentiment = generate_fuzzy_logic()
    df_sentiment, result = process_sentiments(df_sentiment, sentiment)
    save_results(result, df_sentiment)
    print_summary(df_sentiment, result)

