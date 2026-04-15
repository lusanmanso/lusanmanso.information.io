from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from textblob import TextBlob

class Email:
    """
    Práctica 2 - Analítica de correos digitales.

    Implementa esta clase respetando exactamente:
    - el nombre de la clase
    - el nombre de los métodos
    - los parámetros de cada método
    - el tipo general de salida solicitado en los docstrings

    Puedes añadir funciones auxiliares privadas si las necesitas.
    """

    REQUIRED_COLUMNS = [
        "email_id", "date", "sender", "recipients", "cc", "subject", "body"
    ]

    def __init__(self, csv_path: str):

        """
        Constructor de la clase.

        Parámetros
        ----------
        csv_path : str
            Ruta al fichero CSV con los correos.

        Tareas mínimas esperadas
        ------------------------
        1. Guardar la ruta como atributo.
        2. Preparar los atributos que usarás durante la práctica:
            - self.df
            - self.graph
            - self.dictionary
            - self.corpus
            - self.lda_model
        """

        self.csv_path = csv_path
        self.df = None
        self.graph = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    """
    Lee el CSV, valida columnas y prepara el DataFrame base.
    Returns
    -------
    pd.DataFrame
        DataFrame procesado.
    """
    def load_data(self) -> pd.DataFrame:
        # Cargar el CSV en un DF de pandas
        df = pd.read_csv(self.csv_path)

        # Verificar que existen las cols obs
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError

        # Convertir date a datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Sustituir nulos de 'cc' a 'subject' y 'body' por cadena vacía
        cols_to_fill = ['cc', 'subject', 'body']
        df[cols_to_fill] = df[cols_to_fill].fillna('')

        # Crear una columan text combinando asunto y cuerpo
        df['text'] = df['subject'] + "" + df['body']

        # Almacenar el df
        self.df = df
        return self.df

    """
    Construye un grafo dirigido de interacciones entre remitentes y destinatarios.
    Notas
    -----
    - En 'recipients' y 'cc' puede haber varias direcciones separadas por ';'.
    Returns
    -------
    nx.DiGraph
        Grafo dirigido con pesos en las aristas.
    """
    def build_interaction_graph(self, include_cc: bool = True) -> nx.DiGraph:
        self.graph = nx.DiGraph() # Crear un digrafo y guardarlo en el atrib de la clase

        if self.df is None or self.df.empty:
            return self.graph

        # iterar sobre cada correo (fila de df)
        for _, row in self.df.iterrows():
            if pd.isna(row['sender']):
                continue

            sender = str(row['sender']).strip().lower()

            if not sender or sender == 'nan' : # si no hay remitente pues saltar
                continue

            def process_dest(destinations_str):
                if not destinations_str:
                    return

                # separar por ';' y quitar espacios en blanco
                destinations_str = destinations_str.replace(',', ';') # en caso de que usen ',' como separador
                destinations = [d.strip().lower() for d in str(destinations_str).split(';') if d.strip]

                for dest in destinations:
                    if self.graph.has_edge(sender, dest):
                        self.graph[sender][dest]['weight'] += 1 # si la arista existe peso +1
                    else: # sino se crea con peso 1
                        self.graph.add_edge(sender, dest, weight=1)

            process_dest(row['recipients'])

            # si tuviese destinatarios en copia
            if include_cc:
                process_dest(row['cc'])

        return self.graph

    """
    Calcula sentimiento con TextBlob.
    Returns
    -------
    pd.DataFrame
        DataFrame actualizado con las columnas de sentimiento.
    """
    def analyze_sentiment(self, text_column: str = "text") -> pd.DataFrame:
        # verif basica
        if self.df is None or text_column not in self.df.columns:
            raise ValueError(f"El DataFrame no está inicializado o falta la columna '{text_column}'")

        # subfun para aplicar TextBlob
        def extract_sentiment(text):
            blob = TextBlob(str(text))
            # devolver ambos valores
            return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])

        # aplicar y guardar resultados
        self.df[['polarity', 'subjectivity']] = self.df[text_column].apply(extract_sentiment)

        # asignar etiquetas de sentimiento
        def categorize_polarity(pol):
            if pol > 0.1:
                return 'positive'
            elif pol < -0.1:
                return 'negative'
            else:
                return 'neutral'

        self.df['sentiment_label'] = self.df['polarity'].apply(categorize_polarity)

        return self.df

    """
    Preprocesa un texto para LDA.

    Libertad del alumno:
    - Puedes decidir cómo tokenizar.
    - Puedes eliminar stopwords.
    - Puedes aplicar stemming o lematización si lo deseas.

    Restricción:
    - Debe devolver una lista de tokens lista para construir el corpus.

    Returns
    -------
    List[str]
    """
    def preprocess_text_for_lda(self, text: str) -> List[str]:
        return simple_preprocess(str(text), deacc=True) # tokenizar y eliminar acentos


    """
    Entrena un modelo LDA con gensim.

    Returns
    -------
    tuple
        (lda_model, dictionary, corpus)
    """
    def train_topic_model(
        self,
        num_topics: int = 3,
        passes: int = 15,
        random_state: int = 42
    ) -> Tuple[LdaModel, Dictionary, List[List[tuple]]]:

        if self.df is None or 'text' not in self.df.columns:
            raise ValueError("DataFrame no inicializado o falta la columna 'text'")

        # preprocesar textos
        processed_texts = self.df['text'].apply(self.preprocess_text_for_lda).tolist()
        # construir dictionary del gensim mapeando cada palabra a un unico id
        self.dictionary = Dictionary(processed_texts)
        # construir corpus bag-of-words para contar la feecuencia de cada palabra en cada correo
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]

        # entrenar el lda
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            random_state=random_state,
            passes=passes
        )

        # guardarlo como atrib y devolverlo
        return self.lda_model, self.dictionary, self.corpus


    """
    Asigna a cada correo su tema dominante.

    Returns
    -------
    pd.DataFrame
        DataFrame con asignación temática.
    """
    def assign_topics(self) -> pd.DataFrame:
        if self.lda_model is None or self.corpus is None:
            raise ValueError("No LDA o no corpus")

        dominant_topics = []
        topic_keywords = []

        for bow in self.corpus:
            topic_probs = self.lda_model[bow] # devuelve (topic_id, prob) por tema

            dominant_topic = max(topic_probs, key=lambda x: x[1])[0] # tema con mayor probabilidad
            dominant_topics.append(dominant_topic)

            # obtener las palabras clave del tema dominante
            words = [word for word, weight in self.lda_model.show_topic(dominant_topic, topn=5)]
            topic_keywords.append(", ".join(words))

        # añadimos nuevas cols al df
        self.df['dominant_topic'] = dominant_topics
        self.df['topic_keywords'] = topic_keywords

        return self.df


    """
    Genera un resumen estructurado por tema.

    Returns
    -------
    pd.DataFrame
    """
    def get_topic_report(self, topn_words: int = 5) -> pd.DataFrame:
        if 'dominant_topic' not in self.df.columns or 'polarity' not in self.df.columns:
            raise ValueError("Falta asignación de temas o modelo LDA")

        # agrupamos por el tema dominante
        grouped = self.df.groupby('dominant_topic')

        # calculamos métricas requeridas
        report_data = []
        for topic_id, group in grouped:
            keywords = group['topic_keywords'].iloc[0]
            num_emails = len(group)
            mean_polarity = group['polarity'].mean()

            report_data.append({
                'topic_id': topic_id,
                'keywords': keywords,
                'num_emails': num_emails,
                'mean_polarity': mean_polarity
            })

        report_df = pd.DataFrame(report_data).sort_values(by='topic_id').reset_index(drop=True)
        return report_df

    """
    Devuelve los correos enviados por un remitente concreto.
    """
    def get_emails_by_sender(self, sender: str) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("DataFrame no inicializado")

        return self.df[self.df['sender'] == sender]

    """
    Devuelve los correos asociados a un tema concreto.
    """
    def get_emails_by_topic(self, topic_id: int) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("DataFrame no inicializado")

        return self.df[self.df['dominant_topic'] == topic_id]

    """
    Devuelve métricas básicas del grafo.

    Formato mínimo esperado:
    {
        "num_nodes": ...,
        "num_edges": ...,
        "density": ...
    }
    """
    def graph_metrics(self) -> Dict[str, float]:
        if self.graph is None: # si no esta construido el grafo devolver cero
            return {"num_nodes": 0.0, "num_edges": 0.0, "density": 0.0}

        # calculamos y devolvemos métricas solicitadas
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph)
        }
