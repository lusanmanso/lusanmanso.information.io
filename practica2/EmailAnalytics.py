from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
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

    def load_data(self) -> pd.DataFrame:
        """
        Lee el CSV, valida columnas y prepara el DataFrame base.

        Requisitos mínimos:
        - Cargar el CSV en un DataFrame de pandas.
        - Verificar que existen las columnas obligatorias.
        - Convertir 'date' a datetime.
        - Sustituir nulos de 'cc', 'subject' y 'body' por cadena vacía.
        - Crear una columna 'text' combinando asunto y cuerpo.
        - Almacenar el DataFrame en self.df.

        Returns
        -------
        pd.DataFrame
            DataFrame procesado.
        """
        pass

    def build_interaction_graph(self, include_cc: bool = True) -> nx.DiGraph:
        """
        Construye un grafo dirigido de interacciones entre remitentes y destinatarios.

        Requisitos mínimos:
        - Crear un nx.DiGraph.
        - Añadir una arista desde el remitente hacia cada destinatario principal.
        - Si include_cc=True, incluir también las direcciones en copia.
        - Si una arista ya existe, incrementar su peso.
        - Almacenar el grafo en self.graph.

        Notas
        -----
        - En 'recipients' y 'cc' puede haber varias direcciones separadas por ';'.

        Returns
        -------
        nx.DiGraph
            Grafo dirigido con pesos en las aristas.
        """
        pass

    def analyze_sentiment(self, text_column: str = "text") -> pd.DataFrame:
        """
        Calcula sentimiento con TextBlob.

        Requisitos mínimos:
        - Aplicar TextBlob sobre la columna indicada.
        - Crear las columnas:
          * polarity (float)
          * subjectivity (float)
          * sentiment_label (str)
        - El criterio de etiquetado puede ser libre, pero debe generar
          al menos las clases: 'positive', 'neutral' y 'negative'.

        Returns
        -------
        pd.DataFrame
            DataFrame actualizado con las columnas de sentimiento.
        """
        pass

    def preprocess_text_for_lda(self, text: str) -> List[str]:
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
        pass

    def train_topic_model(
        self,
        num_topics: int = 3,
        passes: int = 15,
        random_state: int = 42
    ) -> Tuple[LdaModel, Dictionary, List[List[tuple]]]:
        """
        Entrena un modelo LDA con gensim.

        Requisitos mínimos:
        - Preprocesar los textos.
        - Construir un Dictionary de gensim.
        - Construir el corpus bag-of-words.
        - Entrenar un LdaModel.
        - Guardar dictionary, corpus y lda_model como atributos.

        Returns
        -------
        tuple
            (lda_model, dictionary, corpus)
        """
        pass

    def assign_topics(self) -> pd.DataFrame:
        """
        Asigna a cada correo su tema dominante.

        Requisitos mínimos:
        - Utilizar self.lda_model y self.corpus.
        - Crear, al menos, estas columnas:
          * dominant_topic (int)
          * topic_keywords (str)

        Returns
        -------
        pd.DataFrame
            DataFrame con asignación temática.
        """
        pass

    def get_topic_report(self, topn_words: int = 5) -> pd.DataFrame:
        """
        Genera un resumen estructurado por tema.

        Formato mínimo esperado del DataFrame de salida:
        - topic_id
        - keywords
        - num_emails
        - mean_polarity

        Puedes añadir columnas extra si aportan valor.

        Returns
        -------
        pd.DataFrame
        """
        pass

    def get_emails_by_sender(self, sender: str) -> pd.DataFrame:
        """
        Devuelve los correos enviados por un remitente concreto.
        """
        pass

    def get_emails_by_topic(self, topic_id: int) -> pd.DataFrame:
        """
        Devuelve los correos asociados a un tema concreto.
        """
        pass

    def graph_metrics(self) -> Dict[str, float]:
        """
        Devuelve métricas básicas del grafo.

        Formato mínimo esperado:
        {
            "num_nodes": ...,
            "num_edges": ...,
            "density": ...
        }
        """
        pass
