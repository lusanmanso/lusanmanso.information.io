from bs4 import BeautifulSoup
import pandas as pd
import regex as re

class WikipediaAnalytics:
    def __init__(self, list_of_strings):
        """
        Constructor: guarda la lista de strings.
        list_of_strings puede ser rutas a HTML, URLs, nombres, etc.
        """
        self.sources = list_of_strings
        self.df = None  # Se inicializará en scrap()

    def scrap(self):
        """
        Inicializa el DataFrame.
        Aquí deberías procesar tus archivos HTML y llenar self.df.
        En este ejemplo se deja un df básico como placeholder.
        """
        pass

    
    def select_row_by_value(self, col_name, value):
        """
        Devuelve la fila (o filas) cuyo valor en col_name coincide con value.
        """
        pass

    def get_columns(self, col_names):
        """
        Recibe una columna o lista de columnas y devuelve esa parte del DataFrame.
        """
        pass

    def aggregate_column(self, col_name, operation):
        """
        Agrega una columna numérica según una operación dada:
        - 'max'
        - 'min'
        - 'mean'
        Devuelve el resultado como un float.
        """
        pass