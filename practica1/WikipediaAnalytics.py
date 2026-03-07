from bs4 import BeautifulSoup
import pandas as pd
import regex as re

class WikipediaAnalytics:
    def __init__(self, list_of_strings):
        """
        Constructor: guarda la lista de strings.
        list_of_strings puede ser rutas a HTML, URLs, nombres, etc.
        """

        if not isinstance(list_of_strings, (list, tuple)):
            raise ValueError("list_of_strings debe ser una lista o tuple de strings.")

        if not all(isinstance(item, str) for item in list_of_strings):
            raise ValueError("Todos los elementos de list_of_strings deben ser strings.")

        self.sources = list_of_strings
        self.df = None  # Se inicializará en scrap()

    def scrap(self):
      """
      Inicializa el DataFrame.
      Aquí deberías procesar tus archivos HTML y llenar self.df.
      En este ejemplo se deja un df básico como placeholder.
      """

      """
      - self.df siempre existe al terminar y tiene mismas columnas en el mismo orden + cada fila representa un país
      """
      target_columns = [
         "Country Name",
         "Latitude (º)",
         "Longitude(º)",
         "Area (KM^2)",
         "Water (%)",
         "Population (hab.)",
         "Density (hab./km^2)",
         "GDP ($)",
         "Last Event Date",
      ]

      records = []

      self.df = pd.DataFrame.from_records(records, columns=target_columns)
      self.df["Last Event Date"] = pd.to_datetime(self.df["Last Event Date"], errors='coerce')

      return self

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
