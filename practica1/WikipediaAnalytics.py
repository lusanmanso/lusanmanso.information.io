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

      # 1. Abrir el archivo HTML local y parsearlo con bs4 para convertirlo en un objeto navegable
      for route in self.sources:
         with open(route, 'r', encoding='utf-8') as file:
            content = file.read()

         # Crear objeto beautifulsoup
         soup = BeautifulSoup(content, 'html.parser')

         # Buscar la tabla que en wikipedia es siempre una infobox
         table = soup.find('table', {'class': 'infobox'})

         country_name, area, water, population, density, GPD, last_event, latitude, longitude = "", "", "", "", "", "", "", "", ""

         for row in table.find_all('tr'):
            header = row.find('th')
            data = row.find('td')

            # verificar que tienen datos
            if header and data:
               header_text = header.get_text(strip=True).lower()

               # Buscar coincidencias
               if 'cabecera' in header_text:
                  country_name = data.get_text(strip=True)
               elif 'superficie' in header_text:
                  area = data.get_text(strip=True)
               elif 'agua' in header_text:
                  water = data.get_text(strip=True)
               elif 'poblacion total' in header_text:
                  population = data.get_text(strip=True)
               elif 'densidad' in header_text:
                  density = data.get_text(strip=True)
               elif 'pib' in header_text:
                  GPD = data.get_text(strip=True)
               elif 'formación' in header_text:
                  last_event = data.get_text(strip=True)
               elif 'coordenadas' in header_text:
                  latitude, longitude = data.find('span', class_='latitud').get_text(strip=True), data.find('span', class_='longitude').get_text(strip=True)

         # dicc para comprobacion
         raw_data = {
            'country_name': country_name,
            'area': area,
            'water': water,
            'population': population,
            'density': density,
            'GPD': GPD,
            'last_event': last_event,
            'latitude': latitude,
            'longitude': longitude
         }


         # print de comprobacion para espania
         if 'espania' in route.lower():
            print(f"Comprobacion de datos {route}")
            for key, value in raw_data.items():
               print(f"{key}: {value}")
            print("fin")

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

if __name__ == "__main__":
   files = ["files/espania_es.html"]
   scraper = WikipediaAnalytics(files)
   scraper.scrap()