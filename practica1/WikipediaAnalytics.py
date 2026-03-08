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

   # Para los numeros
   def clean_number(self, value):
      if not value:
         return None

      match = re.search(r'([\d\s\.,\xa0\u202f\u200b\n\t]+)', value)
      if match:
         raw_number = match.group(1)
         cleaned_number = re.sub(r'[^\d,]', '', raw_number)
         number = cleaned_number.replace(',', '.')

      return float(number)

   # Para las fechas
   def clean_date(self, value):
      if not value:
         return None

      months = { 'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12' }

      # (1 o dos numeros) + " de" + (letras del mes) + " de" + (4 numeros del año)
      match = re.findall(r'(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})', value.lower())

      if match:
         latest_date = match[-1]  # coger la última fecha encontrada

         day = latest_date[0].zfill(2)
         month_name = latest_date[1]
         year = latest_date[2]

         month_number = months[month_name]

      return f"{day}/{month_number}/{year}"

   def scrap(self):

      # 1. Abrir el archivo HTML local y parsearlo con bs4 para convertirlo en un objeto navegable
      for route in self.sources:
         with open(route, 'r', encoding='utf-8') as file:
            content = file.read()

         # Crear objeto beautifulsoup
         soup = BeautifulSoup(content, 'html.parser')

         match_title = re.search(r'"wgTitle":"([^"]+)"', content) # !: nombre del pais ocupa dos columnas no tiene <td>, uso regex

         # Buscar la tabla que en wikipedia es siempre una infobox
         table = soup.find('table', {'class': 'infobox'})

         area, water, population, density, GPD, last_event, latitude, longitude = "", "", "", "", "", "", "", ""

         current_section = None  # evitar UnboundLocalError, sirve para marcar por secciones porque el valor no está en la misma fila

         for row in table.find_all('tr'):
            cells = row.find_all(['th', 'td']) # celdas porque header y data no funcionan

            # atrapar flags de solo una columna
            row_text = row.get_text(strip=True).lower()

            if 'superficie' in row_text:
               current_section = 'area'
            elif 'población' in row_text:
               current_section = 'population'
            elif 'pib' in row_text:
               current_section = 'gdp'
            elif 'formación' in row_text:
               current_section = 'history'

            # atrapar flags de 2 celdas
            if len(cells) >= 2:
               label = cells[0].get_text(strip=True).lower() # etiqueta de la fila
               value = cells[1].get_text(strip=True).lower()

               # AREA
               if current_section == 'area' and 'total' in label and 'km' in value:
                     area = self.clean_number(value)
               # AGUA
               elif 'agua' in label:
                  water = self.clean_number(value)
                  current_section = None # reset para evitar coger el valor de otra section

               # POBLACION
               if current_section == 'population' and 'censo' in label:
                  population = self.clean_number(value)
               # DENSIDAD
               elif 'densidad' in label:
                  density = self.clean_number(value)
                  current_section = None

               # GPD
               if current_section == 'gpd' and 'total' in label and 'dólares' in value:
                  GPD = self.clean_number(value)
                  current_section = None

               # FECHA
               if current_section == 'history':
                  date = self.clean_date(value)
                  if date:
                     last_event = date
                  current_section = None


         """
            'last_event': last_event,
            'latitude': latitude,
            'longitude': longitude
         """
         # dicc para comprobacion
         raw_data = {
            'country_name': match_title.group(1),
            'area': float(area),
            'water': water,
            'population': population,
            'density': density,
            'GPD': GPD,
            'last_event': last_event,
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
