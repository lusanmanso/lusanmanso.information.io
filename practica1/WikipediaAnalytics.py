from bs4 import BeautifulSoup
import pandas as pd
import regex as re
import numpy as np

class WikipediaAnalytics:
   def __init__(self, list_of_strings):
      """
      Constructor: guarda la lista de strings.
      list_of_strings puede ser rutas a HTML, URLs, nombres, etc.
      """
      self.sources = list_of_strings
      self.df = None  # Se inicializará en scrap()

   def clean_number(self, value):
    if not value:
        return None

    txt = value.lower()
    if '-' in txt:
        return np.nan

    m = re.search(r'([\d\s\.,\xa0\u202f\u200b\n\t]+)', txt)
    if not m:
        return np.nan

    raw = m.group(1)
    num = re.sub(r'[^\d,\.]', '', raw)  # conservar coma y punto

    # Normalizar separadores decimal/miles
    if ',' in num and '.' in num:
        if num.rfind(',') > num.rfind('.'):
            num = num.replace('.', '').replace(',', '.')
        else:
            num = num.replace(',', '')
    else:
        num = num.replace(',', '.')

    try:
        return float(num)
    except ValueError:
        return np.nan

   # Para el PIB
   def clean_gdp(self, value):
      if not value:
         return None

      value = value.lower()

      match = re.search(r'([\d\s\.,\xa0\u202f\u200b]+)', value)

      if match:
         raw_number = match.group(1)
         base_gdp = re.sub(r'[^\d,]', '', raw_number).replace(',', '.')

         base_gdp = float(base_gdp)
         # aplicar multiplicadores segun escala
         if 'trillones' in value:
            return base_gdp * 1e18
         elif 'billones' in value:
            return base_gdp * 1e12
         elif 'millones' in value or 'mill' in value:
            return base_gdp * 1e6
         else:
            return base_gdp

   # Para las fechas
   def clean_date(self, value):
      if not value:
         return None

      months = { 'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04', 'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08', 'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12' }

      value_clean = re.sub(r'\[.*?\]', ' ', value.lower())

      # (1 o dos numeros) + " de" + (letras del mes) + " de" + (4 numeros del año)
      match = re.findall(r'(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})', value_clean)

      if match:
         latest_date = match[-1]  # coger la última fecha encontrada

         day = latest_date[0].zfill(2)
         month_name = latest_date[1]
         year = latest_date[2]

         if month_name in months:
            month_number = months[month_name]
            return f"{day}/{month_number}/{year}"

      return None

   def clean_coordinates(self, value):
      if not value:
         return None, None

      pattern = r'(\d+)[°º]\s*(\d+)[\'′]\s*(\d+)["″]\s*([NS])\s*(\d+)[°º]\s*(\d+)[\'′]\s*(\d+)["″]\s*([EOW])'
      match = re.search(pattern, value.upper())

      if match:
         lat_deg = float(match.group(1))
         lat_min = float(match.group(2))
         lat_sec = float(match.group(3))
         lat_hem = match.group(4)

         lat_decimal = lat_deg + (lat_min / 60) + (lat_sec / 3600)
         if lat_hem == 'S':
               lat_decimal *= -1

         lon_deg = float(match.group(5))
         lon_min = float(match.group(6))
         lon_sec = float(match.group(7))
         lon_hem = match.group(8)

         lon_decimal = lon_deg + (lon_min / 60) + (lon_sec / 3600)
         if lon_hem in ['O', 'W']:
               lon_decimal *= -1

         return lat_decimal, lon_decimal

      return None, None

   def scrap(self):
      country_list = []

      # 1. Abrir el archivo HTML local y parsearlo con bs4 para convertirlo en un objeto navegable
      for route in self.sources:
         with open(route, 'r', encoding='utf-8') as file:
            content = file.read()

         # Crear objeto beautifulsoup
         soup = BeautifulSoup(content, 'html.parser')

         match_title = re.search(r'"wgTitle":"([^"]+)"', content) # !: nombre del pais ocupa dos columnas no tiene <td>, uso regex

         # Buscar la tabla que en wikipedia es siempre una infobox
         table = soup.find('table', {'class': 'infobox'})

         area, water, population, density, GDP, last_event, latitude, longitude = None, None, None, None, None, None, None, None


         coord_span = soup.find('span', class_='geo-dms')


         if coord_span:
            coord_text = coord_span.get_text(separator=" ", strip=True)
            """"
            print(f"DEBUG: coord_text = '{coord_text}'")  # Ver exactamente qué se extrae
            print(f"DEBUG: repr = {repr(coord_text)}")    # Ver caracteres Unicode
            """
            
            lat_calc, lon_calc = self.clean_coordinates(coord_text)
            if lat_calc is not None and lon_calc is not None:
               latitude = lat_calc
               longitude = lon_calc


         current_section = None  # evitar UnboundLocalError, sirve para marcar por secciones porque el valor no está en la misma fila

         for row in table.find_all('tr'):
            cells = row.find_all(['th', 'td']) # celdas porque header y data no funcionan

            # atrapar flags de solo una columna
            row_text = row.get_text(strip=True).lower()

            if 'superficie' in row_text:
               current_section = 'area'
            elif 'población' in row_text:
               current_section = 'population'
            elif 'pib' in row_text and 'nominal' in row_text:
               current_section = 'gdp'

            # atrapar flags de 2 celdas
            if len(cells) >= 2:
               label = cells[0].get_text(separator=" ", strip=True).lower() # etiqueta de la fila
               value = cells[1].get_text(separator=" ", strip=True).lower()

               # AREA
               if current_section == 'area' and 'total' in label and 'km' in value:
                     area = self.clean_number(value)
               # AGUA
               elif 'agua' in label:
                  water = self.clean_number(value)
                  current_section = None # reset para evitar coger el valor de otra section

               # POBLACION
               if current_section == 'population' and ('censo' in label or 'estimación' in label):
                  population = self.clean_number(value)
               # DENSIDAD
               elif 'densidad' in label:
                  density = self.clean_number(value)
                  current_section = None

               # GDP
               if current_section == 'gdp' and 'total' in label:
                  GDP = self.clean_gdp(value)
                  current_section = None

               # FECHA
               date_found = self.clean_date(value)
               if date_found:
                  last_event = date_found

         # dicc para comprobacion
         raw_data = {
            'Country Name': match_title.group(1),
            'Area (KM^2)': area,
            'Water (%)': water,
            'Population (hab.)': population,
            'Density (hab./km^2)': density,
            'GDP ($)': GDP,
            'Last Event Date': last_event,
            'Latitude (º)': latitude,
            'Longitude(º)': longitude
         }

         country_list.append(raw_data)

         self.df = pd.DataFrame(country_list)

         # Añadimos errors='coerce'
         self.df['Last Event Date'] = pd.to_datetime(
             self.df['Last Event Date'],
             format='%d/%m/%Y',
             errors='coerce'
         )

         pass

   def select_row_by_value(self, col_name, value):
      """
      Devuelve la fila (o filas) cuyo valor en col_name coincide con value.
      """
      if self.df is None or self.df.empty:
         raise ValueError("df empty. Run scrap() first.")

      if col_name not in self.df.columns:
         raise ValueError(f"Column '{col_name}' does not exist in df.")

      filtered_rows = self.df[self.df[col_name] == value]
      list = filtered_rows.to_dict('records')
      result = pd.DataFrame(list)

      if result.empty:
         return None

      return result

   def get_columns(self, col_names):
         """
         Recibe una columna o lista de columnas y devuelve esa parte del DataFrame.
         """
         if self.df is None or self.df.empty:
            raise ValueError("df empty. Run scrap() first.")

         if isinstance(col_names, str):
            col_names = [col_names]

         for col in col_names:
            if col not in self.df.columns:
               raise ValueError("Invalid column")

         """"
         missing_cols = [col for col in col_names if col not in self.df.columns]
         if missing_cols:
            raise KeyError("Invalid column")
         """

         extracted = self.df[col_names]
         list = extracted.to_dict('records')

         result = pd.DataFrame(list)

         return result


   def aggregate_column(self, col_name, operation):
      """
      Agrega una columna numérica según una operación dada:
      - 'max'
      - 'min'
      - 'mean'
      Devuelve el resultado como un float.
      """
      if self.df is None or self.df.empty:
         raise ValueError("df empty. Run scrap() first.")

      if col_name not in self.df.columns:
         raise ValueError("Column not found")

      # Restrigir operaciones permitidas
      valid_ops = ['max', 'min', 'mean']
      if operation not in valid_ops:
         raise ValueError("Invalid operation")

      # Verificar que col es numerica
      if not pd.api.types.is_numeric_dtype(self.df[col_name]):
         raise ValueError(f"Column '{col_name}' must be numeric for aggregation.")

      # Ignorar NaN en la agregación
      value = self.df[col_name].dropna().agg(operation)
      value = np.nan if pd.isna(value) else float(value)

      return value

if __name__ == "__main__":
   # Cambiamos las contrabarras (\) por barras normales (/)
   files = [
       "espania_es.html",
       "grecia_es.html",
       "italia_es.html",
       "polonia_es.html",
       "serbia_es.html"
   ]
   # Nota: Si tu carpeta 'files' está dentro de 'practica1',
   # pon "practica1/files/espania_es.html", etc.

   analytics = WikipediaAnalytics(files)
   analytics.scrap()

   print(analytics.df.to_string())
   print(analytics.df.dtypes)
