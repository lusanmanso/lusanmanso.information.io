Easter Flight Data Challenge

Contexto

Estáis participando en un hackathon de ingeniería de datos.

Vuestra misión es determinar cuál de los siguientes destinos es el más conveniente para viajar desde Madrid durante Semana Santa:

Budapest

Praga

Viena

El análisis debe cubrir un periodo exacto: del 29 de marzo al 5 de abril.

Ventana total de análisis: 8 días consecutivos (Semana Santa).

Objetivo

Desarrollar un sistema automatizado utilizando Selenium que permita:

Extraer datos de vuelos desde un comparador de vuelos.

Estructurar y limpiar el conjunto de datos.

Calcular estadísticas agregadas por destino.

Determinar de forma objetiva cuál es el mejor destino.

Requisitos Técnicos Obligatorios

Es obligatorio el uso de Selenium (WebDriver + esperas explícitas).

Todo el proceso debe estar completamente automatizado.

Se deben scrapear vuelos para cada día dentro del periodo del 29 de marzo al 5 de abril.

Para cada día y destino debe obtenerse un mínimo de 5 vuelos válidos.

Para cada vuelo se debe extraer, como mínimo:

date

destination

price

duration_minutes (convertido correctamente a minutos)

stops

Estructura de Datos (Archivos de Salida)

1. Archivo flights.csv

Los resultados deben almacenarse con exactamente las siguientes columnas:

date

destination

price

duration_minutes

stops

2. Archivo summary.csv

Se debe generar un segundo archivo con exactamente las siguientes columnas:

destination

avg_price

std_price

min_price

avg_duration

direct_ratio

final_score

Fórmula Obligatoria

La puntuación final de cada destino debe calcularse exactamente con la siguiente fórmula:

final_score = (avg_price * 0.5) + (avg_duration * 0.3) + (std_price * 0.2)

El destino con el menor final_score será considerado el mejor destino.

No se permite modificar esta fórmula.

Visualizaciones Obligatorias

Se deben generar automáticamente los siguientes archivos PNG (todos los gráficos deben generarse programáticamente):

price_trend.png

score_comparison.png

Condiciones para que el Proyecto Sea Válido

Un proyecto se considerará válido únicamente si:

Cubre los 8 días completos del periodo (29 de marzo al 5 de abril).

Incluye los 3 destinos.

Contiene al menos 5 vuelos válidos por día y destino.

No contiene valores nulos.

La duración está correctamente convertida a minutos.

El summary.csv coincide con los datos recalculados automáticamente.

La fórmula del final_score está correctamente implementada.

Criterio de Victoria y Entrega

Victoria: Gana el equipo que primero entregue un proyecto completamente válido. El orden de llegada se determinará por el momento de entrega que supere la validación automática.

Desempate: Si ningún equipo logra cumplir todos los requisitos, se evaluará cuál está más completo.

Entrega: Cada equipo debe entregar:

Script en Python.

flights.csv

summary.csv

Dos archivos PNG (price_trend.png y score_comparison.png).

El script podrá ser ejecutado para su verificación.
