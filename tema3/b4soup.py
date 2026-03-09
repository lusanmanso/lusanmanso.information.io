from bs4 import BeautifulSoup
import requests

url = "https://books.toscrape.com/"
res = requests.get(url)
html = BeautifulSoup(res.text, 'html.parser')

books = html.select("article.product_pod")

titles = []
prices = []
ratings = []

for book in books:
    # Título (está en el atributo title del <a>)
    title = book.select_one("h3 a")["title"]
    titles.append(title)

    # Precio
    price = book.select_one("p.price_color").text
    prices.append(price)

    # Rating (está en la segunda clase)
    rating = book.select_one("p.star-rating")["class"][1]
    ratings.append(rating)

print(f"Títulos: {titles}")
print(f"Precios: {prices}")
print(f"Ratings: {ratings}")

# Página siguiente
next_page_data = html.select_one("li.next a")

if next_page_data:
    next_page = url + next_page_data["href"]
else:
    next_page = None

print(f"Página siguiente: {next_page}")