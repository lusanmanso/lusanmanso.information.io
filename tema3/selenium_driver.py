from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd

# Configuramos el WebDriver
def create_driver():
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument('--headless') #No arrancamos el browser
    chrome_options.add_argument('--no-sandbox')
    chrome_options.headless = True
    wd = webdriver.Chrome(options=chrome_options)
    #driver = webdriver.Chrome(options=chrome_options)
    return wd

# Función para obtener la información de un Pokémon
def get_pokemon_info(url):
    #driver = create_driver()
    driver.get(url)

    try:
        pokemon_name_qs = ".summary .product_title"
        pokemon_price_qs = "p.price"
        pokemon_desc_qs = ".summary .woocommerce-product-details__short-description p"
        pokemon_stock_qs = ".summary .stock"
        pokemon_image_qs = ".woocommerce-product-gallery__image .wp-post-image"
        pokemon_price_curr_qs = ".woocommerce-Price-currencySymbol"

        pokemon_name = driver.find_element(By.CSS_SELECTOR, pokemon_name_qs).text
        print(f"Scraping {pokemon_name}")

        pokemon_price_curr = driver.find_element(By.CSS_SELECTOR, pokemon_price_curr_qs).text
        pokemon_price = driver.find_element(By.CSS_SELECTOR, pokemon_price_qs).text
        pokemon_price = float(pokemon_price.replace(pokemon_price_curr, "").strip())

        pokemon_desc = driver.find_element(By.CSS_SELECTOR, pokemon_desc_qs).text
        pokemon_stock = int(driver.find_element(By.CSS_SELECTOR, pokemon_stock_qs).text.split(" ")[0])
        pokemon_image = driver.find_element(By.CSS_SELECTOR, pokemon_image_qs).get_attribute("src")

    except Exception as e:
        driver.quit()
        return f"Error with page: {url}, {e}"

    #driver.quit()
    return {"name": pokemon_name,
            "image_url": pokemon_image,
            "description": pokemon_desc,
            "price": pokemon_price,
            "currency": pokemon_price_curr,
            "stock": pokemon_stock}

# Función para obtener los elementos de la página
def get_page_elements(url):
    #driver = create_driver()
    correct= False
    #while()
    driver.get(url)
    print(driver.title)

    try:
        print(f"Scraping: {url}")

        urls_qs = "ul .product .woocommerce-LoopProduct-link"
        next_url_qs = ".page-numbers .next"

        urls = [elem.get_attribute("href") for elem in driver.find_elements(By.CSS_SELECTOR, urls_qs)]

        next_url_elem = driver.find_elements(By.CSS_SELECTOR, next_url_qs)
        next_url = next_url_elem[0].get_attribute("href") if next_url_elem else None

    except Exception as e:
        driver.quit()
        return f"Error with: {url}, {e}"

    #driver.quit()
    return urls, next_url

# Scrapeamos todas las páginas
next_page = "https://scrapeme.live/shop/"
pokemons_list = []
driver = create_driver()

while next_page is not None:
    try:
      urls, next_page = get_page_elements(next_page)
      for url in urls:
          pokemon_info = get_pokemon_info(url)
          pokemons_list.append(pokemon_info)
      break # Solo leemos la primera página
    except Exception as e:
      #print("Error")
      print(f"Error scraping page: {e}")
      print("----------")

df = pd.DataFrame(pokemons_list)
print(df)