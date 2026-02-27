from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


DESTINATIONS = {
    "Budapest": "bud",
    "Praga": "prg",
    "Viena": "vie",
}

REQUIRED_FLIGHTS_COLUMNS = ["date", "destination", "price", "duration_minutes", "stops"]
REQUIRED_SUMMARY_COLUMNS = [
    "destination",
    "avg_price",
    "std_price",
    "min_price",
    "avg_duration",
    "direct_ratio",
    "final_score",
]


@dataclass
class FlightRecord:
    date: str
    destination: str
    price: float
    duration_minutes: int
    stops: int


def build_date_range(year: int) -> list[date]:
    start = date(year, 3, 29)
    end = date(year, 4, 5)
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def build_skyscanner_url(target_date: date, destination_iata: str) -> str:
    date_token = target_date.strftime("%y%m%d")
    return (
        f"https://www.skyscanner.es/transporte/vuelos/mad/{destination_iata}/{date_token}/"
        "?adultsv2=1&cabinclass=economy&childrenv2=&inboundaltsenabled=false"
        "&outboundaltsenabled=false&preferdirects=false&ref=home&rtn=0"
    )


def parse_price(text: str) -> float | None:
    normalized = text.replace("\xa0", " ")
    euro_pattern = re.compile(r"(?:€\s*)?(\d{1,3}(?:[.\s]\d{3})*(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*€?")
    match = euro_pattern.search(normalized)
    if not match:
        return None

    numeric_text = match.group(1).replace(" ", "").replace(".", "").replace(",", ".")
    try:
        value = float(numeric_text)
    except ValueError:
        return None

    return value if value > 0 else None


def parse_duration_minutes(text: str) -> int | None:
    normalized = text.lower().replace("\xa0", " ")

    hours_minutes = re.search(r"(\d+)\s*h(?:\s*(\d+)\s*m(?:in)?)?", normalized)
    if hours_minutes:
        hours = int(hours_minutes.group(1))
        minutes = int(hours_minutes.group(2)) if hours_minutes.group(2) else 0
        total = hours * 60 + minutes
        return total if total > 0 else None

    minutes_only = re.search(r"(\d+)\s*m(?:in)?", normalized)
    if minutes_only:
        total = int(minutes_only.group(1))
        return total if total > 0 else None

    return None


def parse_stops(text: str) -> int | None:
    normalized = text.lower().replace("\xa0", " ")

    direct_tokens = ["directo", "sin escalas", "direct", "non-stop", "nonstop"]
    if any(token in normalized for token in direct_tokens):
        return 0

    stop_match = re.search(r"(\d+)\s*(?:escala|escalas|stop|stops)", normalized)
    if stop_match:
        return int(stop_match.group(1))

    return None


def create_driver(headless: bool) -> WebDriver:
    options = ChromeOptions()

    if headless:
      options.add_argument("--headless=new")

    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=es-ES")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(options=options)


def maybe_accept_cookies(driver: WebDriver) -> None:
    candidate_selectors = [
        "button#onetrust-accept-btn-handler",
        "button[aria-label*='Aceptar']",
        "button[aria-label*='Accept']",
        "button[data-testid*='cookie']",
    ]

    for selector in candidate_selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            try:
                elements[0].click()
                time.sleep(1)
                return
            except Exception:
                continue


def is_captcha_page(driver: WebDriver) -> bool:
    current_url = driver.current_url.lower()
    page_source = driver.page_source.lower()
    signals = ["captcha", "captcha-v2", "sttc/px", "are you human", "robot"]
    return any(signal in current_url or signal in page_source for signal in signals)


def wait_for_results_page(driver: WebDriver, timeout_seconds: int) -> None:
    wait = WebDriverWait(driver, timeout_seconds)

    def has_result_cards(d: WebDriver) -> bool:
        selectors = [
            "article[data-testid*='flight']",
            "[data-testid*='offer']",
            "div[class*='DayViewCard']",
            "div[class*='Ticket']",
        ]
        return any(d.find_elements(By.CSS_SELECTOR, selector) for selector in selectors)

    try:
        wait.until(lambda d: has_result_cards(d) or "€" in d.page_source)
        if is_captcha_page(driver):
            raise RuntimeError("Skyscanner ha devuelto una página de captcha/bot protection.")
    except TimeoutException as exc:
        if is_captcha_page(driver):
            raise RuntimeError(
                "Skyscanner ha activado captcha/bot protection y no se pueden leer resultados automáticamente."
            ) from exc
        raise RuntimeError("No se cargaron resultados de vuelos dentro del tiempo esperado.") from exc


def get_flight_cards(driver: WebDriver) -> list:
    selectors = [
        "article[data-testid*='flight']",
        "[data-testid*='offer']",
        "div[class*='DayViewCard']",
        "div[class*='Ticket']",
    ]
    cards = []
    for selector in selectors:
        cards.extend(driver.find_elements(By.CSS_SELECTOR, selector))

    unique = []
    seen_ids = set()
    for card in cards:
        try:
            marker = card.id
        except StaleElementReferenceException:
            continue
        if marker not in seen_ids:
            unique.append(card)
            seen_ids.add(marker)
    return unique


def extract_card_price(card_text: str, card_element) -> float | None:
    price_selectors = [
        "[data-testid*='price']",
        "span[class*='Price']",
        "div[class*='Price']",
    ]

    for selector in price_selectors:
        try:
            elements = card_element.find_elements(By.CSS_SELECTOR, selector)
        except StaleElementReferenceException:
            continue
        for element in elements:
            parsed = parse_price(element.text)
            if parsed is not None:
                return parsed

    return parse_price(card_text)


def extract_card_duration(card_text: str, card_element) -> int | None:
    duration_selectors = [
        "[data-testid*='duration']",
        "span[class*='Duration']",
        "div[class*='Duration']",
    ]

    for selector in duration_selectors:
        try:
            elements = card_element.find_elements(By.CSS_SELECTOR, selector)
        except StaleElementReferenceException:
            continue
        for element in elements:
            parsed = parse_duration_minutes(element.text)
            if parsed is not None:
                return parsed

    return parse_duration_minutes(card_text)


def extract_card_stops(card_text: str, card_element) -> int | None:
    stops_selectors = [
        "[data-testid*='stops']",
        "span[class*='Stops']",
        "div[class*='Stops']",
    ]

    for selector in stops_selectors:
        try:
            elements = card_element.find_elements(By.CSS_SELECTOR, selector)
        except StaleElementReferenceException:
            continue
        for element in elements:
            parsed = parse_stops(element.text)
            if parsed is not None:
                return parsed

    return parse_stops(card_text)


def try_load_more_results(driver: WebDriver) -> None:
    load_more_selectors = [
        "button[data-testid*='load']",
        "button[aria-label*='Mostrar']",
        "button[aria-label*='Show more']",
    ]

    for selector in load_more_selectors:
        buttons = driver.find_elements(By.CSS_SELECTOR, selector)
        if buttons:
            try:
                buttons[0].click()
                time.sleep(2)
                return
            except Exception:
                continue


def scrape_route_for_date(
    driver: WebDriver,
    target_date: date,
    destination_name: str,
    destination_iata: str,
    min_flights: int,
    timeout_seconds: int,
    manual_captcha_wait_seconds: int,
) -> list[FlightRecord]:
    url = build_skyscanner_url(target_date, destination_iata)
    driver.get(url)

    if is_captcha_page(driver):
        if manual_captcha_wait_seconds > 0:
            print(
                "[AVISO] Captcha detectado. Resuélvelo manualmente en la ventana del navegador "
                f"en los próximos {manual_captcha_wait_seconds} segundos..."
            )
            time.sleep(manual_captcha_wait_seconds)
        else:
            raise RuntimeError(
                "Captcha detectado al abrir Skyscanner. Ejecuta en modo visible (sin --headless) "
                "y usa --manual-captcha-wait para permitir resolución manual."
            )

    maybe_accept_cookies(driver)
    wait_for_results_page(driver, timeout_seconds=timeout_seconds)

    records: list[FlightRecord] = []
    seen_signatures: set[tuple[float, int, int]] = set()

    for _ in range(10):
        cards = get_flight_cards(driver)
        for card in cards:
            try:
                card_text = card.text
                price = extract_card_price(card_text, card)
                duration_minutes = extract_card_duration(card_text, card)
                stops = extract_card_stops(card_text, card)
            except StaleElementReferenceException:
                # La card cambió mientras se parseaba; se ignora y se vuelve a intentar en el siguiente barrido.
                continue

            if price is None or duration_minutes is None or stops is None:
                continue

            signature = (price, duration_minutes, stops)
            if signature in seen_signatures:
                continue

            records.append(
                FlightRecord(
                    date=target_date.isoformat(),
                    destination=destination_name,
                    price=price,
                    duration_minutes=duration_minutes,
                    stops=stops,
                )
            )
            seen_signatures.add(signature)

            if len(records) >= min_flights:
                return records

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        try_load_more_results(driver)

    return records


def scrape_all_flights(
    year: int,
    headless: bool,
    min_flights: int,
    timeout_seconds: int,
    manual_captcha_wait_seconds: int,
) -> pd.DataFrame:
    date_range = build_date_range(year)
    all_records: list[FlightRecord] = []

    driver = create_driver(headless=headless)
    try:
        for target_date in date_range:
            for destination_name, destination_iata in DESTINATIONS.items():
                print(f"[SCRAPE] {target_date.isoformat()} -> {destination_name}")
                records = scrape_route_for_date(
                    driver=driver,
                    target_date=target_date,
                    destination_name=destination_name,
                    destination_iata=destination_iata,
                    min_flights=min_flights,
                    timeout_seconds=timeout_seconds,
                    manual_captcha_wait_seconds=manual_captcha_wait_seconds,
                )

                if len(records) < min_flights:
                    raise RuntimeError(
                        "No se alcanzó el mínimo de vuelos válidos "
                        f"({len(records)}/{min_flights}) para {destination_name} en {target_date.isoformat()}."
                    )

                all_records.extend(records)
    finally:
        driver.quit()

    return pd.DataFrame([record.__dict__ for record in all_records], columns=REQUIRED_FLIGHTS_COLUMNS)


def validate_flights(df: pd.DataFrame, year: int, min_flights: int) -> pd.DataFrame:
    if list(df.columns) != REQUIRED_FLIGHTS_COLUMNS:
        raise ValueError("El archivo de vuelos no tiene exactamente las columnas requeridas.")

    if df.isnull().any().any():
        raise ValueError("Se detectaron valores nulos en flights.csv.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.date
    df["destination"] = df["destination"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="raise")
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="raise").astype(int)
    df["stops"] = pd.to_numeric(df["stops"], errors="raise").astype(int)

    if (df["price"] <= 0).any():
        raise ValueError("Hay precios no válidos (<= 0).")
    if (df["duration_minutes"] <= 0).any():
        raise ValueError("Hay duraciones no válidas (<= 0).")
    if (df["stops"] < 0).any():
        raise ValueError("Hay valores de escalas no válidos (< 0).")

    expected_dates = set(build_date_range(year))
    observed_dates = set(df["date"].unique())
    if observed_dates != expected_dates:
        missing = sorted(expected_dates - observed_dates)
        extra = sorted(observed_dates - expected_dates)
        raise ValueError(f"Cobertura de fechas incorrecta. Faltan={missing}, Extra={extra}")

    expected_destinations = set(DESTINATIONS.keys())
    observed_destinations = set(df["destination"].unique())
    if observed_destinations != expected_destinations:
        missing = sorted(expected_destinations - observed_destinations)
        extra = sorted(observed_destinations - expected_destinations)
        raise ValueError(f"Cobertura de destinos incorrecta. Faltan={missing}, Extra={extra}")

    per_group_counts = df.groupby(["date", "destination"]).size()
    too_few = per_group_counts[per_group_counts < min_flights]
    if not too_few.empty:
        formatted = ", ".join([f"{idx[0]}-{idx[1]}:{count}" for idx, count in too_few.items()])
        raise ValueError(f"Hay combinaciones fecha-destino con menos de {min_flights} vuelos: {formatted}")

    df["date"] = df["date"].astype(str)
    return df.sort_values(["date", "destination", "price", "duration_minutes"]).reset_index(drop=True)


def build_summary(df_flights: pd.DataFrame) -> pd.DataFrame:
    grouped = df_flights.groupby("destination", as_index=False).agg(
        avg_price=("price", "mean"),
        std_price=("price", "std"),
        min_price=("price", "min"),
        avg_duration=("duration_minutes", "mean"),
        direct_ratio=("stops", lambda x: (x == 0).mean()),
    )

    grouped["std_price"] = grouped["std_price"].fillna(0.0)
    grouped["final_score"] = (
        grouped["avg_price"] * 0.5
        + grouped["avg_duration"] * 0.3
        + grouped["std_price"] * 0.2
    )

    grouped = grouped[REQUIRED_SUMMARY_COLUMNS]
    return grouped.sort_values("final_score", ascending=True).reset_index(drop=True)


def validate_summary(df_flights: pd.DataFrame, df_summary: pd.DataFrame) -> None:
    if list(df_summary.columns) != REQUIRED_SUMMARY_COLUMNS:
        raise ValueError("summary.csv no tiene exactamente las columnas requeridas.")

    recalculated = build_summary(df_flights)
    merged = df_summary.merge(
        recalculated,
        on="destination",
        suffixes=("_reported", "_recalc"),
        how="inner",
    )

    if len(merged) != len(DESTINATIONS):
        raise ValueError("summary.csv no contiene exactamente los 3 destinos requeridos.")

    numeric_fields = ["avg_price", "std_price", "min_price", "avg_duration", "direct_ratio", "final_score"]
    for field in numeric_fields:
        left = merged[f"{field}_reported"].astype(float)
        right = merged[f"{field}_recalc"].astype(float)
        if not left.combine(right, lambda a, b: math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)).all():
            raise ValueError(f"summary.csv no coincide con el recálculo automático en el campo {field}.")


def generate_price_trend(df_flights: pd.DataFrame, output_file: Path) -> None:
    trend = (
        df_flights.groupby(["date", "destination"], as_index=False)["price"].mean().sort_values("date")
    )
    trend["date"] = pd.to_datetime(trend["date"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for destination, subset in trend.groupby("destination"):
        ax.plot(subset["date"], subset["price"], marker="o", label=destination)

    ax.set_title("Evolución del precio medio por destino")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio medio (€)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def generate_score_comparison(df_summary: pd.DataFrame, output_file: Path) -> None:
    ordered = df_summary.sort_values("final_score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ordered["destination"], ordered["final_score"])
    ax.set_title("Comparativa de final_score por destino")
    ax.set_xlabel("Destino")
    ax.set_ylabel("Final score (menor es mejor)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='.', type=str, help='Carpeta de salida')
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--headless", action="store_true", help="Ejecutar sin UI")
    parser.add_argument("--manual-captcha-wait", type=int, default=120)
    parser.add_argument("--min-flights", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=45)

    return parser.parse_args()


def main() -> None:

   parser = argparse.ArgumentParser()

   args = parser.parse_args()
   output_dir: Path = Path(args.output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Iniciando scraping de vuelos...")
    df_flights = scrape_all_flights(
        year=args.year,
        headless=args.headless,
        min_flights=args.min_flights,
        timeout_seconds=args.timeout,
        manual_captcha_wait_seconds=args.manual_captcha_wait,
    )

    df_flights = validate_flights(df_flights, year=args.year, min_flights=args.min_flights)

    flights_path = output_dir / "flights.csv"
    df_flights.to_csv(flights_path, index=False)
    print(f"[OK] flights.csv generado en: {flights_path}")

    df_summary = build_summary(df_flights)
    validate_summary(df_flights, df_summary)

    summary_path = output_dir / "summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[OK] summary.csv generado en: {summary_path}")

    generate_price_trend(df_flights, output_dir / "price_trend.png")
    generate_score_comparison(df_summary, output_dir / "score_comparison.png")
    print(f"[OK] price_trend.png generado en: {output_dir / 'price_trend.png'}")
    print(f"[OK] score_comparison.png generado en: {output_dir / 'score_comparison.png'}")

    best_destination = df_summary.sort_values("final_score", ascending=True).iloc[0]["destination"]
    best_score = df_summary.sort_values("final_score", ascending=True).iloc[0]["final_score"]
    print(f"[RESULTADO] Mejor destino: {best_destination} (final_score={best_score:.2f})")


if __name__ == "__main__":
    main()
