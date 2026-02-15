from typing import List, Dict, Optional
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import quote_plus
import json

load_dotenv()


class CurrencyConverter:
    """Pobiera prawdziwe kursy walut z NBP API."""
    
    def __init__(self):
        self.usd_to_pln = self.get_current_rate()
    
    def get_current_rate(self) -> float:
        """Pobiera kurs USD/PLN z NBP."""
        try:
            print("Pobieram kurs USD/PLN z NBP API...")
            response = requests.get(
                "https://api.nbp.pl/api/exchangerates/rates/a/usd/?format=json",
                timeout=10
            )
            if response.status_code == 200:
                rate = response.json()['rates'][0]['mid']
                print(f"Kurs: 1 USD = {rate:.4f} PLN\n")
                return rate
        except Exception as e:
            print(f"Blad NBP API: {e}")
            return 4.00
        return 4.00


class AmazonScraper:
    """Scraper dla Amazon - faktycznie probuje pobrac dane."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def get_best_sellers(self, category: str = "electronics", limit: int = 10) -> List[Dict]:
        """Probuje pobrac bestsellery z Amazon."""
        print(f"Probuje pobrac {limit} produktow z Amazon Best Sellers...")
        
        if category.lower() == "electronics":
            url = "https://www.amazon.com/Best-Sellers-Electronics/zgbs/electronics"
        elif category.lower() == "home":
            url = "https://www.amazon.com/Best-Sellers-Home-Kitchen/zgbs/home-garden"
        else:
            url = f"https://www.amazon.com/Best-Sellers/zgbs/{category}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            print(f"Status kod Amazon: {response.status_code}")
            
            if response.status_code != 200:
                print(f"BLAD: Amazon zwrocil status {response.status_code}")
                print("Mozliwe przyczyny:")
                print("  - Amazon wykryl bota")
                print("  - Wymaga CAPTCHA")
                print("  - IP zablokowane")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: zapisz HTML do pliku
            with open('amazon_debug.html', 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            print("HTML zapisany do amazon_debug.html")
            
            products = []
            
            items = soup.find_all('div', {'class': 'zg-grid-general-faceout'})
            print(f"Znaleziono {len(items)} elementow (selektor 1)")
            
            if not items:
                items = soup.find_all('div', {'id': re.compile(r'gridItemRoot')})
                print(f"Znaleziono {len(items)} elementow (selektor 2)")
            
            if not items:
                items = soup.find_all('div', {'data-asin': True})
                print(f"Znaleziono {len(items)} elementow (selektor 3)")
            
            for item in items[:limit]:
                try:
                    title = None
                    title_selectors = [
                        ('div', {'class': '_cDEzb_p13n-sc-css-line-clamp-3_g3dy1'}),
                        ('div', {'class': 'p13n-sc-truncate'}),
                        ('span', {'class': 'zg-text-center-align'}),
                        ('a', {'class': 'a-link-normal'})
                    ]
                    
                    for tag, attrs in title_selectors:
                        elem = item.find(tag, attrs)
                        if elem:
                            title = elem.get_text(strip=True)
                            break
                    
                    # Cena
                    price = 0.0
                    price_selectors = [
                        ('span', {'class': 'p13n-sc-price'}),
                        ('span', {'class': '_cDEzb_p13n-sc-price_3mJ9Z'}),
                        ('span', {'class': 'a-price-whole'}),
                        ('span', {'class': 'a-offscreen'})
                    ]
                    
                    for tag, attrs in price_selectors:
                        elem = item.find(tag, attrs)
                        if elem:
                            price_text = elem.get_text(strip=True)
                            price = self._extract_price(price_text)
                            if price > 0:
                                break
                    
                    if title and price > 0:
                        products.append({
                            'name': title,
                            'price_usd': price
                        })
                        print(f"  OK: {title[:50]}... ${price}")
                    else:
                        print(f"  BRAK: title={bool(title)}, price={price}")
                
                except Exception as e:
                    print(f"  BLAD parsowania: {e}")
                    continue
            
            print(f"\nPobrano {len(products)} produktow z Amazon")
            
            if len(products) == 0:
                print("\nAmazon blokuje scraping!")
                print("Rozwiazania:")
                print("1. Uzyj Amazon Product Advertising API")
                print("2. Uzyj Selenium z headless browser")
                print("3. Uzyj platnego serwisu scrapingowego")
            
            return products
        
        except Exception as e:
            print(f"BLAD scrapingu Amazon: {e}")
            return []
    
    def _extract_price(self, price_text: str) -> float:
        """Wyciaga cene z tekstu."""
        price_text = price_text.replace('$', '').replace(',', '').replace(' ', '')
        numbers = re.findall(r'\d+\.?\d*', price_text)
        if numbers:
            return float(numbers[0])
        return 0.0


class AllegroSearcher:
    """Scraper dla Allegro - faktycznie probuje pobrac dane."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'pl-PL,pl;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
    
    def search_price(self, product_name: str) -> Optional[float]:
        """Probuje znalezc cene na Allegro."""
        search_query = self._simplify_query(product_name)
        encoded_query = quote_plus(search_query)
        url = f"https://allegro.pl/listing?string={encoded_query}"
        
        print(f"  Szukam: {search_query}")
        print(f"  URL: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            print(f"  Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"  BLAD: Allegro zwrocil status {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            filename = f"allegro_debug_{search_query[:20].replace(' ', '_')}.html"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            print(f"  HTML zapisany do {filename}")
            
            if 'captcha' in response.text.lower() or 'robot' in response.text.lower():
                print("  CAPTCHA wykryte!")
                return None

            price = None

            price_elem = soup.find('span', {'data-box-name': 'price'})
            if price_elem:
                price = self._extract_price_pln(price_elem.get_text())
                print(f"  Znaleziono (selektor 1): {price} PLN")
                return price
            
            price_elems = soup.find_all('span', {'class': re.compile('.*price.*', re.I)})
            print(f"  Znaleziono {len(price_elems)} elementow z 'price'")
            for elem in price_elems[:5]:
                price = self._extract_price_pln(elem.get_text())
                if price > 0:
                    print(f"  Znaleziono (selektor 2): {price} PLN")
                    return price
            
            all_text = soup.get_text()
            prices = re.findall(r'(\d+[,\s]?\d*)\s*zł', all_text)
            if prices:
                price = self._extract_price_pln(prices[0])
                if price > 0:
                    print(f"  Znaleziono (regex): {price} PLN")
                    return price
            
            print("  NIE ZNALEZIONO ceny")
            print("  Allegro prawdopodobnie blokuje lub uzywa JavaScript")
            return None
        
        except Exception as e:
            print(f"  BLAD: {e}")
            return None
    
    def _simplify_query(self, product_name: str) -> str:
        """Upraszcza nazwe produktu."""
        name = product_name.lower()
        name = re.sub(r'\(.*?\)', '', name)
        name = re.sub(r'\d+(st|nd|rd|th)\s+gen(eration)?', '', name)
        
        words = name.split()[:4] 
        return ' '.join(words)
    
    def _extract_price_pln(self, price_text: str) -> float:
        """Wyciaga cene z tekstu."""
        price_text = price_text.replace('zł', '').replace(',', '.').replace(' ', '')
        numbers = re.findall(r'\d+\.?\d*', price_text)
        if numbers:
            return float(numbers[0])
        return 0.0


class PriceAnalyzer:
    """Analizator - TYLKO z prawdziwych danych."""
    
    def __init__(self):
        self.converter = CurrencyConverter()
        self.amazon = AmazonScraper()
        self.allegro = AllegroSearcher()
    
    def analyze_products(self, category: str = "electronics", limit: int = 10) -> List[Dict]:
        """Glowna analiza."""
        print("="*80)
        print("AMAZON-ALLEGRO PRICE ANALYZER - PRAWDZIWY SCRAPING ( xd )")
        print("="*80)
        print()
        
        amazon_products = self.amazon.get_best_sellers(category, limit)
        
        if not amazon_products:
            return []
        
        # Szukaj cen na Allegro
        print("\n" + "="*80)
        print("Szukam cen na Allegro...")
        print("="*80 + "\n")
        
        results = []
        
        for i, product in enumerate(amazon_products, 1):
            print(f"\n[{i}/{len(amazon_products)}] {product['name'][:60]}...")
            
            allegro_price = self.allegro.search_price(product['name'])
            
            if not allegro_price:
                print("  POMINIETY - brak ceny Allegro")
                continue
            
            amazon_pln = product['price_usd'] * self.converter.usd_to_pln
            profit = allegro_price - amazon_pln
            profit_pct = (profit / amazon_pln * 100) if amazon_pln > 0 else 0
            
            results.append({
                'product': product['name'],
                'amazon_usd': product['price_usd'],
                'amazon_pln': amazon_pln,
                'allegro_pln': allegro_price,
                'profit': profit,
                'profit_pct': profit_pct
            })
            
            print(f"  Amazon:  ${product['price_usd']:.2f} = {amazon_pln:.2f} PLN")
            print(f"  Allegro: {allegro_price:.2f} PLN")
            print(f"  Zysk:    {profit:.2f} PLN ({profit_pct:+.1f}%)")
            
            time.sleep(2) 
        
        results.sort(key=lambda x: x['profit'], reverse=True)
        
        print(f"\n\nUDALO SIE PRZEANALIZOWAC: {len(results)}/{len(amazon_products)} produktow")
        
        return results
    
    def display_summary(self, results: List[Dict]):
        """Wyswietla podsumowanie."""
        if not results:
            print("\nBRAK WYNIKOW")
            return
        
        print("\n" + "="*80)
        print("PODSUMOWANIE")
        print("="*80)
        print()
        
        for i, r in enumerate(results[:3], 1):
            print(f"{i}. {r['product'][:60]}")
            print(f"   Zysk: {r['profit']:.2f} PLN ({r['profit_pct']:+.1f}%)")
            print()


def main():
    """Glowna funkcja."""
    
    print("\nUWAGA: Ten program probuje NAPRAWDE scrapowac strony ale mu się nie uda bo są zabezpieczenia :/.")
    
    input("Nacisnij ENTER zeby kontynuowac...")
    
    analyzer = PriceAnalyzer()
    results = analyzer.analyze_products("electronics", 10)
    analyzer.display_summary(results)


if __name__ == "__main__":
    main()