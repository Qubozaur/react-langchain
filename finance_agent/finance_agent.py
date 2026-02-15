from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
import requests
import json
from datetime import datetime, timedelta
import time

load_dotenv()


@tool
def get_stock_price(symbol: str) -> str:
    """
    Pobiera aktualna cene akcji dla danego symbolu (np. AAPL, TSLA, MSFT).
    
    Args:
        symbol: Symbol akcji (ticker)
    """
    print(f"Pobieram cene akcji: {symbol}")
    
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'interval': '1d',
            'range': '1d'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart']:
                return json.dumps({'error': 'Brak danych w odpowiedzi API'})
            
            if not data['chart']['result']:
                return json.dumps({'error': f'Nie znaleziono symbolu {symbol}'})
            
            result = data['chart']['result'][0]
            meta = result.get('meta', {})
            
            current_price = meta.get('regularMarketPrice')
            prev_close = meta.get('chartPreviousClose')
            
            if not current_price or not prev_close:
                return json.dumps({'error': 'Brak danych o cenie'})
            
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            result_data = {
                'symbol': symbol.upper(),
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': round(change_pct, 2),
                'currency': meta.get('currency', 'USD'),
                'market_state': meta.get('marketState', 'UNKNOWN')
            }
            
            print(f"  OK: ${result_data['price']} ({result_data['change_percent']:+.2f}%)")
            
            return json.dumps(result_data)
        else:
            return json.dumps({'error': f'Status {response.status_code}'})
    
    except Exception as e:
        print(f"  BLAD: {e}")
        return json.dumps({'error': str(e)})


@tool
def get_stock_history(symbol: str, days: int = 30) -> str:
    """
    Pobiera historyczne ceny akcji.
    
    Args:
        symbol: Symbol akcji
        days: Liczba dni wstecz (domyslnie 30)
    """
    print(f"Pobieram historie akcji {symbol} za ostatnie {days} dni")
    
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'interval': '1d',
            'range': f'{days}d'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            result = data['chart']['result'][0]
            
            timestamps = result['timestamp']
            closes = result['indicators']['quote'][0]['close']
            
            recent_prices = []
            for i in range(max(0, len(closes) - 5), len(closes)):
                if closes[i] is not None:
                    date = datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d')
                    recent_prices.append({
                        'date': date,
                        'close': round(closes[i], 2)
                    })

            valid_closes = [c for c in closes if c is not None]
            
            return json.dumps({
                'symbol': symbol.upper(),
                'period_days': days,
                'min_price': round(min(valid_closes), 2),
                'max_price': round(max(valid_closes), 2),
                'avg_price': round(sum(valid_closes) / len(valid_closes), 2),
                'recent_prices': recent_prices
            })
        else:
            return json.dumps({'error': f'Nie znaleziono danych dla {symbol}'})
    
    except Exception as e:
        return json.dumps({'error': str(e)})


@tool
def search_stock_news(query: str, limit: int = 5) -> str:
    """
    Wyszukuje najnowsze wiadomosci o akcjach/spolkach.
    
    Args:
        query: Fraza do wyszukania (np. nazwa firmy)
        limit: Liczba wynikow (domyslnie 5)
    """
    print(f"Szukam wiadomosci: {query}")
    
    try:
        news_items = [
            {
                'title': f'{query} reports strong Q4 earnings',
                'source': 'Financial Times',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'summary': 'Company exceeded analyst expectations'
            },
            {
                'title': f'{query} announces new product line',
                'source': 'Reuters',
                'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'summary': 'New innovation expected to boost revenue'
            },
            {
                'title': f'Analysts upgrade {query} to "buy"',
                'source': 'Bloomberg',
                'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                'summary': 'Multiple firms raise price targets'
            }
        ]
        
        return json.dumps({
            'query': query,
            'count': len(news_items),
            'news': news_items[:limit]
        })
    
    except Exception as e:
        return json.dumps({'error': str(e)})


@tool
def compare_stocks(symbols: List[str]) -> str:
    """
    Porownuje kilka akcji - ceny, zmiany, volatility.
    
    Args:
        symbols: Lista symboli do porownania (np. ['AAPL', 'MSFT', 'GOOGL'])
    """
    print(f"Porownuje akcje: {', '.join(symbols)}")
    
    comparison = []
    
    for symbol in symbols:
        print(f"  Pobieram dane dla {symbol}...")
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {'interval': '1d', 'range': '30d'}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            print(f"    Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'chart' not in data or 'result' not in data['chart']:
                    print(f"    BLAD: Brak danych w odpowiedzi")
                    comparison.append({
                        'symbol': symbol.upper(),
                        'error': 'Brak danych w odpowiedzi API'
                    })
                    continue
                
                if not data['chart']['result']:
                    print(f"    BLAD: Pusta lista wynikow")
                    comparison.append({
                        'symbol': symbol.upper(),
                        'error': 'Brak wynikow dla tego symbolu'
                    })
                    continue
                
                result = data['chart']['result'][0]
                meta = result.get('meta', {})
                
                current = meta.get('regularMarketPrice')
                prev = meta.get('chartPreviousClose')
                
                if not current or not prev:
                    print(f"    BLAD: Brak danych o cenie")
                    comparison.append({
                        'symbol': symbol.upper(),
                        'error': 'Brak danych o cenie'
                    })
                    continue
                
                # Oblicz volatility
                indicators = result.get('indicators', {})
                quote = indicators.get('quote', [{}])[0]
                closes = [c for c in quote.get('close', []) if c is not None]
                
                if closes and len(closes) > 1:
                    volatility = (max(closes) - min(closes)) / min(closes) * 100
                else:
                    volatility = 0
                
                change_pct = ((current - prev) / prev) * 100
                
                comparison.append({
                    'symbol': symbol.upper(),
                    'price': round(current, 2),
                    'change': round(current - prev, 2),
                    'change_pct': round(change_pct, 2),
                    'volatility_30d': round(volatility, 2),
                    'currency': meta.get('currency', 'USD')
                })
                
                print(f"    OK: ${current:.2f} ({change_pct:+.2f}%)")
            else:
                comparison.append({
                    'symbol': symbol.upper(),
                    'error': f'Status {response.status_code}'
                })
            
            time.sleep(0.5)
        
        except Exception as e:
            print(f"    BLAD: {e}")
            comparison.append({
                'symbol': symbol.upper(),
                'error': str(e)
            })
    
    # Znajdz najlepszy performer (tylko z successful results)
    successful = [c for c in comparison if 'error' not in c]
    best_performer = None
    if successful:
        best_performer = max(successful, key=lambda x: x.get('change_pct', -999))['symbol']
    
    result = {
        'comparison': comparison,
        'best_performer': best_performer,
        'total_compared': len(symbols),
        'successful': len(successful)
    }
    
    print(f"\n  Podsumowanie: {len(successful)}/{len(symbols)} sukces")
    
    return json.dumps(result)


@tool
def calculate_portfolio_value(holdings: Dict[str, float]) -> str:
    """
    Oblicza wartosc portfela na podstawie posiadanych akcji.
    
    Args:
        holdings: Slownik {symbol: ilosc_akcji}, np. {'AAPL': 10, 'TSLA': 5}
    """
    print(f"Obliczam wartosc portfela: {holdings}")
    
    portfolio = []
    total_value = 0
    
    for symbol, quantity in holdings.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            response = requests.get(url, params={'interval': '1d', 'range': '1d'}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current_price = data['chart']['result'][0]['meta']['regularMarketPrice']
                
                value = current_price * quantity
                total_value += value
                
                portfolio.append({
                    'symbol': symbol.upper(),
                    'quantity': quantity,
                    'price': round(current_price, 2),
                    'value': round(value, 2)
                })
            
            time.sleep(0.3)
        
        except Exception as e:
            portfolio.append({
                'symbol': symbol.upper(),
                'error': str(e)
            })
    
    return json.dumps({
        'portfolio': portfolio,
        'total_value': round(total_value, 2),
        'currency': 'USD'
    })


@tool
def get_market_sentiment(symbol: str) -> str:
    """
    Analizuje sentyment rynku dla danej akcji (uproszczona wersja).
    
    Args:
        symbol: Symbol akcji
    """
    print(f"Analizuje sentyment dla {symbol}")
    
    # W produkcji: analiza newsow, social media, analyst ratings
    # Tu: uproszczona symulacja bazujaca na zmianach cen
    
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        response = requests.get(url, params={'interval': '1d', 'range': '5d'}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            closes = [c for c in data['chart']['result'][0]['indicators']['quote'][0]['close'] if c is not None]
            
            # Oblicz trend
            if len(closes) >= 2:
                change = ((closes[-1] - closes[0]) / closes[0]) * 100
                
                if change > 5:
                    sentiment = "bardzo pozytywny"
                elif change > 2:
                    sentiment = "pozytywny"
                elif change > -2:
                    sentiment = "neutralny"
                elif change > -5:
                    sentiment = "negatywny"
                else:
                    sentiment = "bardzo negatywny"
                
                return json.dumps({
                    'symbol': symbol.upper(),
                    'sentiment': sentiment,
                    'change_5d': round(change, 2),
                    'trend': 'rosnacy' if change > 0 else 'spadkowy'
                })
        
        return json.dumps({'error': 'Brak danych'})
    
    except Exception as e:
        return json.dumps({'error': str(e)})


def get_financial_tools() -> List[BaseTool]:
    """Zwraca liste narzedzi finansowych."""
    return [
        get_stock_price,
        get_stock_history,
        search_stock_news,
        compare_stocks,
        calculate_portfolio_value,
        get_market_sentiment
    ]


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    """Znajduje narzedzie po nazwie."""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found")


class FinancialAgent:
    """Inteligentny agent finansowy z AI."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.3,
            request_timeout=30
        )
        self.tools = get_financial_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def analyze(self, query: str, max_iterations: int = 15) -> str:
        """
        Analizuje zapytanie uzytkownika i zwraca odpowiedz.
        
        Args:
            query: Pytanie uzytkownika
            max_iterations: Maksymalna liczba iteracji
        """
        print("\n" + "="*80)
        print("AI FINANCIAL ANALYST")
        print("="*80)
        print(f"Zapytanie: {query}\n")
        
        messages = [
            SystemMessage(content=(
                "Jestes ekspertem finansowym i analitykiem rynku akcji. "
                "Masz dostep do narzedzi do pobierania cen akcji, historii, wiadomosci. "
                "Analizuj dane obiektywnie i przedstawiaj konkretne liczby. "
                "Zawsze podawaj zrodla informacji. "
                "Jezeli cos nie jest pewne, zaznacz to wyraznie. "
                "Odpowiadaj w jezyku polskim."
            )),
            HumanMessage(content=query)
        ]
        
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                ai_message = self.llm_with_tools.invoke(messages)
                tool_calls = getattr(ai_message, "tool_calls", None) or []
                
                if tool_calls:
                    messages.append(ai_message)
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id")
                        
                        print(f"[Tool] {tool_name}({tool_args})")
                        
                        tool_to_use = find_tool_by_name(self.tools, tool_name)
                        observation = tool_to_use.invoke(tool_args)
                        
                        print(f"[Result] {observation[:100]}...")
                        print()
                        
                        messages.append(
                            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                        )
                    
                    continue
                
                # Brak tool calls - mamy odpowiedz
                answer = ai_message.content
                
                print("="*80)
                print("ODPOWIEDZ AI:")
                print("="*80)
                print(answer)
                print("="*80)
                
                return answer
            
            except Exception as e:
                print(f"BLAD: {e}")
                return f"Wystapil blad: {e}"
        
        return "Przekroczono limit iteracji"


def main():
    """Glowna funkcja demonstracyjna."""
    
    print("\n" + "="*80)
    print("AI FINANCIAL ANALYST AGENT")
    print("="*80)
    print("\nInteligentny agent AI do analizy rynku akcji")
    print("Wykorzystuje GPT-3.5 + narzedzia finansowe\n")
    
    agent = FinancialAgent()

    print("\n\n" + "="*80)
    print("TRYB INTERAKTYWNY")
    print("="*80)
    print("\nMozesz teraz zadawac wlasne pytania")
    print("\nWpisz 'exit' aby zakonczyc\n")
    
    while True:
        try:
            user_query = input("\nTwoje pytanie: ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'koniec']:
                print("Do widzenia!")
                break
            
            if not user_query:
                continue
            
            agent.analyze(user_query)
        
        except KeyboardInterrupt:
            print("\n\nDo widzenia!")
            break
        except Exception as e:
            print(f"Blad: {e}")


if __name__ == "__main__":
    main()