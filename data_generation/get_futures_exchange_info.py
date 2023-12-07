import requests
import pandas as pd

def get_futures_trading_rules():
    url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'  # API endpoint for futures trading rules
    response = requests.get(url)
    data = response.json()

    symbols_data = []
    for symbol_data in data['symbols']:
        symbol = symbol_data['symbol']
        min_qty = None
        min_notional = None
        tick_size = None
        for rule in symbol_data['filters']:
            if rule['filterType'] == 'LOT_SIZE':
                min_qty = float(rule['minQty'])
            elif rule['filterType'] == 'MIN_NOTIONAL':
                min_notional = float(rule['notional'])
            elif rule['filterType'] == 'PRICE_FILTER':
                tick_size = float(rule['tickSize'])

        symbols_data.append({
            'symbol': symbol,
            'min_qty': min_qty,
            'min_notional': min_notional,
            'tick_size': tick_size
        })

    return symbols_data

# Retrieve futures trading rules
trading_rules = get_futures_trading_rules()

# Create DataFrame
df = pd.DataFrame(trading_rules)
df.to_csv('./futures_trading_rules/futures_trading_rules.csv', index=False)