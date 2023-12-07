import os
from datetime import datetime
from collections import deque
from binance.client import Client
import argparse

import alpha_collection
from utils import *
from decimal import Decimal
import time
import csv
from const import *


def get_current_futures_position(symbols):
    account_info = client.futures_account()
    positions = account_info['positions']
    df_futures_position = pd.DataFrame([position for position in positions if position['symbol'] in symbols]).set_index("symbol")[['entryPrice', 'positionAmt', 'leverage']].astype(float)
    max_withdraw_amount = float(account_info['maxWithdrawAmount'])
    return df_futures_position, max_withdraw_amount

def create_order(symbol, price, quantity, leverage, side=None, is_dryrun=False):
    if side is None:
        side = "BUY" if quantity > 0 else "SELL"
    quantity = str(abs(Decimal(str(quantity))))
    if not is_dryrun and quantity != '0':
        try:
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                price=price,
                quantity=quantity,
                timeInForce='GTC',
                type="LIMIT",
                reduceOnly=False
            )
        except Exception as e:
            print(f'Failed to create order for {symbol} {side} {quantity} at price {price}', "Exception:", type(e).__name__)
            time.sleep(0.0001)
            return False, None
    data = [symbol, side, quantity, price, leverage]
    return True, data

def log_order(order_data_list, is_dryrun=True):
    #current timestamp in date time format
    print(datetime.now())
    msg_list = [f'{data[1]} {data[2]} {data[0]} at price {data[3]}, usd {round(float(data[2]) * float(data[3]), 2)}, leverage={data[4]}' for data in order_data_list]
    if is_dryrun:
        for msg in msg_list:
            print(msg)
    else:
        csv_file = './logs/buy_log.csv'
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            if not file_exists:
                writer.writerow(['time', 'symbol', 'side', 'quantity', 'price', 'leverage'])

            for data in order_data_list:
                writer.writerow([str(datetime.now())] + [str(e) for e in data])
                msg = f'{data[1]} {data[2]} {data[0]} at price {data[3]}, usd {round(float(data[2]) * float(data[3]), 2)}, leverage={data[4]}'
                print(msg)

def slack_order(order_data_list, is_dryrun=False, slack_token=None):
    if slack_token:
        columns = ['symbol', 'side', 'quantity', 'price', 'leverage']
        rows = [columns] + [[str(element) for element in data] for data in order_data_list]
        width = max([max(len(str(cell)) for cell in column) for column in zip(*rows)])
        message = ""
        for row in rows:
            message += " | ".join(cell.ljust(width) for cell in row)
            message += "\n"
        channel = SLACK_DRYRUN_ORDER_LOG_CHANNEL if is_dryrun else SLACK_ORDER_LOG_CHANNEL
        send_slack_message(message, slack_token, channel)


def log_total_quantity(quantity):
    csv_file = './logs/total_quantity.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        if not file_exists:
            writer.writerow(['time', 'quantity'])
        writer.writerow([str(datetime.now()), str(quantity)])

def slack_total_quantity(quantity, is_dryrun=False, slack_token=None):
    if slack_token:
        msg = f'Total quantity: {quantity}'
        channel = SLACK_DRYRUN_TOTAL_QUANTITY_CHANNEL if is_dryrun else SLACK_TOTAL_QUANTITY_CHANNEL
        send_slack_message(msg, slack_token, channel)

def cancle_order_and_close_all_positions(symbols, is_dryrun=False):
    if is_dryrun:
        return
    cancel_all_orders(symbols)
    for symbol in symbols:
        df_current_futures_position, _ = get_current_futures_position(symbols)
        side = Client.SIDE_BUY if df_current_futures_position.loc[symbol]['positionAmt'] < 0 else Client.SIDE_SELL # close position
        quantity = str(abs(df_current_futures_position.loc[symbol]['positionAmt']))
        try:
            order_response = client.futures_create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            position = 'LONG' if side == Client.SIDE_BUY else 'SHORT'
            msg = f"Force-closed position for {position} {symbol} {quantity}"
            send_slack_message(msg, slack_token, SLACK_SOMETHING_IRREGULAR_CHANNEL)
            print(msg)
        except Exception as e:
            print(f"Error placing order: {e}")

def get_remaining_orders(symbols):
    open_orders = {}
    for symbol in symbols:
        open_orders_symbol = client.futures_get_open_orders(symbol=symbol)
        if len(open_orders_symbol) > 0:
            open_orders[symbol] = open_orders_symbol[0] #length is always 1
    return open_orders

def renew_order_if_not_meet(symbols, leverage):
    open_orders = get_remaining_orders(symbols)
    if len(open_orders) == 0:
        return True
    current_price = {symbol: float(client.futures_ticker(symbol=symbol)['lastPrice']) for symbol in open_orders.keys()}
    for symbol, order in open_orders.items():
        canceled = cancel_order(order, symbol)
        if canceled:
            usdt_amount = float(order['origQty']) * float(order['price'])
            adjusted_quantity_trimmed = trim_quantity(symbol, usdt_amount, current_price[symbol])
            create_order(symbol=symbol, price=current_price[symbol], quantity=adjusted_quantity_trimmed, leverage=leverage, is_dryrun=is_dryrun, side=order['side'])
            msg = f"Renewed order into {order['side']} {symbol} {adjusted_quantity_trimmed} at price {current_price[symbol]}"
            print(msg)
            send_slack_message(msg, slack_token, SLACK_SOMETHING_IRREGULAR_CHANNEL)
    return False

def order_with_quantity(df, quantity_column_name, price_column_name, is_dryrun=False, leverage=1):
    unfilled_symbols = deque(df.index)
    order_data_list = []
    tried, try_max_cnt = 0, 100
    while unfilled_symbols:
        symbol = unfilled_symbols.popleft()
        is_success, data = create_order(symbol=symbol, price=df.loc[symbol, price_column_name], quantity=df.loc[symbol, quantity_column_name], leverage=leverage, is_dryrun=is_dryrun)
        if not is_success:
            unfilled_symbols.append(symbol)
        else:
            tried += 1
            order_data_list.append(data)
        if tried >= try_max_cnt:
            cancle_order_and_close_all_positions(symbols, is_dryrun=is_dryrun)
            break
    return order_data_list


def cancel_order(order, symbol):
    order_id = order['orderId']
    cancel_response = client.futures_cancel_order(symbol=symbol, orderId=order_id)
    if cancel_response['status'] == 'CANCELED':
        msg = f"Order {order_id} ({order['side']} {order['origQty']} {symbol} at price {order['price']}) canceled successfully."
        print(msg)
        send_slack_message(msg, slack_token, SLACK_SOMETHING_IRREGULAR_CHANNEL)
        return True
    else:
        msg = f"Failed to cancel order {order_id}. Error: {cancel_response['msg']}"
        print(msg)
        send_slack_message(msg, slack_token, SLACK_SOMETHING_IRREGULAR_CHANNEL)
        return False

def cancel_all_orders(symbols):
    cancel_any = False
    for symbol in symbols:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            cancel_order(order, symbol)
            cancel_any = True
    if cancel_any:
        send_slack_message("Canceling all orders", slack_token, SLACK_SOMETHING_IRREGULAR_CHANNEL)


def log_position(df, past_quantity_column_name, change_quantity_column_name, entry_price_column_name, current_price_column_name):
    csv_file = './logs/position.csv'
    file_exists = os.path.isfile(csv_file)
    time_str = str(datetime.now())
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        if not file_exists:
            writer.writerow(['time', 'symbol', 'past_quantity', 'new_quantity', 'entry_price', 'current_price'])
        for symbol in df.index:
            past_quantity = df.loc[symbol, past_quantity_column_name]
            new_quantity = df.loc[symbol, change_quantity_column_name] + past_quantity
            entry_price = df.loc[symbol, entry_price_column_name]
            current_price = df.loc[symbol, current_price_column_name]
            writer.writerow([time_str, symbol, past_quantity, new_quantity, entry_price, current_price])

def slack_position(df, past_quantity_column_name, change_quantity_column_name, entry_price_column_name, current_price_column_name, is_dryrun=False, slack_token=None):
    if slack_token:
        msgs = []
        rows = [['symbol', 'past_quantity', 'new_quantity', 'entry_price', 'current_price']]
        for symbol in df.index:
            past_quantity = str(df.loc[symbol, past_quantity_column_name])
            new_quantity = str(df.loc[symbol, change_quantity_column_name] + df.loc[symbol, past_quantity_column_name])
            entry_price = str(round(df.loc[symbol, entry_price_column_name], 4))
            current_price = str(df.loc[symbol, current_price_column_name])
            rows.append([symbol, past_quantity, new_quantity, entry_price, current_price])
        column_widths = [max(len(str(cell)) for cell in column) for column in zip(*rows)]
        message = ""
        for row in rows:
            message += " | ".join(cell.ljust(width) for cell, width in zip(row, column_widths))
            message += "\n"
        channel = SLACK_DRYRUN_POSITION_CHANNEL if is_dryrun else SLACK_POSITION_CHANNEL
        send_slack_message(message, slack_token, channel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Receive input')
    parser.add_argument('--dryrun', default=True, help='dryrun', type=lambda x: x.lower() != 'false')
    parser.add_argument('--leverage', default=4, help='leverage', type=int)
    parser.add_argument('--budget_allocation', default=0.64, help='budget_allocation', type=float)
    parser.add_argument('--budget_keep', default=600, help='budget_keep', type=float)
    parser.add_argument('--api_key', help='api_key')
    parser.add_argument('--api_secret', help='api_secret')
    parser.add_argument('--slack_token', help='slack_token')
    args = parser.parse_args()
    is_dryrun, leverage, budget_allocation, budget_keep = bool(args.dryrun), int(args.leverage), float(args.budget_allocation), float(args.budget_keep)
    api_key, api_secret, slack_token = args.api_key, args.api_secret, args.slack_token
    if api_key is None or api_secret is None:
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_API_KEY')
    if slack_token is None:
        slack_token = os.getenv('PERSONAL_SLACK_TOKEN')
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    client = Client(api_key, api_secret)
    print(f'is_dryrun={is_dryrun}, leverage={leverage}, budget_allocation={budget_allocation}')
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
    dict_df_klines = {}
    df_current_futures_position, max_withdraw_amount = get_current_futures_position(symbols)
    old_leverage = df_current_futures_position['leverage'].iloc[0] # we assume leverage are all the same
    if not is_dryrun:
        cancel_all_orders(symbols)
        for symbol in symbols:
            client.futures_change_leverage(symbol=symbol, leverage=str(leverage))
    alphas = alpha_collection.Alphas()
    past_price = {symbol: [float(kline[4]) for kline in client.futures_historical_klines(symbol, '1h', '127 hours ago UTC')[:-1]] for symbol in symbols}
    current_price = {symbol: float(client.futures_ticker(symbol=symbol)['lastPrice']) for symbol in symbols}
    dict_df_close = {symbol: pd.DataFrame({'close': past_price[symbol] + [current_price[symbol]]}) for symbol in symbols}
    df_weight = pd.DataFrame(alphas.close_position_in_nday_bollinger_band_ewm(dict_df_close, n=125, shift=0)[0].iloc[-1].T.rename('next_position_usdt'))
    df_current_price_and_amount = pd.DataFrame.from_dict(current_price, orient='index', columns=['price']).join(df_current_futures_position)
    total_quantity_usdt = ((df_current_price_and_amount['positionAmt'].abs() * df_current_price_and_amount['price']).sum() / old_leverage + max_withdraw_amount)
    trading_quantity_usdt = (total_quantity_usdt - budget_keep) * budget_allocation
    df_quantity = df_weight * trading_quantity_usdt * leverage
    df_quantity_and_price = df_quantity.join(df_current_price_and_amount)\
        .assign(current_position_usdt=lambda x: x.positionAmt * x.price)\
        .assign(changing_position_usdt=lambda x: x.next_position_usdt - x.current_position_usdt)\
        .assign(margin_increase=lambda x: x.next_position_usdt.abs() - x.current_position_usdt.abs())
    df_quantity_and_price_trimmed = trim_quantity_df(df_quantity_and_price, usdt_column_name='changing_position_usdt', price_column_name='price').sort_values('margin_increase')
    order_data_list = order_with_quantity(df_quantity_and_price_trimmed, quantity_column_name='quantity_trimmed', price_column_name='price', leverage=leverage, is_dryrun=is_dryrun)
    if not is_dryrun:
        log_total_quantity(total_quantity_usdt)
        log_position(df_quantity_and_price_trimmed, past_quantity_column_name='positionAmt',
                     change_quantity_column_name='quantity_trimmed', entry_price_column_name='entryPrice',
                     current_price_column_name='price')
        log_order(order_data_list, is_dryrun=is_dryrun)
    slack_order(order_data_list, is_dryrun=is_dryrun, slack_token=slack_token)
    slack_total_quantity(total_quantity_usdt, is_dryrun=is_dryrun, slack_token=slack_token)
    slack_position(df_quantity_and_price_trimmed, past_quantity_column_name='positionAmt',
                   change_quantity_column_name='quantity_trimmed', entry_price_column_name='entryPrice',
                   current_price_column_name='price', is_dryrun=is_dryrun, slack_token=slack_token)
    while not is_dryrun and len(get_remaining_orders(symbols)) > 0:
        time.sleep(900)
        is_all_filled = renew_order_if_not_meet(symbols, leverage)
        if is_all_filled:
            break
    print()
