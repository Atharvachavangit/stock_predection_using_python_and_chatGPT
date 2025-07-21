import json
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import streamlit as st
from openai import OpenAI


client = OpenAI(api_key=open('API_KEY.txt', 'r').read().strip())


def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].close)

def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14 - 1, adjust=False).mean()
    ema_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return str((100 - (100 / (1 + rs))).iloc[-1])

def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    histogram = MACD - signal
    return f'{MACD.iloc[-1]}, {signal.iloc[-1]}, {histogram.iloc[-1]}'

def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'])
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()

functions = [
    {
        'name': 'get_stock_price',
        'description': 'Gets the latest stock price given the ticker symbol of a company',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'Stock ticker symbol (e.g., AAPL)'
                }
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_SMA',
        'description': 'Calculate Simple Moving Average',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string'},
                'window': {'type': 'integer'}
            },
            'required': ['ticker', 'window']
        }
    },
    {
        'name': 'calculate_EMA',
        'description': 'Calculate Exponential Moving Average',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string'},
                'window': {'type': 'integer'}
            },
            'required': ['ticker', 'window']
        }
    },
    {
        'name': 'calculate_RSI',
        'description': 'Calculate RSI (Relative Strength Index)',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_MACD',
        'description': 'Calculate MACD indicator',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'plot_stock_price',
        'description': 'Plot stock price over the past year',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string'}
            },
            'required': ['ticker']
        }
    }
]

available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price
}

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title(' Stock Analysis Chatbot Assistant')

user_input = st.text_input('Your Input')

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': user_input})

        # Step 1: Ask OpenAI for a response
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=st.session_state['messages'],
            functions=functions,
            function_call='auto'
        )

        response_message = response.choices[0].message
        st.session_state['messages'].append(response_message)

        # Step 2: If a function is called
        if response_message.function_call:
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)

            function_to_call = available_functions.get(function_name)
            function_response = function_to_call(**function_args)

            # Step 3: Add function result to conversation
            st.session_state['messages'].append({
                'role': 'function',
                'name': function_name,
                'content': function_response
            })

            if function_name == 'plot_stock_price':
                st.image('stock.png')
            else:
                # Step 4: Ask OpenAI to summarize the result
                second_response = client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=st.session_state['messages']
                )
                final_content = second_response.choices[0].message.content
                st.text(final_content)
                st.session_state['messages'].append({
                    'role': 'assistant',
                    'content': final_content
                })

        # If no function call, just show the response
        else:
            st.text(response_message.content)
            st.session_state['messages'].append({
                'role': 'assistant',
                'content': response_message.content
            })

    except Exception as e:
        st.error(f" Error: {e}")
