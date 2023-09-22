import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from binance.exceptions import BinanceAPIException
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QComboBox, QTextEdit, 
                             QHBoxLayout, QDoubleSpinBox, QFormLayout, QSpinBox, QTabWidget, QFrame, QMessageBox, QDateEdit)
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from PyQt5.QtCore import QDate
import time
from datetime import datetime

class ChartWindow(QWidget):

    def __init__(self, data, trade_events, predictions, main_window, client):
        super().__init__()
        self.data = data
        self.trade_events = trade_events
        self.predictions = predictions
        self.main_window = main_window
        

        # Dodaj widgety związane z wykresem
        self.startDateLabel = QLabel("Data początkowa:")
        self.startDateInput = QDateEdit(self)
        self.startDateInput.setCalendarPopup(True)
        self.endDateLabel = QLabel("Data końcowa:")
        self.endDateInput = QDateEdit(self)
        self.endDateInput.setCalendarPopup(True)
        self.startDateInput.setDate(QDate.currentDate())
        self.endDateInput.setDate(QDate.currentDate())
        self.indicatorsCheckbox = QComboBox(self)
        self.indicatorsCheckbox.addItems(["Średnia krocząca", "RSI", "MACD"])
        
        self.initUI()

        self.startDateInput.dateChanged.connect(self.refresh_chart)
        self.endDateInput.dateChanged.connect(self.refresh_chart)
        self.refreshButton = QPushButton('Odśwież', self)
        self.layout.addWidget(self.refreshButton)
        self.refreshButton.clicked.connect(self.refresh_chart)
        interval = self.main_window.intervalCombo.currentText()
        self.client = client
        klines = self.client.futures_klines(symbol="BTCUSDT", interval=interval, limit=100)



        self.data = self.main_window.prepare_data(klines)
        self.plot_data()
        
        try:
            interval = self.main_window.intervalCombo.currentText()
            klines = self.client.futures_klines(symbol="BTCUSDT", interval=interval, limit=100)
            self.data = self.main_window.prepare_data(klines)
        except BinanceAPIException as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się połączyć z Binance: {e.message}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił nieoczekiwany błąd: {str(e)}")

    def update_data(self):
        interval = self.main_window.intervalCombo.currentText()
        klines = self.main_window.client.futures_klines(symbol="BTCUSDT", interval=interval, limit=100)
        self.data = self.parent().prepare_data(klines)

    def refresh_chart(self):
        self.update_data()
        self.plot_data()


    def initUI(self):
        self.layout = QVBoxLayout()
        
        # Dodaj widgety do layoutu
        self.layout.addWidget(self.startDateLabel)
        self.layout.addWidget(self.startDateInput)
        self.layout.addWidget(self.endDateLabel)
        self.layout.addWidget(self.endDateInput)
        self.layout.addWidget(QLabel("Wskaźniki:"))
        self.layout.addWidget(self.indicatorsCheckbox)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.plot_data()

    def plot_data(self):
        if self.data.empty:  # Sprawdź, czy DataFrame jest pusty
            return  # Jeśli tak, zakończ funkcję
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.data['close'], label='Cena zamknięcia')
        ax.plot(self.data['ma7'], label='Średnia krocząca 7 dni')
        ax.plot(self.data['ma21'], label='Średnia krocząca 21 dni')
        ax.set_title('Historia cen BTCUSDT')
        ax.set_xlabel('Data')
        ax.set_ylabel('Cena')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        self.canvas.draw()
        for index, event in self.trade_events:
            color = 'g' if event == 'buy' else 'r'
            ax.plot(index, self.data['close'].iloc[index], color + 'o')
        ax.plot(self.predictions, 'bo', label='Przewidywania')

        # Filtruj dane według zakresu dat
        start_date = self.startDateInput.date().toPyDate()
        end_date = self.endDateInput.date().toPyDate()
        print(self.data['timestamp'].head())

        # Przekształć kolumnę 'timestamp' do formatu datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')


        # Filtruj dane według zakresu dat
        start_date = self.startDateInput.date().toPyDate()
        end_date = self.endDateInput.date().toPyDate()

        start_datetime = pd.Timestamp(start_date)
        end_datetime = pd.Timestamp(end_date)
        filtered_data = self.data[(self.data['timestamp'] >= start_datetime) & (self.data['timestamp'] <= end_datetime)]


        # Rysuj wybrane wskaźniki
        selected_indicator = self.indicatorsCheckbox.currentText()
        if selected_indicator == "Średnia krocząca":
            ax.plot(filtered_data['ma7'], label='MA7')
            ax.plot(filtered_data['ma21'], label='MA21')
        elif selected_indicator == "RSI":
            ax.plot(filtered_data['RSI'], label='RSI')
        elif selected_indicator == "MACD":
            ax.plot(filtered_data['MACD'], label='MACD')

class BinanceFuturesBotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.client = None  # Inicjacja atrybutu client jako None
        self.data = pd.DataFrame()
        self.initUI()
        self.trade_events = []  # [(index, 'buy'), (index, 'sell'), ...]
        
    def initUI(self):
        self.layout = QVBoxLayout()
        self.predictions = []

        # Tytuł i ustawienia okna
        self.setWindowTitle('Binance Futures Bot')
        self.setWindowIcon(QIcon('icon.png'))
        self.resize(700, 600)

        # Pola do wprowadzania klucza API i sekretu API
        self.apiKeyLabel = QLabel("API Key:")
        self.apiKeyInput = QLineEdit(self)
        self.apiSecretLabel = QLabel("API Secret:")
        self.apiSecretInput = QLineEdit(self)
        self.apiSecretInput.setEchoMode(QLineEdit.Password)

        # Ukrycie sekretu API
        self.layout.addWidget(self.apiKeyLabel)
        self.layout.addWidget(self.apiKeyInput)
        self.layout.addWidget(self.apiSecretLabel)
        self.layout.addWidget(self.apiSecretInput)

        # Przycisk połączenia z Binance Futures
        self.connectButton = QPushButton('Połącz z Binance Futures', self)
        self.connectButton.clicked.connect(self.connect_to_binance)
        self.layout.addWidget(self.connectButton)

        # Wybór interwału czasowego
        self.intervalCombo = QComboBox(self)
        self.intervalCombo.addItems(["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"])
        self.layout.addWidget(QLabel("Interwał czasowy:"))
        self.layout.addWidget(self.intervalCombo)

        # Przycisk do pobrania danych historycznych
        self.fetchDataButton = QPushButton('Pobierz dane historyczne', self)
        self.fetchDataButton.clicked.connect(self.fetch_historical_data)
        self.layout.addWidget(self.fetchDataButton)

        # Pole tekstowe do wyświetlania logów
        self.logText = QTextEdit(self)
        self.layout.addWidget(self.logText)

         # Przycisk do trenowania modelu
        self.trainModelButton = QPushButton('Wytrenuj model', self)
        self.trainModelButton.clicked.connect(self.train_ml_model)
        self.layout.addWidget(self.trainModelButton)

        # Przycisk do uruchamiania bota
        self.startBotButton = QPushButton('Uruchom bota', self)
        self.startBotButton.clicked.connect(self.run_bot)
        self.layout.addWidget(self.startBotButton)
        
        # Dodaj przycisk do otwarcia okna z wykresem
        self.showChartButton = QPushButton('Pokaż wykres', self)
        self.showChartButton.clicked.connect(self.show_chart_window)
        self.layout.addWidget(self.showChartButton)

        # Przycisk wyboru progu transakcji
        self.thresholdDialog = QDialog()
        self.thresholdDialog.setWindowTitle("Ustaw próg transakcji")

        self.thresholdLabel = QLabel("Próg transakcji:")
        self.thresholdInput = QLineEdit()
        self.thresholdInput.setPlaceholderText("0.01")

        self.thresholdButton = QPushButton("Zapisz")
        self.thresholdButton.clicked.connect(self.on_threshold_button_clicked)

        layout = QVBoxLayout()
        layout.addWidget(self.thresholdLabel)
        layout.addWidget(self.thresholdInput)
        layout.addWidget(self.thresholdButton)

        self.thresholdDialog.setLayout(layout)

        self.thresholdAction = QAction("Ustaw próg transakcji", self)
        self.thresholdAction.triggered.connect(self.on_threshold_action_triggered)

        self.menuBar.addAction(self.thresholdAction)

        self.setLayout(self.layout)

    def on_threshold_button_clicked(self):
        threshold = float(self.thresholdInput.text())
        self.threshold = threshold
        self.logText.append(f"Ustawiono próg transakcji na {threshold}%.")
        self.thresholdDialog.close()

    def on_threshold_action_triggered(self):
        self.thresholdDialog.show()

    def predict_prices(self, X):
        return self.model.predict(X)
    
    def make_trading_decision(self, current_price, predicted_price):
        threshold = float(self.thresholdInput.text())

        if abs(predicted_price - current_price) / current_price > threshold:
            # Składanie transakcji
            if predicted_price > current_price:
                self.trade_events.append((len(self.data) - 1, 'buy'))
            else:
                self.trade_events.append((len(self.data) - 1, 'sell'))
        else:
                # Nie składanie transakcji
                self.logText.append(
                    f"Nie składam transakcji. Różnica między przewidywaną ceną a bieżącą ceną wynosi {abs(predicted_price - current_price) / current_price}%.")


    def run_bot(self):
        try:
            if not hasattr(self, 'model'):
             QMessageBox.critical(self, "Błąd", "Najpierw wytrenuj model!")
            return

            interval = self.intervalCombo.currentText()
            klines = self.client.futures_klines(symbol="BTCUSDT", interval=interval, limit=100)
            data = self.prepare_data(klines)
            X = data.drop(['target', 'timestamp', 'close_time', 'ignore'], axis=1)
            predicted_prices = self.predict_prices(X)
            current_price = data['close'].iloc[-1]
            predicted_price = predicted_prices[-1]
            self.make_trading_decision(current_price, predicted_price)
            self.predictions.append(predicted_price)
        except BinanceAPIException as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas działania bota: {e.message}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił nieoczekiwany błąd: {str(e)}")

    def connect_to_binance(self):
        api_key = self.apiKeyInput.text().strip()
        api_secret = self.apiSecretInput.text().strip()

        # Sprawdzenie, czy klucze zostały podane
        if not api_key or not api_secret:
            QMessageBox.information(self, "Sukces", "Pomyślnie połączono z Binance Futures!")
            self.show_chart_window()

        try:
            self.client = Client(api_key, api_secret)
            self.client.futures_api_url = 'https://fapi.binance.com/fapi/v1'  
            exchange_info = self.client.futures_exchange_info()
            QMessageBox.information(self, "Sukces", "Pomyślnie połączono z Binance Futures!")
        except BinanceAPIException as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się połączyć z Binance Futures: {e.message}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił nieoczekiwany błąd: {str(e)}")

            QMessageBox.information(self, "Sukces", "Pomyślnie połączono z Binance Futures!")
            self.show_chart_window()


    def fetch_historical_data(self):
        if not hasattr(self, 'client'):
            QMessageBox.critical(self, "Błąd", "Najpierw połącz się z Binance!")
            return
        try:
            interval = self.intervalCombo.currentText()
            klines = self.client.futures_klines(symbol="BTCUSDT", interval=interval, limit=100)
            self.data = self.prepare_data(klines)
        except BinanceAPIException as e:
            QMessageBox.critical(self, "Błąd", f"Nie udało się pobrać danych historycznych: {e.message}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił nieoczekiwany błąd: {str(e)}")

        self.show_chart_window()

    def prepare_data(self, klines):
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Obliczanie wskaźników technicznych
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['26ema'] = df['close'].ewm(span=26).mean()
        df['12ema'] = df['close'].ewm(span=12).mean()
        df['MACD'] = (df['12ema'] - df['26ema'])

        # Obliczanie RSI
        delta = df['close'].diff()
        window = 14
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Wypełnianie brakujących wartości
        df = df.dropna()

        # Przesunięcie kolumny 'close' o -1, aby przewidywać przyszłą cenę
        df['target'] = df['close'].shift(-1)
        df = df.dropna()

        return df
    
    def split_data(self, df):
        X = df.drop(['target', 'timestamp', 'close_time', 'ignore'], axis=1)
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        self.logText.append(f"Mean Squared Error: {mse}")

    def train_ml_model(self):
        try:
            X_train, X_test, y_train, y_test = self.split_data(self.data)
            self.model = self.train_model(X_train, y_train)
            self.evaluate_model(self.model, X_test, y_test)
            self.logText.append("Model został wytrenowany!")
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas trenowania modelu: {str(e)}")

    def generate_signals(self, data):
        # Obliczanie średnich kroczących
        short_window = 50
        long_window = 200
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        signals['short_mavg'] = data['close'].rolling(window=short_window).mean()
        signals['long_mavg'] = data['close'].rolling(window=long_window).mean()

        # Generowanie sygnałów
        signals['signal'] = 0.0
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
        signals['positions'] = signals['signal'].diff()

        return signals
    
    def show_chart_window(self):
        self.chart_window = ChartWindow(self.data, self.trade_events, self.predictions, self, self.client)
        self.intervalCombo.currentIndexChanged.connect(self.chart_window.refresh_chart)
        self.chart_window.show()

    
def execute_trades(self, signals):
    for i in range(1, len(signals)):
        if signals['positions'].iloc[i] == 1.0:
            # Składanie zlecenia kupna
            self.client.futures_create_order(symbol="BTCUSDT", side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=1)  # Przykładowa ilość
        elif signals['positions'].iloc[i] == -1.0:
            # Składanie zlecenia sprzedaży
            self.client.futures_create_order(symbol="BTCUSDT", side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=1)  # Przykładowa ilość


def run_app():
    app = QApplication([])
    mainWindow = BinanceFuturesBotApp()
    mainWindow.show()
    app.exec_()

# Uruchomienie aplikacji
run_app()