This is a bitcoin trading bot I made for my final project. It trades BTC/USDT on a Binance demo account, so no real money is ever used, and it can't be switched over to real money either. It looks at the market on its own and decides when to buy and sell. You don't pick any trades yourself.

You need Python installed to run it. Open a terminal inside the project folder (on Windows, open the folder, click the address bar at the top, type cmd and press enter) and type this to install everything it needs:

pip install -r requirements.txt

Then make your settings file by copying .env.example and calling the copy .env. On Windows that is:

copy .env.example .env

On a Mac it is cp instead of copy. Open the new .env file in Notepad. The line that says TRADING_MODE decides how the bot runs. paper is the default. It uses the real live bitcoin price but fakes the trades, and it needs no keys (it still needs internet though). demo places orders on the Binance demo account, which needs keys (see below). testnet is the same idea but on a different Binance test site that uses its own made up prices. If you put live it will refuse to start.

To start the bot type this and press enter:

python main.py live

If python doesn't work try py or python3. It checks every 60 seconds, and each time a 15 minute candle finishes it makes a decision. Most of the time that decision is HOLD. That's normal, it's picky and only trades when a lot of things line up at the same time. Press Ctrl+C to stop it. Everything it does is saved in the logs folder. trading_bot.log is the full log and trades.csv is just the trades. If you stop the bot while it has a trade open it remembers it for next time. To start completely fresh, delete the logs folder.

To use demo mode you need demo keys. Log into your Binance account, open Binance Demo Trading and go to this page:

https://demo.binance.com/en/my/settings/api-management

Make an HMAC key there and copy the secret straight away because it only shows once. Then put the keys in the .env file like this, with no quotes and no spaces:

BINANCE_API_KEY=your key here

BINANCE_API_SECRET=your secret here

TRADING_MODE=demo

Don't share your keys with anyone. The .env file is already set up so git won't upload it.

Run python main.py live again and the first thing you will see is MODE: DEMO. You can have the Binance demo trading page open next to it and watch the trades show up.

If you don't want to wait around for trades there is also a backtest. It runs the bot over old prices and gives you a report with the win rate, ROI, Sharpe ratio and drawdown. To try it with the sample data that is included:

python main.py backtest --data sample_data/sample_btcusdt_15m.csv --relaxed-demo --plot

The sample data is fake, so --relaxed-demo loosens the rules a bit just so it makes some trades. Don't use that with real data. To use real prices, download a year of them first and then run the backtest on that file:

python fetch_historical_data.py --days 365 --out btc_data.csv

python main.py backtest --data btc_data.csv --plot

The report gets saved as backtest_report.txt and the graph as backtest_report_equity.png.

There are also three test files that check everything works without needing internet:

python tests/test_orchestration.py

python tests/test_exchange_modes.py

python tests/test_bot_logic.py
