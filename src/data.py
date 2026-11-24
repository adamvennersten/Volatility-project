"""Hämta kursdata och räkna ut avkastning och volatilitet."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRADING_DAYS = 252  # ungefär antal handelsdagar per år

# Yahoo-ticker -> namn vi använder i kolumnerna
TICKERS = {
    "^OMX": "OMXS30",
    "VOLV-B.ST": "Volvo B",
    "ERIC-B.ST": "Ericsson B",
    "HM-B.ST": "H&M B",
    "SEB-A.ST": "SEB A",
    "ATCO-A.ST": "Atlas Copco A",
}

def load_prices(tickers=TICKERS, start="2010-01-01", refresh=False) -> pd.DataFrame:
    """Justerade stängningskurser, en kolumn per aktie.

    Sparas i data/prices.csv så att vi bara laddar ner en gång.
    Kör med refresh=True för att hämta ny data.
    """
    path = DATA_DIR / "prices.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf

    # threads=False: undviker "database is locked" i yfinance
    raw = yf.download(list(tickers), start=start, auto_adjust=True,
                      progress=False, threads=False)
    prices = raw["Close"].rename(columns=tickers).dropna(how="all")

    missing = [n for n in tickers.values() if n not in prices or prices[n].isna().all()]
    if missing:
        raise RuntimeError(f"Kunde inte hämta {missing}. Kör igen.")

    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(path)
    return prices

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daglig log-avkastning: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def realized_vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Volatiliteten de senaste `window` dagarna (inklusive idag), årsräknad.

    sqrt(medel av r^2) — med dagliga data är medelavkastningen ~0,
    så det är i praktiken standardavvikelsen.
    """
    return np.sqrt(returns.pow(2).rolling(window).mean() * TRADING_DAYS)


def future_vol(returns: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Volatiliteten de KOMMANDE `horizon` dagarna (t+1 ... t+horizon).

    Det här är målvariabeln vi ska förutsäga. Raden för dag t innehåller
    alltså framtida information — använd den aldrig som feature.
    """
    sq = returns.pow(2)
    fwd = sq[::-1].rolling(horizon).mean()[::-1].shift(-1)
    return np.sqrt(fwd * TRADING_DAYS)
