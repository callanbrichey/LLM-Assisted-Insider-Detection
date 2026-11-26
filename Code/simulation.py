import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DAYS = 6_000
DEFAULT_NSTEPS = 23_400 // 5
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_ANNUAL_DRIFT = 0.1
DEFAULT_ANNUAL_VOLATILITY = 0.2
DATA_DIR = Path('../Outputs/data')
LOGS_DIR = Path('../Outputs/logs')
INTRADAY_CSV = DATA_DIR / 'synthetic_intraday_market_data.csv'


def development_day_count(days):
    return days * 3 // 4


def require_code_directory():
    if Path.cwd().name != 'Code' or not Path('simulation.py').is_file():
        raise SystemExit('Run this command from the repository’s Code directory.')


class MarketSimulation():          
   
    def __init__(self,
                 
                 ## Model Parameters
                 days: int,                  # Number of days in simulation (required)
                 nsteps: int = DEFAULT_NSTEPS, # Number of time steps in single trading day (defaults to number of 5-second intervals in full trading day)
                 days_in_year: int = DEFAULT_TRADING_DAYS_PER_YEAR, # Number of trading days in a year
                 
                 ## Asset Parameters
                 P0: float = 100.0,         # Initial asset price (expressed in $)
                 muP: float = DEFAULT_ANNUAL_DRIFT, # Mean annual return
                 sigmaP: float = DEFAULT_ANNUAL_VOLATILITY, # Annual volatility (as standard deviation)
                 
                 ## Market Liquidity Parameters of Noisy Traders
                 muU: float = 0,         # Mean noisy order flow per Kyle (1985)
                 targetU: float = 10**6, # Mean number of shares traded per day
                 ratioU: float = 0.03,   # Daily standard deviation of noisy order flow as a fraction of daily total volume (buy + sell orders)

                 ## Informed Trader Parameter
                 alpha : float = 0.4): # The probability that the informed trader is active on a given day
       
        ## Store Inputs
        self.days = days
        self.nsteps = nsteps
        self.days_in_year = days_in_year
        self.P0 = P0
        self.logP0 = np.log(P0)
        self.muP = muP
        self.sigmaP = sigmaP
        self.muU = muU
        self.targetU = targetU
        self.ratioU = ratioU
        self.alpha = alpha
       
        self.T = self.days / self.days_in_year                                # Time horizon expressed in years
        self.dd = self.T / self.days                                          # Change in days
        self.dt = self.dd / self.nsteps                                       # Change in time intraday
        self.sigmaU = self.ratioU * np.sqrt(self.days_in_year) * self.targetU # Annualised standard deviation of noisy order flow
        self.varU = self.sigmaU**2                                            # Annualised variance of noisy order flow
        
        pass
   
    def true_value(self):

        ## Calculate the daily true (or liquidation) values of the asset, which 
        ## follows a geometric Brownian motion (Back, 1992)
        logV_d = self.logP0 + np.cumsum((self.muP - 0.5 * self.sigmaP**2)*self.dd + self.sigmaP*np.random.randn(self.days)*np.sqrt(self.dd))
           
        return logV_d
   
    def simulation(self, logV_d):
       
        t = np.linspace(0, self.dd, self.nsteps + 1)
       
        ## NOTE: For all variables below, '_d' indicates w.r.t a trading day and '_t' indicates w.r.t an intraday time step
        logP_d = [self.logP0] # Initial Price
        logP_t = np.zeros((self.days, self.nsteps + 1))
       
        Net_Ord_Flow = np.zeros((self.days, self.nsteps))
       
        K_d = np.zeros(self.days)

        var_logP_d = self.sigmaP**2 * self.dd
        
        for day in range(self.days):      

            lmda_d = np.sqrt(var_logP_d / (self.varU * self.dd)) # Market Maker Parameter (Kyle, 1985). Adjusted on a per-day basis

            ## Opening log price on a given day is equal to the last closing price
            logP_t[day, 0] = logP_d[-1] # * (1 + self.muP*self.dd) # Uncomment if you would like the MM to price in the drift, keep it
                                                                   # commented if you would like it to be private information for the informed trader.

            k = 1 if np.random.uniform() <= self.alpha else 0 # Determine if insider trader is active on a given day
            K_d[day] = k # K_d is the signal for each day
           
            for i in range(self.nsteps):
           
                Sigma_t = max(((self.dd - t[i]) / self.dd) * var_logP_d, 1e-12)      # Tiny epsilon for numerical stability        
                beta_t = (1/np.sqrt(self.alpha)) * self.varU * lmda_d / Sigma_t  # Informed Trader Parameter, Kyle (1985)
                ## Note: for x_t, see above; may want to multiply P_t[day, i] by * (1 + self.muP*self.dt)
                x_t = beta_t * (logV_d[day] - logP_t[day, i]) * self.dt                # Kyle (1985) and Back (1992). Note that the informed trades are of smaller order (dt)
                u_t = np.random.normal(self.muU, self.sigmaU) * np.sqrt(self.dt) # versus the uninformed trades sqrt(dt).
               
                Net_Ord_Flow[day, i] = k * x_t + u_t # Calculate aggregate volume at each time step
           
                logP_t[day, i+1] = logP_t[day, i] + np.sqrt(self.alpha) * lmda_d * Net_Ord_Flow[day, i] # Calculate intraday prices at each time step     
                
            logP_d.append(logP_t[day, -1]) # Append daily closing prices
               
            ## Force end-of-day convergence to the true value for numerical stability. Not implemented because the
            ## informed trader is only active on certain days, meaning we only want the closing price to converge to
            ## the true value on days they are active.
            # logP_d[-1] = logV_d[day]
            # logP_t[day, -1] = logV_d[day]
           
        return logP_t, Net_Ord_Flow, K_d
   
    def dataset(self, logP_t, Net_Ord_Flow, K_d, filename=None):
        """Build an intraday table for a simulation or a prefix starting on day 1."""
        days, nsteps = Net_Ord_Flow.shape
        day = np.repeat(np.arange(1, days + 1), nsteps)
        timestamp = np.tile(np.arange(nsteps), days)
        price = np.exp(logP_t[:, 1:]).ravel()
        net_ord_flw = Net_Ord_Flow.ravel()
        signal = np.repeat(K_d.astype(int), nsteps)
        returns = np.diff(logP_t, axis=1).ravel()

        data = pd.DataFrame({
            'Day': day,
            'Timestamp': timestamp,
            'Price ($)': price,
            'Log Return': returns,
            'NetOrderFlow': net_ord_flw,
            'Signal': signal
        })

        if filename is not None:
            data.to_csv(filename, index=False)
        return data 


def run_simulation(days, nsteps, seed):
    """Run the simulation and return its inputs for downstream stages."""
    if seed is not None:
        np.random.seed(seed)
    sim = MarketSimulation(days=days, nsteps=nsteps)
    daily_log_values = sim.true_value()
    log_prices, order_flow, signals = sim.simulation(daily_log_values)
    return sim, daily_log_values, log_prices, order_flow, signals


def terminal_log_price_gaps(log_prices, daily_log_values, signals):
    """Return closing log-price gaps for active and inactive insider days."""
    gaps = log_prices[:, -1] - daily_log_values
    return gaps[signals == 1], gaps[signals == 0]

if __name__ == '__main__':
    require_code_directory()
    print('Starting Simulation...')
    start = time.perf_counter()
    sim = MarketSimulation(DEFAULT_DAYS)
    daily_log_values = sim.true_value()
    log_prices, order_flow, signals = sim.simulation(daily_log_values)
    INTRADAY_CSV.parent.mkdir(parents=True, exist_ok=True)
    sim.dataset(log_prices, order_flow, signals, INTRADAY_CSV)
    stop = time.perf_counter()
    print('Simulation Terminated.')
    print(f'Total Simulation Time: {stop - start:.2f} seconds.')

