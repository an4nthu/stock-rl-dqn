import numpy as np

class StockTradingEnv:
    def __init__(self, data, initial_balance=1000):
        self.data = data.reset_index(drop=True) # ensure 0 based index
        self.n_steps = len(data)  #num of rows
        self.initial_balance = initial_balance
        self.reset()  #calls reset from init

    # Environment state
    def reset(self):  #initialize balance, profit,current steps etc
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.total_shares_sold = 0
        self.current_step = 0
        return self._get_state()
            
        
    def _get_state(self):
        
        frame = self.data.iloc[self.current_step]  #get current day's row (date , closing price)
        return np.array([
            self.balance,
            self.shares_held,
            self.net_worth,  #cash + stock value
            frame['Close']
        ], dtype=np.float32)
    
    def step(self, action):
        if self.current_step >= len(self.data):
            return np.array([]), 0, True, {}

        done = False
        current_price = self.data.iloc[self.current_step]['Close']

        # Action: 0 = hold, 1 = buy, 2 = sell
        if action == 1:  # Buy one share if cash allows
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price

        elif action == 2:  # Sell one share if holding
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                self.total_shares_sold += 1

        self.current_step += 1  #This day over
        self.prev_net_worth=self.net_worth #prev day
        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth-self.prev_net_worth # trains agent (reward based) for EACH DAY.
        

        if self.current_step == len(self.data) : # All days over
            return np.array([]), reward, True, {}
        state = self._get_state()  #state after this day , return for next day

        return state, reward, done, {}
    
    def render(self):
        if(self.current_step==0):
            print(f"Initial balance : {self.initial_balance}")
        else:
            print(f"Step: {self.current_step}/{self.n_steps} | "
                f"Price: {self.data.iloc[self.current_step-1]['Close']} | "
                f"Balance: {self.balance:.2f} | "
                f"Shares: {self.shares_held} | "
                f"Net Worth: {self.net_worth:.2f}")



