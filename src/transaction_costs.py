class TransactionCostModel:
    def __init__(self, commission_bps=1, spread_bps=2, slippage_bps=2):
        self.commission_bps = commission_bps
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
    
    def get_total_bps(self):
        """Returns the total transaction cost in basis points."""
        return self.commission_bps + self.spread_bps + self.slippage_bps
    
    def get_total_rate(self):
        """Returns the total transaction cost rate as a decimal (e.g., 5 bps = 0.0005)."""
        return self.get_total_bps() / 10000.0
    
    def calculate_cost(self, notional):
        """
        Calculates the transaction cost given a traded notional amount.
        """
        return notional * self.get_total_rate()
