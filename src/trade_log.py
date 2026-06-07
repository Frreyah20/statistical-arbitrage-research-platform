import pandas as pd 

def extract_trades(positions):
    trades = []
    entry_date = None
    position_type = 0

    if len(positions) > 0 and positions.iloc[0] != 0:
        entry_date = positions.index[0]
        position_type = positions.iloc[0]
        
    for i in range(1, len(positions)):
        previous = positions.iloc[i-1]
        current = positions.iloc[i]

        if previous == 0 and current != 0:
            entry_date = positions.index[i]
            position_type = current

        elif previous != 0 and current == 0:
            exit_date = positions.index[i]
            holding_days = (exit_date - entry_date).days
            trades.append((
                entry_date,
                exit_date,
                position_type,
                holding_days
            ))
            position_type = 0
            entry_date = None
    if position_type != 0:
        trades.append(
            (
                entry_date,
                positions.index[-1],
                position_type,
                (positions.index[-1] - entry_date).days
            )
        )


    return pd.DataFrame(trades, columns = ["Entry Date", "Exit Date", "Position Type", "Holding Days"])



