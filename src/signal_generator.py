import pandas as pd 

def generate_signals(zscore, entry_threshold = 2.0, exit_threshold = 0.5):
    signals = pd.Series(0, index = zscore.index)
    signals[zscore > entry_threshold] = -1 #short the spread (short first stock, long second stock)
    signals[zscore < -entry_threshold] = 1 #long the spread (long first stock, short second stock)
    #0 = no position
    return signals

def generate_positions(zscore, entry_threshold = 2.0, exit_threshold = 0.5):
    positions = pd.Series(0, index = zscore.index)
    current_position = 0
    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if current_position == 0:
            if z < -entry_threshold:
                current_position = 1

            elif z > entry_threshold:
                current_position = -1

        elif current_position == 1:
            if z > -exit_threshold:
                current_position = 0
        
        elif current_position == -1:
            if z < exit_threshold:
                current_position = 0

        positions.iloc[i] = current_position
    return positions
            
            


    