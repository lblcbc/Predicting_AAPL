# Predicting_AAPL
### (Then GS and ZM)

Random forest model that trains on historical data of AAPL containing, for each day, market return (%) (market the stock is listed on, in this case NASDAQ (GS = NYSE, ZM = NASDAQ), open-close % change, high-low %, volume (relative to first volume of dataset), and further the previous days' values (up to 62 past days) of all these metrics. As a result the resulting data used to train contained 903 days worth of data. The label (aim of prediction) is to predict the next days return, first through 5 labels (jump (next day's return > 2.5%), up (> 0.5%), flat (0.5%-(-0.5%)), down (< -0.5%), fall (< -2.5%)), then second just predicting if tomorrow return is positive (up) or negative (down). 

Results summaries (accuracy):
AAPL (5 label): 52%
AAPL (2 label): 67%
GS (2 label): 69%
ZM (2 label): 68%

Overall, results were surprisingly consistent, regarding GS different sector to AAPL, and ZM contrasting downward trend between data period (11/2020-07/2023), we used ZM to see if the model was achieving the 60%+ just by learning AAPL and GS went up more than they went down.

While this was interesting, there exist way more sophosticated and established papers trying to forecast stock prices/returns, so this experiment will end here for now.
