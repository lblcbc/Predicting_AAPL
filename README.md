# Predicting_AAPL
### (Then GS and ZM)

Random forest model that trains on historical data of AAPL containing, for each day, market return (%) (market the stock is listed on, in this case NASDAQ (GS = NYSE, ZM = NASDAQ), open-close % change, high-low %, volume (relative to first volume from the dataset), and further the previous days' values (up to 62 past days) of all these metrics. As a result, the data used to train contained 903 days worth of data. The label (aim of prediction) is to predict the next days return, first through 5 labels (jump (next day's return > 2.5%), up (> 0.5%), flat (0.5%-(-0.5%)), down (< -0.5%), fall (< -2.5%)), then second just predicting if tomorrow return is positive (up) or negative (down). 

Quick summary of key results (accuracy):
AAPL (5 label): 52%
AAPL (2 label): 67%
GS (2 label): 69%
ZM (2 label): 68%

Overall, results were surprisingly consistent, regarding GS being in a different sector to AAPL, and ZM contrasting with its downward trend between the historical data period (11/2020-07/2023) - ZM was used primarily to test if the model was achieving the 60%+ just by learning AAPL and GS went up more than they went down.

While this was interesting, there exist way more sophisticated and established papers trying to forecast stock prices/returns, so this experiment will end here for now. That said, I will still happily share some sample trees from the training and testing process along the way.

Sample tree (from random forest), 5 labels, AAPL:

![tree_sample_labels_5_aapl](https://github.com/lblcbc/Predicting_AAPL/assets/136857271/0d7a07dd-d92a-4cad-856e-5ad3c493a46a)


Sample tree, 2 labels, AAPL (and same sample for GS and ZM as the same Random Forest model was used):

![tree_sample_labels_2_aapl](https://github.com/lblcbc/Predicting_AAPL/assets/136857271/b32686b4-ec54-4618-8201-135b2af8a8c3)









