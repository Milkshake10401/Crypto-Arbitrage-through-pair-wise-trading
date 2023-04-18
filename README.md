Crypto Arbitrage Pair-wise Trading - Final Project (Math 140)

The topic we chose for our project was a form of statistical arbitrage, in particular Pairs Trading applied to the cryptocurrency market. The main portion of this project involves standard statistical methods to identify pairs of historically related cryptocurrencies and a construction of a neural network to exploit market differences in the price of two paired cryptocurrencies.

The bulk of our project was collecting data from the internet, developing a Python script to identify pairs of cryptocurrencies, and using a neural netowrk to predict future prices in order to exploit arbitrage opportunities. The neural network used historical data to train, in order to estimate exactly when to enter and exit trading positions using the cryptocurrency pairs.

In order to accomplish this task we had to build a web scraper using the BeautifulSoup Python library in order to allow for future conduction of trades. We also gathered historical data to select pairs and train the neural network. This historical data was from cryptodatadownload.com. The historical data was 50 crypto coins over a 5 month period. 

This data was then fed into the python script which conducted statistical analysis to identify pairs that possessed similar price time-series behavior in the market. For us to arrive that at these pairs, we had to use several python libraries. The libraries we picked in Python allowed us to use the clustering algorithm called OPTICS, perform a cointegration test and view the characteristic of the Hurst Exponent. 

We first use OPTICS to group cryptocurrency data. We then used the cointegration test on pairs within the clusters, to see if there exists a linear combination of the two cryptocurrency price time-series that is stationary. After this process was accomplished, we then used the HURST exponent, to evaluate if the spread, or in our naive appoach, the difference between the elements of the pair of two cointegrated cryptocurrencies, had a stationary time-series that reverted back to
it’s mean.

After the Python program ran successfully, we attained three pairs that we analyzed. We identified the cryptocurrency pairs of Celer and Doge, Compound and Solana, and Shiba and TRX. We then trained a neural network to predict when to enter and exit the market. Here the neural network we used was a Long Short-term Memory(LSTM) layer followed by a fully connected layer.

With the time that we had, the neural network was not able to successfully predict future pricing of paired cryptocurrencies effectively to net a profit from given market arbitrage opportunities. We successfully completed our goal of identifying pairs, and we attempted to complete the goal of using the neural network to identify and make profitable trades. We believe more time and resources would have allowed for a better final outcome for the project. We also believe the substantial amount of programs that were completed set us up for future success when continuing to develop this project.
