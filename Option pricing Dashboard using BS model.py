#!/usr/bin/env python
# coding: utf-8

# Steps to follow:
# 
# 1. Business Objective
# 
# 2. Load Market Data
# 
# 3. Calculate Returns
# 
# 4. Estimate Volatility
# 
# 5. Price Option
# 
# 6. Calculate Greeks
# 
# 7. Stock Price Sensitivity
# 
# 8. Volatility Sensitivity
# 
# 9. Scenario Analysis
# 
# 10. Risk Summary Dashboard
# 
# 11. Key Findings

# Objective:
# 
# Estimate the fair value of a European Call Option on Reliance Industries using Black-Scholes.
# 
# Assess:
# 
# - Option Value
# - Delta Risk
# - Volatility Risk
# - Scenario Risk

# In[78]:


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# In[79]:


ticker='RELIANCE.NS'

rel=yf.download(ticker, start='2016-06-01', end='2026-06-01')
rel


# In[80]:


rel.tail(20)


# In[81]:


rel['returns']=np.log(rel['Close']/rel['Close'].shift(1))
rel.dropna()


# In[82]:


rel['returns'].mean()*100


# In[83]:


daily_vol=rel['returns'].std()
daily_vol


# In[84]:


annual_vol=daily_vol*np.sqrt(252)  #same as daily_vol*(252)**0.5
annual_vol


# Volatility answers:
# 
# How unpredictable is the stock?
# 
# -> the stock is annually 27% volatile.

# In[85]:


from scipy.stats import norm


# In[86]:


#black-scholes function for call price:

def call_price(S,K,T,r,sigma):

    d1= (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*(T**0.5))
    d2= d1- sigma*T**0.5

    call= (S*norm.cdf(d1))-(K*np.exp(-r*T)*norm.cdf(d2))

    return call


# In[87]:


#give the inputs and accordingly will get output via function

S= rel['Close'].iloc[-1]  # .iloc[-1] is basically to select the very last (most recent) closing price, e.g., 29th may

K=S*1.05 #assuming strike price is 5% higher than current price i.e., pricing call at OTM
#K=S*0.70 # deep ITM
#K=S #ATM

T=0.5 # in years
r=0.06 #risk-free interest rate
sigma=annual_vol


# In[88]:


option_value= call_price(S,K,T,r,sigma)
option_value


# So, 
# 
# -the current price is the latest closing price here which 29th May
# 
# -if we adjust K below/above/remain same to S, accordingly call option will be priced at ITM/OTM/ATM respectively.
# 
# Option value/premium = intrinsic value + time value

# In[89]:


def delta(S,K,T,r,sigma):

    d1= (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*(T**0.5))

    return norm.cdf(d1)  #this is delta


# In[90]:


delta= delta(S,K,T,r,sigma)
delta


# #FIRST RISK MEASURE
# 
# If stock increase by 100, call price will increase by 50

# # how option price change due to price sensitivity

# In[91]:


# Extract the raw number
current_S = float(rel['Close'].iloc[-1])  #the recent closing price

# now let say price vary from -20% to +20% of the raw_S, consider 20 such prices
stock_range = np.arange(current_S * 0.8, current_S * 1.2, 20)
stock_range


# In[92]:


option_prices1=[]

for stock in stock_range:
    call_value_price_sensitivity= call_price(stock,K,T,r,sigma)  #using cell 11 & 12, we can change variable value(K,T,r,sigma) as per requirement
    
    option_prices1.append(call_value_price_sensitivity)  #.append() = this allow each element to be in the exisitng list altogether

option_prices1


# In[93]:


# plot to visualize the relationship between stock prices and option value

plt.figure(figsize=(8,5))
plt.plot(stock_range, option_prices1)
plt.title('call price vs stock price')
plt.xlabel('stock price')
plt.ylabel('call price')
plt.grid()
plt.show()  


# # option price change due to volatility senstivity

# In[94]:


vol_range=np.arange(0.10,0.60,0.02)
vol_range


# In[95]:


option_prices2=[]

for volatility in vol_range:
    call_value_volatility_senstivity=call_price(S,K,T,r,volatility)
    
    option_prices2.append(call_value_volatility_senstivity)
    
option_prices2


# In[96]:


# plot to visualize the relationship between stock prices and option value

plt.figure(figsize=(8,5))
plt.plot(vol_range, option_prices2)
plt.title('call price vs volatility change')
plt.xlabel('volatility change')
plt.ylabel('call price')
plt.grid()
plt.show()  


# SO the above is vega risk

# # scenarios analysis

# In[97]:


# create various scenarios(worst/best/no change) to get the idea of how my portfolio would get changed 

scenarios=[-0.30,-0.10,0,0.10,0.30]


# market crashed by -30% / -10% or got a push by +30% / +10%

# In[98]:


new_S = S # recent closing price i.e, 29th may

resulted_price=[]

for s in scenarios:
    new_stock= new_S*(1+s)
    
    new_option_price=call_price(new_stock,K,T,r,sigma)
    
    pnl= new_option_price-option_value
    
    resulted_price.append([s,new_stock,new_option_price,pnl])
    
resulted_price


# So the resulted_price is basically - 
# 
# market change by---> new stock price----> new call price ----> profit or loss comparing with the intial call price

# In[99]:


#create dataframe for the above result

scenario_df= pd.DataFrame(resulted_price,columns=['scenario','stock price','option price', 'PnL'])


# In[100]:


scenario_df


# # Dashboard 

# In[102]:


dashboard= pd.DataFrame({
    'Metrics':['Current stock price','volatiltiy','option value','delta'],
    'Value':[round(S,2),round(sigma,3),round(option_value,2),np.round(delta,4)]
})

dashboard


# In[108]:


raw_S = float(rel['Close'].iloc[-1])
raw_option_value= float(option_value)
raw_delta= float(delta[0])

print("="*50)

print("OPTION RISK DASHBOARD")

print("="*50)

print(f"Stock Price : {raw_S:.2f}")

print(f"Volatility : {sigma:.2%}")

print(f"Option Value : {raw_option_value:.2f}")

print(f"Delta : {raw_delta:.2f}")

print("="*50)


# In[ ]:




