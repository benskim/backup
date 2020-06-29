# mathmatics usage in Finance 2

## Integration, Cointegration, and Stationarity

### Stationarity 

status of no significant pattern  in time series

: 랜덤한 움직임을 갖지만 시점마다 유사하게 동작하는 특징이 있어 시간에 따라 안정된 분포를 갖는다. 

> Wold's theorem
>
> Stationary time series can be expressed as the combination of white noise
> Stationary time series can be expressed as Moving Average Representation

### Weak stationarity 
: 시간에 따라 일정한 평균, 분산, 공분산을 갖는 확률과정으로 정의된다. 

- Time series Yt has same mean at any time moments
- Time series Yt has same variance at any time moments 
- covariance of two time series Yt, Ys have same as other pairs with |t−s|=h ; Cov(Yt,Ys)=Cov(Yt+d,Ys+d)

### Non-stationarity
 : sample mean won't be useful for any forecasting of future state

- the changing mean of the series makes data non-stationary.
- noisy data and limited sample size  : HARD to recognize random noise or trend

### How to change Non-stationarity to stationarity

- log transformation: log(Yt) - 분산이 커지는 경향을 갖는 시계열을 안정화 시킴.
- Difference: diff(Yt) - 추세를 제거하는 효과를 거둠.
- seasonality difference : diff(Yt,s) - 계절추세를 제거하는 효과를 거둠

```python
from statsmodels.tsa.stattools import coint, adfuller

# Additive Returns
returns_I1 = X.diff()[1:]
# Multiplicative Returns
returns_ = X.pct_change()[1:]

# H_0 in adfuller is unit root exists (non-stationary)
# We must observe significant p-value to convince ourselves that the series is stationary
pvalue = adfuller(returns_I1)[1]
if pvalue < cutoff:
  print 'p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely stationary.'
else:
  print 'p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely non-stationary.'


```

### Order of Integration

the minimum number of differences required to obtain a [covariance-stationary](https://en.wikipedia.org/wiki/Stationary_process#Weak_or_wide-sense_stationarity) series ; I ( #diff )

### Cointegration

- each time series hase shared same 'order of integration'

- new time seires, created by linear combination of existing time series, has lower order of integration than.

  > Y = X1 - X2

#### Test for Cointegration

- built-in method

```python
from statsmodels.tsa.stattools import coint

X1,X2 = prices['AAPL'], prices['MSFT']
# MacKinnon’s approximate 
if coint(X1, X2)[1] < 0.05 :
  print 'two assets were cointegrated over the same timeframe'
```

- Self-made code

If cointegrated, we can remove ![$X_2$](https://render.githubusercontent.com/render/math?math=X_2&mode=inline)'s depedency on ![$X_1$](https://render.githubusercontent.com/render/math?math=X_1&mode=inline). The combination ![$X_2 - \beta X_1 = \alpha + \epsilon$](https://render.githubusercontent.com/render/math?math=X_2%20-%20%5Cbeta%20X_1%20%3D%20%5Calpha%20%2B%20%5Cepsilon&mode=inline) should be stationary.

```python
X1,X2 = prices['AAPL'], prices['MSFT']
X1 = sm.add_constant(X1)
results = sm.OLS(X2, X1).fit()
X1 = X1['AAPL']
b = results.params['AAPL']
Z = X2 - b * X1
if adfuller(Z)[1] < 0.05 :
  print 'two assets were cointegrated over the same timeframe'
```



* Tests for consistency of stationarity such as cross validation and out of sample testing are necessary. 



## AutoRegressive model/process

### Definition - 1. Concept 

output variable depends linearly on its own previous values and on a [stochastic](https://en.wikipedia.org/wiki/Stochastic_variable) term

### Definition - 2. Covariance stationary conditions 

For an AR model to function properly, we must require that the time series is covariance stationary. 

1. The expected value of the time series is constant and finite at all times, i.e. E[yt]=μ and μ<∞ for all values of t.
2. The variance of the time series is constant and finite for all time periods.
3. The covariance of the time series with itself for a fixed number of periods in either the future or the past is constant and finite for all time periods

> Tips : How to make a non-stationary time series stationary
>
> -  difference, log transformation, box-cox transformation

### Characteristics 

- has more extreme values than data from a normal distribution : **tail-fatness**
- has a tail heavy and non-normal distribution of outcomes
- **its estimates of variance will be wrong**

> Tips : Newey-West estimate for correcting variance : tail risk

### How to build model

1. Testing for AR behavior

```python
from statsmodels import tsa
from statsmodels.tsa.stattools import acf, pacf

# We have to set a confidence level for our intervals, we choose the standard of 95%,
# corresponding with an alpha of 0.05.
X_acf, X_acf_confs = acf(X, nlags=nlags, alpha=0.05)
X_pacf, X_pacf_confs = pacf(X, nlags=nlags, alpha=0.05)


def plot_acf(X_acf, X_acf_confs, title='ACF'):
    # The confidence intervals are returned by the functions as (lower, upper)
    # The plotting function needs them in the form (x-lower, upper-x)
    errorbars = np.ndarray((2, len(X_acf)))
    errorbars[0, :] = X_acf - X_acf_confs[:,0]
    errorbars[1, :] = X_acf_confs[:,1] - X_acf

    plt.plot(X_acf, 'ro')
    plt.errorbar(range(len(X_acf)), X_acf, yerr=errorbars, fmt='none', ecolor='gray', capthick=2)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(title);
    
plot_acf(X_acf, X_acf_confs)
plot_acf(X_pacf, X_pacf_confs, title='PACF')
```

2. Fitting a model
3. Choosing the number of lags

```python
N = 10
AIC = np.zeros((N, 1))

for i in range(N):
    model = tsa.api.AR(X)
    model = model.fit(maxlag=(i+1))
    AIC[i] = model.aic
    
AIC_min = np.min(AIC)
model_min = np.argmin(AIC)

print 'Relative Likelihoods'
print np.exp((AIC_min-AIC) / 2)
print 'Number of parameters in minimum AIC model %s' % (model_min+1)
```

4. Evaluating residuals

```python
model = tsa.api.AR(X)
model = model.fit(maxlag=3)

from statsmodels.stats.stattools import jarque_bera

score, pvalue, _, _ = jarque_bera(model.resid)

if pvalue < 0.10:
    print 'We have reason to suspect the residuals are not normally distributed.'
else:
    print 'The residuals seem normally distributed.'
```

5. out of sample testing

   

## AutoRegressive Conditionally Heteroskedastic (ARCH)

- Generalized ARCH = GARCH

1. Testing
2. Fitting / choosing parameters

- using MLE : Maximum Likelihood Estimate
- using GMM :  Generalized Method of Moments

3. Evaluating

4. OOS testing

```python

```



## 