# mathmatics usage in Finance 1

## numpy

```python
import numpy as np

N = 10
assets = np.zeros((N, 100))
returns = np.zeros((N, 100))
R_1 = np.random.normal(1.01, 0.03, 100)
returns[0] = R_1
assets[0] = np.cumprod(R_1)

nanmean_p = np.nanmean(v)
cov_mat = np.cov(returns)
vol_a_plt = np.sqrt(np.var(np.dot(weights, returns), ddof=1)) # sample variance
```

## pandas

```python
returns = pd.DataFrame(np.random.normal(1, 3, (100, 10)))
prices = returns.cumprod()

# resampling date-index
def custom_resampler(array_like):
    """ Returns the first value of the period """
    return array_like[0]

prices.resample('M', how='median')
prices.resample('M', how=custom_resampler)

# converting timezone 
prices.tz_convert('US/Eastern')
pd.date_range(start=start, end=end, freq='D', tz='UTC')

# imputing missing index/values 
prices.reindex(calendar_dates, method='ffill')
prices.fillna(method='bfill')

# built-in methods
pd.rolling_mean(prices, 30)
```

## plotting

```python
# pandas plot + plt config
prices.plot()
plt.title('Randomly-generated Prices')
plt.legend(loc=0);

# Plot a histogram using 20 bins
R = data['MSFT'].pct_change()[1:]
plt.hist(R, bins=20, cumulative=True)

# line 
plt.plot(data['MSFT'])
plt.plot(data['AAPL'])
# scatter
plt.scatter(data['MSFT'], data['AAPL'])
```

## measure of central tendency

```python
import scipy.stats as stats

# mode
stats.mode(prices)[0][0]
# Mode of bins
returns = pricing.pct_change()[1:]
hist, bins = np.histogram(returns, 20)
maxfreq = max(hist)
[(bins[i], bins[i+1]) for i, j in enumerate(hist) if j == maxfreq]
# geometric / harmonic
R_G = stats.gmean(prices)
init_price*R_G**T
stats.hmean(prices)
# skewness
if mode < median < mean:
    print  'The returns are positivly skewed.'
if mean < median < mode: 
    print 'The returns are negativly skewed.'
if mean == median == mode:
    print 'There is no Skewness
```

## measure of dispersion : volatility

* high volatile/variable , low accruate prediction

### range 

```python
# range : min - max
np.ptp(X)
```



### mean absolute devation

```python
abs_dispersion = [np.abs(mu - x) for x in X]
MAD = np.sum(abs_dispersion)/len(abs_dispersion)			
```



### Standard deviation 

- getting proportion of samples within k-std of mean : using Chebyshev's inequality
- getting portfolio variance(volatility)
- volatility interpretation 
  - if standard deviation > a quarter of the range, then the data is extremely volatile
  - less volatility : in the out-of-sample periods, the ratios of all window lengths stay mainly within 1 standard deviation of the mean.

```python
k = 1.25
dist = k*np.std(X)
l = [x for x in X if abs(x - mu) <= dist]
'Observations within {k} stds of mean: {l}'
'Confirm : {float(len(l))/len(X)} > {1 - 1/k**2}

# semi-standard deviation
B = 19 # <- mean or target value
lows_B = [e for e in X if e <= B]
semivar_B = sum(map(lambda x: (x - B)**2,lows_B))/len(lows_B) # list - float
semivar = np.sum( (lows - mu) ** 2 ) / len(lows) # list - list(numpy float64)

# portfolio variance
prices.rolling(window = 30).var()
cov_mat = np.cov(asset1, asset2)
cov = cov_mat[0,1]
v1, v2 = cov_mat[0,0], cov_mat[1,1]
w1 = 0.87; w2 = 1 - w1

pvariance = (w1**2)*v1+(w2**2)*v2+(2*w1*w2)*cov
```

## Statistical moments  : normality test

> Normality makes diverse statistical analysis, and provides a lot of information

### Skewness

- positive skewed (>0)
  - positive values > negative values 
    : right fat tail
    : mean > median > mode
- negative skewed (<0)
  - vice versa

```python
# lognormal dist
skew = stats.skew(returns)

if skew > 0:
    print 'The distribution is positively skewed'
elif skew < 0:
    print 'The distribution is negatively skewed'
else:
    print 'The distribution is symmetric'
    
print 'The returns of NFLX have a strong positive skew, meaning their volatility is characterized by frequent small changes in price with interspersed large upticks.'

# rolling skew
rolling_skew = AMC.rolling(window=60,center=False).skew()
plt.plot(rolling_skew)
plt.xlabel('Day')
plt.ylabel('60-day Rolling Skew')

print "The skew is too volatile to use it to make predictions outside of the sample."

```

### kurtosis

- excess kurtosis = kurtosis - 3 (embedded default?)

```python
# highly peaked and fat tail
# laplace > normal > cosine
stats.kurtosis(returns)

# leptokurtic dist : more frequent mean (k>3)
stats.laplace.stats(moments='k')
# platykurtic dist : fewer frequent mean (k<3)
stats.cosine.stats(moments='k')

'Because the excess kurtosis is negative, Y is platykurtic. Platykurtic distributions cluster around the mean, so large values in either direction are less likely'

'The historical returns of NFLX are strongly leptokurtic. Because of a leptokurtic distribution`s fatter tails, small changes in prices happen less often and large changes are more common. This makes the stock a riskier investment.'
```

### Jarque bera test 

- compares whether sample data has skewness and kurtosis similar to a normal distribution. 
- null hypothesis : sample data came from a normal distribution. 

```python
import statsmodels.stats.stattools import jarque_bera
_, pvalue, _, _ = jarque_bera(returns)

if pvalue > 0.05:
    print 'The returns are likely normal.'
else:
    print 'The returns are likely not normal.'
    
# treat p-values as binary and not try to read into them or compare them. 
```

### instability of parameter estimates

You Never Know, You Only Estimate

- the way of computing instability(=volatility=variability) :
  divided subsets -> estimated parameters from each one -> variability among the results.

- **a simple estimate** 
  - should not be used alone as an estimator
  - is not good for non-normal distribution (of price, return, etc)
    - lost of a lot of information : **useless** mean, std, skewness, kurtosis 
- **alternative estimates** for preidicting future
  - Sharpe ratio : the additional return per unit additional risk achieved by a portfolio, relative to a risk-free return
  - Moving average w/ Bollinger band : averages and standard deviations with a lookback window

> Tips : not always get better estimates by taking more data

```python
def sharpe_ratio(asset, riskfree):
    return np.mean(asset - riskfree)/np.std(asset - riskfree)

# Compute the running Sharpe ratio
running_sharpe = [sharpe_ratio(returns[i-90:i], treasury_ret[i-90:i]) for i in range(90, len(returns))]

# Moving average and Bollinger bands
mu = pd.rolling_mean(pricing, window=90)
mu = pd.rolling_std(pricing, window=90)
ax3.plot(mu)
ax3.plot(mu + std)
ax3.plot(mu - std);
```



## correlation : portfolio

- Determining related assets 
- Constructing a portfolio of uncorrelated assets : Uncorrelated assets produce the best portfolios

```python
np.corrcoef(a1,a2)[0,1]

# find_most_correlated pair assets
def find_most_correlated(data):
    n = data.shape[1]
    keys = data.keys()
    pair = []
    max_value = 0
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = np.corrcoef(S1, S2)[0,1]
            if result > max_value:
                pair = (keys[i], keys[j])
                max_value = result
    return pair, max_value

# keep correlated relationship in future ?
rolling_correlation = pd.rolling_corr(a1, a2, 60)
plt.plot(rolling_correlation)
plt.xlabel('Day')
plt.ylabel('60-day Rolling Correlation')
```

- limitation : non-linear (lagged, logged-change rate ) , sensitive to outlier

> Tips : strong correlation does not hold outside 

## Spearman rank correlation

- Purpose : Determining whether or not two data series move together / monotonic relationship
- Advantage 
  - when data sets may be in different units
  - when observation is not normally distributed ; weird distributions
  - outliers
- Comparison with pearson correlation
  - Spearman rank correlation copes with the non-linear relationship much better at most levels of noise. 
  - it is likely to do worse than regular correlation at very high levels of noise.
  - both are not good to auto-correlated dataset (lagged data)

```python
s_rank_coef , pvalue = stats.spearmanr(expense, sharpe)
if pvalue < 0.05:
  print(f'{} correlated'.format(['positively','negatively'][sign(s_rank_coef)/2]))
  
plt.scatter(expense, sharpe) #  weird clustering not suitable to correlation 
```

> Tips
>
> scatter plot is sometimes more accurate than a statistical number

## Random variable

- Definition : an object which does not have fixed value, but accidentally determined value
  - Determined by the distribution of possible values
  - Determined by probability of  a value(or values) among its distribution
- Usage : modeling patterns
  - Deterministic pattern
  - Random pattern : set pattern to random variable 
    - Sample value from variable at each timestamp
    - modeling the motion in the financial instrument

- Types : same principle but different handling

  |            | probability distribution     | Example                                      |
  | ---------- | ---------------------------- | -------------------------------------------- |
  | Discrete   | Probability mass function    | Binomial Model of Stock Price Movement, CAPM |
  | Continuous | Probability density function | Black-Schole                                 |

  

```python
class UniformRandomVariable:
    def __init__(self, a=0, b=1):
        self.variableType = "uniform"
        self.low = a
        self.high = b
        return
    def draw(self, numberOfSamples):
        samples = np.random.random_integers(self.low, self.high, numberOfSamples)
        return samples
      
 class BinomialRandomVaraible(UniformRandomVariable) : 
    def __init__(self, numberOfTrial = 10 , probabilityOfSuccess = .5):
        self.variableType = "binomial"
        self.numberOfTrial = numberOfTrial
        self.probabilityOfSuccess = probabilityOfSuccess
        return
    def draw(self, numberOfSamples):
        samples = np.random.binomial(self.numberOfTrial,
                                     self.probabilityOfSuccess, numberOfSamples)
        return samples

class ContinuousRandomVariable:
    def __init__(self, a = 0, b = 1):
        self.variableType = ""
        self.low = a
        self.high = b
        return
    def draw(self, numberOfSamples):
        samples = np.random.uniform(self.low, self.high, numberOfSamples)
        return samples
      
class NormalRandomVariable(ContinuousRandomVariable):
    def __init__(self, mean = 0, variance = 1):
        ContinuousRandomVariable.__init__(self)
        self.variableType = "Normal"
        self.mean = mean
        self.standardDeviation = np.sqrt(variance)
        return
    def draw(self, numberOfSamples):
        samples = np.random.normal(self.mean, self.standardDeviation, numberOfSamples)
        return samples
      
 # finding the 1st, 2nd, and third confidence intervals. 
first_ci = (-sigma, sigma)
second_ci = (-2*sigma, 2*sigma)
third_ci = (-3*sigma, 3*sigma)

  
```

## Linear Regression

1. measure of covariance : linear dependence, multidimensional model
2. Use statistical tests and not visualizations to verify your results.

```python
import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math

def linreg(X,Y):
    # Running the linear regression
    X = sm.add_constant(X) # set independent variable
    model = regression.linear_model.OLS(Y, X).fit()
    a = model.params[0] # intercept
    b = model.params[1] # 1st coefficient
    X = X[:, 1]
    # Return summary of the regression and plot results
    X2 = np.linspace(X.min(), X.max(), 100)
    Y_hat = X2 * b + a
    # plot regression 
    plt.scatter(X, Y, alpha=0.3) # Plot the raw data
    plt.plot(X2, Y_hat, 'r', alpha=0.9);  # Add the regression line, colored in red
    return model.summary()

# We have to take the percent changes to get to returns
# Get rid of the first (0th) element because it is NAN
r_a = asset.pct_change()[1:]
r_b = benchmark.pct_change()[1:]

linreg(r_b.values, r_a.values)
```

3. Pay attention to the **standard error** of the parameter estimates. 

- standard error : how stable your parameter estimates 
  - using a rolling window of data 
  - see how much variance there is in the estimates.

4. **Assumptions** for regression model :

- The independent variable is not random(?)
- The variance of the error term is constant across observations. for evaluating the goodness of the fit.
  --> different population! different model !
- The errors are not autocorrelated. If the Durbin-Watson statistic is close to 2, no autocorrelation.
- The errors are normally distributed. If not hold, useless statistics, such as the F-test.
  --> different population ! different model !

[+ Multiple Linear Regression assumption]

- There is no exact linear relationship between the independent variables. 
  = regression can not be expressed in multiple ways

5. Stepwise regression : best model selection for multiple linear regression

- Forward stepwise regression , Backward stepwise regression, stepwise regression
  - starts from an empty model
  - tests each individual variable, usually measured with AIC or BIC (lowest is best). 
  - selects the one that results in the best model quality, 
  - adds the remaining variables one at a time, 
  - repeat until getting unchanged best  the AIC or BIC value at each step. 

> Tips 
>
> - violation of assumptions makes metrics useless
> - R square : the fraction explained by model
> - adjusted R square  : the fraction explained by model (for multiple linear regression)
> - AIC : estimate of KL divergence = predictied dist - true dist.
> - BIC : adjusted AIC to sample size and 
> - Durbin watson : statistics to check autocorrelation
>   0~2 ) positive auto-correlation
>   2 ) no auto correlation
>   2~4 ) negative auto-correlation



5. All estimators is using MLE

```python
import scipy
import scipy.stats
from statsmodels.stats.stattools import jarque_bera
# normality test
_, pval, skew, kurt = jarque_bera(returns)
stat, pval = scipy.stats.mstats.normaltest(returns)
# get mle if normal
mu, std = scipy.stats.norm.fit(returns)
# get mle if exponential 
_, l = scipy.stats.expon.fit(returns, floc=0)

```

### Multiple variate model

- model separates out the contributions of different effects. 
- the S&P 500 (benchmark) : a more reliable predictor of both securities than they were of each other. 
  - gauge the **significance** between the two securities 
  - prevent **confounding** the two variables.

```python
# Run a linear regression on the two assets : ( asset1 ~ asset2 )
slr = regression.linear_model.OLS(asset1, sm.add_constant(asset2)).fit()
print 'SLR beta of asset2:', slr.params[1]

# Run multiple linear regression using asset2 and benchmark as independent variables
mlr = regression.linear_model.OLS(asset1, sm.add_constant(np.column_stack((asset2, benchmark)))).fit()

prediction = mlr.params[0] + mlr.params[1]*asset2 + mlr.params[2]*benchmark
prediction.name = 'Prediction'
```

### Model Instability

- "stability" is to evaluate the accuracy of the model with respect to our sample data.

  : not by how well it explains the dependent variable, but by how stable the regression coefficients are 

- How to analyze instability

  - Biased noise : **residual analysis** 
  - Outlier : detect outliers 
  - Regime change : skim trend breakpoint

```python
# Manually set the point where we think a structural break occurs
breakpoint = 1200
xs = np.arange(len(pricing))
xs2 = np.arange(breakpoint)
xs3 = np.arange(len(pricing) - breakpoint)

# Perform linear regressions 
# on the full data set, the data up to the breakpoint, and the data after
a, b = linreg(xs, pricing)
a2, b2 = linreg(xs2, pricing[:breakpoint])
a3, b3 = linreg(xs3, pricing[breakpoint:])

Y_hat = pd.Series(xs * b + a, index=pricing.index)
Y_hat2 = pd.Series(xs2 * b2 + a2, index=pricing.index[:breakpoint])
Y_hat3 = pd.Series(xs3 * b3 + a3, index=pricing.index[breakpoint:])
```

### Model assumption check : residual analysis and misspecification

#### 3 points to look into

- unbiased (expected value over different samples is the true value)
- consistent (converging to the true value with many samples; stability)

```python
# choose the independent variables which you have reason to believe will be good predictors 
# prediction performance should hold up out of sample - exclude unnecessary vars
```

- efficient (minimized variance)

#### Method

- visualization method 

```python
plt.scatter(model.predict(), residuals);
plt.axhline(0, color='red')
plt.xlabel('TSLA Returns');
plt.ylabel('Residuals');
```

- statistical testing method

#### Recapping assumptions and the way to check 

1. The independent variable is not random(?)

2. The variance of the error term is constant across observations. for evaluating the goodness of the fit.

- Breush-Pagan Lagrange Multiplier test for heteroscedasticity : the residual variance does not depend on the variables in x in the form `sigma_i = sigma * f(alpha_0 + alpha z_i)`

```python
# Testing for Heteroskedasticity
# xs_with_constant = model.model.exog
lagrange_multiplier_stat, pvalue, fstat, fstat_pvalue = stats.diagnostic.het_breushpagan(residuals1, xs_with_constant)
if pvalue < 0.05:
    print "The relationship is heteroscedastic."
```

- Adjusting for Heteroskedasticity
	- Differences Analysis
	- log-transformation : *all the data is positive*
	- box-cox transformation ( including log-transformation ) : *all the data is positive*

```python
# Adjusting api
SLR_MODEL.get_robustcov_results().summary()

# Finding first-order differences in Y_heteroscedastic
Y_heteroscedastic_diff = np.diff(Y_heteroscedastic)
model = sm.OLS(Y_heteroscedastic_diff, sm.add_constant(X[1:])).fit()

# Taking the log of Y_heteroscedastic 
Y_heteroscedastic_log = np.log(Y_heteroscedastic)
model = sm.OLS(Y_heteroscedastic_log, sm.add_constant(X)).fit()

# Finding a power transformation adjusted Y_heteroscedastic
Y_heteroscedastic_box_cox = stats.boxcox(Y_heteroscedastic)[0]
model = sm.OLS(Y_heteroscedastic_box_cox, sm.add_constant(X)).fit()

breusch_pagan_p = stats.diagnostic.het_breushpagan(residuals, model.model.exog)[1]
```

3. The errors are not autocorrelated. If the Durbin-Watson statistic is close to 2, no autocorrelation.

- Box-Pierce and Ljung-Box Tests : the null hypothesis of independence in a given time series

```python
from math import sqrt
# Testing for autocorrelation
_, prices_qstat_pvalues = stats.diagnostic.acorr_ljungbox(y,lags=10)
# another way
# qstat : returns the Ljung-Box q statistic for each autocorrelation coefficient
 _, prices_qstats, prices_qstat_pvalues = statsmodels.tsa.stattools.acf(y, qstat=True)

if any(prices_qstat_pvalues < 0.05):
    print "The residuals are autocorrelated."
```

- Adjusting for Auto-correlation 
  - Difference analysis : for modeling
  - GARCH : for modeling
  - Newey-west : for estimating Standard Error

```python
# Change the regression equation to eliminate serial correlation
# Find the covariance matrix of the coefficients
cov_mat = stats.sandwich_covariance.cov_hac(SLR_MODEL)
# the standard errors of each coefficient from the original model and from the adjustment
print 'Old standard errors:', SLR_MODEL.bse[0], SLR_MODEL.bse[1]
print 'Adjusted standard errors:', sqrt(cov_mat[0,0]), sqrt(cov_mat[1,1])
```

4. The errors are normally distributed. If not hold, useless statistics, such as the F-test.

- jarque_bera

```python
# Testing for Normality
_, pvalue, _, _ = statsmodels.stats.stattools.jarque_bera(residuals)
```

- Adjusting for Normality : same as Heteroskedaciticity

5. There is no exact linear relationship between the independent variables. 

```python
# Testing multicolinearity
# high R-squared, low t-statistics on the coefficients

# Correcting for multicolinearity
# removing one of predictors improves the t-statistics without hurting R-squared.
mlr = regression.linear_model.OLS(a, sm.add_constant(np.column_stack((b1,b2)))).fit()
slr = regression.linear_model.OLS(a, sm.add_constant(b1)).fit()
```

### Model overfitting

#### The reason why overfitting occurs

1. errors depends on independent variables : Heteroskedasticity , non-normality

2. incorrect functional form 
   - High-degree polynomials : more degrees of freedom 
   - [alternatives] select the form based on our expectation of the relationship : rate --> log-linear
3. Different populations, different models

4. confounding factor in look-like relationship between two variables

#### Must-be remembered 

1. real-world processes generally do not follow high-degree polynomial curves. 
   `be winner from tempting to use a quadratic or cubic model to decrease sample error.`

2. fewer parameters is better
   `explain 60% of the data with 2-3 parameters > 90% with 10`
3. Perfect fit = overfitting
4. The choice of **window length** strongly affects the rolling parameter estimate 

```python
# 1.overfitting to many predictors
slr = regression.linear_model.OLS(y, sm.add_constant(x1)).fit()
mlr = regression.linear_model.OLS(y, sm.add_constant(np.column_stack((x1,x2,x3)))).fit()
slr_prediction = slr.params[0] + slr.params[1]*x1
mlr_prediction = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3
# new data
slr_prediction_test = slr.params[0] + slr.params[1]*x1
mlr_prediction_test = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3
# accuracy
N = len(test_returns)
SST = sum((y - np.mean(y))**2)
adj1, adj2 = float(N - 1)/(N - 1 - 1),float(N - 1)/(N - 3 - 1)
SSR_slr,SSR_mlr = sum((slr_prediction2 - y)**2) , sum((mlr_prediction2 - y)**2)
print 'SLR R-squared:', 1 - adj1*SSRs/SST, ', MLR R-squared:', 1 - adj2*SSRm/SST


# 2.overfitting to window length(size)
length_scores = [ [trade(, l) for l in range(255)] for data in (price_1315,price_201618) ] 
best_window = [ np.argmax(i_length_scores) for i_length_scores in length_scores ] 
best_return = length_scores[0][best_window[0]],length_scores[1][best_window[1]]
print '2013-2015 best window: ',best_window[0],', 2016-2018 best window: ',best_window[1]
```



### Model details : Confidence 

#### Definition 

-  a range that will likely contain the population mean with C % sample cases
- Interpretation
  - `over computations of a 95% CI, the true value will be in the interval in 95% of the cases`
  - ` 95% chance that the true value of the mean was within that interval.`

- Purpose : to determine how **accurately** our sample mean **estimates** the population mean.
- Key component : standard error 

#### Characteristics 

- The higher our desired confidence, the larger range we report. 
- The bigger sample size, the shorter our intervals

### Assuming that 

- the way you sample is unbaised
- the data are **normal** and independent : **many datasets in finance are fundamentally non-normal**. 
  - CLT is not always safe
  - T- distribution (substitute)
- Miscalibration : Autocorrelated data

```python
# SE
# ddof = degrees of freedom 
SE= stats.sem(returns, ddof=0)

# lower/upper bound of CI
(l, u) = stats.norm.interval(0.95, loc=np.mean(returns), scale=SE)
(l, u) = stats.t.interval(0.95, df = n-1 , loc=np.mean(returns), scale=SE)

# plot
freq_at_mean = 5
plt.plot([l, u], [freq_at_mean, freq_at_mean], '-', color='r', linewidth=4, label='CI')
plt.plot(np.mean(returns), freq_at_mean, 'o', color='r', markersize=10);
```

> Tips : scipy method
>
> | `pmf` | 확률질량함수(probability mass function)                      |
> | ----- | ------------------------------------------------------------ |
> | `pdf` | 확률밀도함수(probability density function)                   |
> | `cdf` | 누적분포함수(cumulative distribution function)               |
> | `ppf` | 누적분포함수의 역함수(inverse cumulative distribution function) |
> | `sf`  | 생존함수(survival function) = 1 - 누적분포함수               |
> | `isf` | 생존함수의 역함수(inverse survival function)                 |

### Model details : Hypothesis testing

#### The reason why this is necessary

: sample may (or may not) reflect the true state of the underlying process

#### How to Perform Hypothesis Testing:

1. State the hypothesis and the alternative to the hypothesis

2. Identify the appropriate test statistic and its distribution. 

   Ensure that any assumptions about the data are met (stationarity, normality, etc.)

   - The t-distribution (t-test)
   - The standard normal distribution (z-test)
   - The chi-square (χ2) distribution (χ2-test)
   - The F-distribution (F-test)

3. Specify the significance level, α

4. From α and the distribution compute the 'critical value'.

   - In a 'less than or equal to' hypothesis test, the p-value is 1−CDF(Test Statistic)
   - In a 'greater than or equal to' hypothesis test, the p-value is CDF(Test Statistic)
   - In a 'not equal to' hypothesis test, the p-value is 2∗1−CDF(|Test Statistic|)

5. Collect the data and calculate the test statistic

   Ensure issues of time-period bias, or of look-ahead bias 

6. Compare test statistic with critical value and decide whether to accept or reject the hypothesis.

#### Hypothesis on mean : comparing 

The t-distribution : has fatter tails and a lower peak, giving more flexibility 

- equal variance t-test

```python
# Sample mean values
mu_spy, mu_aapl = returns_sample.mean()
s_spy, s_aapl = returns_sample.std()
n_spy = len(returns_sample['SPY'])
n_aapl = len(returns_sample['AAPL'])

test_statistic = ((mu_spy - mu_aapl) - 0)/((s_spy**2/n_spy) + (s_aapl**2/n_aapl))**0.5
df = ((s_spy**2/n_spy) + (s_aapl**2/n_aapl))**2/(((s_spy**2 / n_spy)**2 /n_spy)+((s_aapl**2 / n_aapl)**2/n_aapl))

# scipy api
scipy.stats.ttest_ind(returns_sample['SPY'],returns_sample['AAPL'])
```

- welch's unequal variance t-test

```python
# scipy api
scipy.stats.ttest_ind(returns_sample['SPY'],returns_sample['AAPL'], equal_var=False)
```

#### Hypothesis on variance

χ2 distributions for single variance tests 

```python
alpha = 0.99
null_variance = 0.0001
test_statistic = (len(returns_sample) - 1) * returns_sample.std()**2 / null_variance
crit_value = chi2.ppf(alpha, len(returns_sample) - 1)
if test_statistic > crit_value :
  print('not significant')
```



#### Hypothesis on comparing variances

F distributions for comparisons of variance

```python
from scipy.stats import f
test_statistic = (aapl_std_dev / spy_std_dev)**2
df1,df2 = [len(returns_sample[stock]) - 1 for stock in ['AAPL','SPY'] ]
upper_crit_value = f.ppf(1-alpha/2, df1, df2)
lower_crit_value = f.ppf(alpha/2, df1, df2)
if not lower_crit_value < test_statistic < upper_crit_value:
  print('not significant')
```



## P-hacking and Multiple comparison bias

### p-value 

- is not probability of rejecing null hypothesis
- must be compared with the cutoff and treated as significant/not signficant
- the cutoff = Significance Level
- trade off : lower cutoff -> lower the rate of false positives, lower the chance of true positive
 ```python
if r_s_pvalue < cutoff:
      print 'There is significant evidence of a relationship.'
 ```

### How to avoid Multiple comparison bias

- Run fewer tests.

use common sense about whether there is sufficient evidence that a hypothesis is true. This process of exploring the data, coming up with a hypothesis, then gathering more data and testing the hypothesis on the new data is considered the gold standard in statistical and scientific research. 

- Use out of sample testing

It's crucial that the data set on which you develop your hypothesis is not the one on which you test it.

### Bonferroni correction

If you must run many tests, try to correct your p-values

```python
new_cutoff = significance_level / num_tests
```