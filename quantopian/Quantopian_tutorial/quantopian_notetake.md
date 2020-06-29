## Quantopian 



### Basic knowledge

#### Trading algorithms structure

1. Compute N scalar values for the asset based on a trailing window.
2. Select a set of assets, based on computed values  (1).
3. Calculate portfolio weights, on selected set in (2).
4. Place orders to portfolio allocations by the desired weights (3).

#### Technical challenges 

-  efficiently **querying** large sets of assets
-  performing **computations** on large sets of assets
-  handling **adjustments** (splits and dividends)
-  asset **delistings**

#### Terms  

- alpha factors : express a predictive relationship between some given set of information and future returns.

- alpha score : factor score

- IC : number for quantifying a given alpha factor's predictiveness.

- turnover : ?

- reference to object

- Cross-sectional analysis : compares  its industry peers.

  head-to-head analysis with its biggest competitors 
  industry-wide lens to identify companies with a particular strength

#### The context of the quant workflow

1. Define your trading universe and build an alpha factor 
2. Analyze the predictiveness of your alpha factor 
3. Create a trading strategy based on your alpha factor 
4. Backtest your trading strategy and analyze the results 

### Main API

#### **Pipeline** 

: data extraction and manipulation tool

1. Providing data 

- pricing data 
- sentiment data : PsychSignal , sentdex
- Fundamental data : Morningstar 

2. Transferring data between Research and the IDE

- saving a reference(data)  into  columns

3. Built-in calculations

4. Universe selection : set of assets ( assets in portfolio )

- ex : liquid and easy to trade

- Research : utility functions 
  - pricing, volume, and returns data ( 8000+ US equities, 2002~)

```python
# Execute pipeline over evaluation period
from quantopian.pipeline import Pipeline
from quantopian.pipeline import factors,data,experimental
pipeline_output = run_pipeline(
    make_pipeline(),
    start_date=period_start,
    end_date=period_end
)
```

#### Research  

: `ONLY Notebook`  

- pricing, volume, and returns data ( 8000+ US equities, 2002~)
- run_pipeline ()

```python
# Research environment functions
from quantopian.research import prices, run_pipeline

# Query pricing data for all assets present during
# evaluation period
asset_prices = prices(
    asset_list,
    start=period_start,
    end=period_end
)
```

#### Alphalens 

: `ONLY Notebook`

using for testing the quality of selection strategy

- Combine factor and prices
- Classifies(generate) target based on score
- Compute forward returns
- Visualize data
- ...transaction costs or market impact

```python
# Get asset forward returns and quantile classification
# based on sentiment scores
factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor=pipeline_output['factor_score'],
    prices=asset_prices,
    quantiles=2, # binary classification
    periods=(1,5,10), # forward returns to holding days
)
```

#### **Alogorithm** 
: build , attach and analyze(backtest) algorithm

- Process data streams 
- Generate an output

```python 
def initialize(context):
  # Attach pipeline to algorithm
  algo.attach_pipeline(
        make_pipeline(),
        'data_pipe'
    )
  # Schedule rebalance function
  schedule_function(
      rebalance,
      date_rule=date_rules.week_start(),
      time_rule=time_rules.market_open()
  )         
def before_trading_start(context,data):
  # Get pipeline output and store it in context
  log.info()
def rebalance(context,data):
  log.info()
```

#### Optimize

- Portfolio optimization  : Rebalance portfolio
  - Create objective function result
  - Set consstraint parameters : leverage, pos_size, turnover
    - turnover : net sale?

```python
import quantopian.optimize as opt
# Create MaximizeAlpha objective
      objective = opt.MaximizeAlpha(alpha)
  
# Rebalance portfolio using objective
# and list of constraints
      algo.order_optimal_portfolio(
          objective=objective,
          constraints=[
            constrain_pos_size,
            max_leverage,
            dollar_neutral,
            max_turnover,
            # factor_risk_constraints,
          ]
      )
```

#### Backtest 

- Pyfolio 
  - visualization for algorithm's behavior, risk exposures over time
- Risk Model
  : Manage our portfolio's exposure to common risk factors
  - calculate asset exposure to 16 risk factors

```python
# Constrain target portfolio's risk exposure
# By default, max sector exposure is set at 0.2,
#  and max style exposure is set at 0.4
factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            version=opt.Newest
        )
```

### Pipeline API 

#### Basic usage 

- Defining pipeline : Pipeline API
  - Importing Data
    - Datasets , Boundcolumns : types, currency
    - DatasetFamily
    - CustomData
  - Defining Computation
    - Factors : built-in, custom, combining, slicing
    - Filters : built-in, comparison operator, built-in methods (Factor/Classifier), combining, masking, custom
    - Classifiers : attribute(Boundcolumns), built-in methods(Factor)
  - Instantiating a Pipeline
    - Columns
    - Domain : time, assets
    - Screen
- Running pipeline in Research API
  - Access data 
    - pipeline
    - special functions 
      - price data : adjust_price
      - risk data
  - Create alpha factors
- Having used pipeline in Algorithm API

#### Types of computations

1. Factors : assets & moment --> numeric (float)
   -  computing target weights
   -  generating alpha signal
   -  constructing other, more complex factors
   -  constructing filters

2. Filters : assets & moment --> bool

3. Classifiers : assets & moment --> categoric (string, int)

> additional utils
>
> - Masking, Combining (Screening ) 
>   : return condition-fulfilled stocks
>   : not excluding ! only including 
> - Customizing** (example below)

```python
class StdDev(CustomFactor):
    def compute(self, today, asset_ids, out, values):
# Calculates the column-wise standard deviation, ignoring NaNs
        out[:] = numpy.nanstd(values, axis=0)
        
class TenDayMeanDifference(CustomFactor):
# Default inputs.
    inputs = [USEquityPricing.close, USEquityPricing.open]
    window_length = 10
    def compute(self, today, asset_ids, out, close, open):
# Calculates the column-wise mean difference,no NaNs
    out[:] = numpy.nanmean(close - open, axis=0)
```

#### Data 

1. DataSets : simply collections of objects 
   - tell the Pipeline where and how to find the inputs to computations. 
     Ex. USEquityPricing.
2. BoundColumn : a column of data that is concretely bound to a DataSet. 
   - **create instance** right upon access to attributes of DataSets. 
   - **must-be Input type** to pipeline computations 
     Ex. USEquityPricing.close.

> tips
>
> * DataSets and BoundColumns : not hold actual data. 
> * Computations don't perform the computation until the pipeline is run. 
> * DataSet and BoundColumns identify the inputs of a computation. 
> * The data is populated later when the pipeline is run.

#### Example

Mean reversion strategy. 

look at the 10-day and 30-day moving averages (close price). 

Let's plan to open

- equally weighted long positions in the 75 securities with the least (most negative) percent difference 
- equally weighted short positions in the 75 with the greatest percent difference. 

```python
def make_pipeline():

    # Base universe filter.
    base_universe = QTradableStocksUS()

    # 10-day close price average.
    mean_10 = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=10,
        mask=base_universe
    )

    # 30-day close price average.
    mean_30 = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=30,
        mask=base_universe
    )

    # Percent difference factor.
    percent_difference = (mean_10 - mean_30) / mean_30

    # Create a filter to select securities to short.
    shorts = percent_difference.top(75)

    # Create a filter to select securities to long.
    longs = percent_difference.bottom(75)

    # Filter for the securities that we want to trade.
    securities_to_trade = (shorts | longs)

    return Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts
        },
        screen=securities_to_trade
    )

```



### Alphalens API

analyz a given alpha factor's effectiveness at predicting future returns.

- Define Alpha Factor on Pipeline API

1. Query Pricing Data

2. Align Data - [get_clean_factor_and_forward_returns()](https://quantopian.github.io/alphalens/alphalens.html#alphalens.utils.get_clean_factor_and_forward_returns) 

3. Visualize Results - create_full_tear_sheet

   ```python
   from alphalens.utils import get_clean_factor_and_forward_returns
   from alphalens.tears import create_full_tear_sheet,create_information_tear_sheet,create_returns_tear_sheet
   
   merged_data = get_clean_factor_and_forward_returns(
     factor=factor_data, 
     prices=pricing_data
   )
   
   create_full_tear_sheet(merged_data)
   ```

### Algorithm API

#### [Initializing an algorithm](https://www.quantopian.com/docs/user-guide/tools/algo-api#initialize)

- Initialize state
- schedule functions
- attach/register a [Pipeline](https://www.quantopian.com/docs/user-guide/tools/pipeline#pipeline-tool).
- setting slippage and commissions : built-in, custom model

#### [Performing computations on data](https://www.quantopian.com/docs/user-guide/tools/algo-api#algorithm-data)

- Import the data : barData lookup, data from pipeline
- perform any necessary computations with the data.

#### [Rebalancing a portfolio of assets](https://www.quantopian.com/docs/user-guide/tools/algo-api#rebalance)

- Buy/sell assets based on your imported data/computations.

#### [Logging and plotting](https://www.quantopian.com/docs/user-guide/tools/algo-api#algorithm-logging)

- Log and plot bookkeeping variables for further analysis.

```python
  
  def initialize(context):
    # Schedule our rebalance function to run 
    # at the start of each week, when the market opens.
    algo.schedule_function(
        my_rebalance,
        algo.date_rules.week_start(),
        algo.time_rules.market_open(),
        algo.calendars.US_EQUITIES
    )

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    
  def compute_target_weights(context, data):
    """
    Compute ordering weights.
    """
    
   def before_trading_start(context, data):
      """
      Get pipeline results.
      """
      
    def my_rebalance(context, data):
      """
      Rebalance weekly.
      """
```

### Optimize - Optimize API

#### Setting objectives

1. [`TargetWeights`](https://www.quantopian.com/docs/api-reference/optimize-api-reference#quantopian.optimize.TargetWeights): target portfolios that identify a list of target assets ; equal weighted or factor weighted (Euclidean Distance) to the targets.
2. [`MaximizeAlpha`](https://www.quantopian.com/docs/api-reference/optimize-api-reference#quantopian.optimize.MaximizeAlpha): predict expected returns; "alpha" values for each asset, new portfolio weights that maximizes the sum of each asset's weight times its alpha value.

#### Setting constraints

1. [`MaxGrossExposure`](https://www.quantopian.com/docs/api-reference/optimize-api-reference#quantopian.optimize.MaxGrossExposure) : a portfolio's gross exposure (i.e., the sum of the absolute value of the portfolio's positions) to be less than a percentage of the current portfolio value.
2. [`NetExposure`](https://www.quantopian.com/docs/api-reference/optimize-api-reference#quantopian.optimize.NetExposure) : a portfolio's net exposure (i.e. the value of the portfolio's longs minus the value of its shorts) to fall between two percentages of the current portfolio value.
3. [`PositionConcentration`](https://www.quantopian.com/docs/api-reference/optimize-api-reference#quantopian.optimize.PositionConcentration) : a portfolio's exposure to each individual asset in the portfolio.
4. [`NetGroupExposure`](https://www.quantopian.com/docs/api-reference/optimize-api-reference#quantopian.optimize.NetGroupExposure) : a portfolio's net exposure to a set of market sub-groups (e.g. sectors or industries).
5. [`FactorExposure`](https://www.quantopian.com/docs/api-reference/optimize-api-reference#quantopian.optimize.FactorExposure) : a portfolio's net weighted exposure to a set of **Risk Factors**.

```python
import quantopian.optimize as opt

objective = opt.MaximizeAlpha(expected_returns)
constraints = [
    opt.MaxGrossExposure(W_max),
    opt.PositionConcentration(min_weights, max_weights),
]
optimal_weights = opt.calculate_optimal_portfolio(objective, constraints)
```

### Optimize - Risk model

> Types of Risk factors
>
> - 11 sector risk : 
> - 5 style risk 
>   - momentum : upswing - downswing
>   - size : big cap - small cap
>   - value : expensive - cheap
>   - short term reversal : reversal winner - reversal loser
>   - volatility : deviation

1. (a constraint parameter to) portfolio construction : 

```python
from quantopian.pipeline.experimental import risk_loading_pipeline

def initialize(context):
    attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')

def before_trading_start(context, data):
    context.risk_loading_pipeline = pipeline_output('risk_loading_pipeline')
    # Multiple pipelines are supported in an algorithm.
    context.my_other_pipeline = pipeline_output('my_other_pipeline')
    
def place_orders(context, data):  
# Constrain our risk exposures. version 0 : default bounds
# which constrain our portfolio to 18% exposure to each sector and 36% to each style factor.
    constrain_sector_style_risk = opt.experimental.RiskModelExposure(  
        risk_model_loadings=context.risk_loading_pipeline,  
        version=0,
    )
  # Supply the constraint to order_optimal_portfolio.
    algo.order_optimal_portfolio(  
        objective=my_objective,  
        constraints=[constrain_sector_style_risk],  
    )
```

2. performance analysis

- performance attribution on a backtest result
- incorporated in pyfolio tearsheets



### Backtest - Pyfolio

1. help analyzes a backtest (pyfolio tear.sheet)

2. provides a wealth of performance statistics (metrics)

- annual/monthly returns, return quantiles
- rolling beta/Sharpe ratios, portfolio turnover

> How to use it
>
> Importing backtested algorithm result
>
> running (pyfolio) tearsheets



#### Custom algorithm Example

neither the long investments nor the short investments must exceed 55% of that investment.