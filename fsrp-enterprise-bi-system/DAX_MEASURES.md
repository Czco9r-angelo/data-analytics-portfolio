# Sample DAX Measures for Power BI Dashboard
## Enterprise Financial BI System

**Note:** All DAX measures use generic table/column names. No confidential data or actual field names included.

---

## 📋 Table of Contents

1. [Expenditure Measures](#expenditure-measures)
2. [Budget Performance Measures](#budget-performance-measures)
3. [Supplier Analysis Measures](#supplier-analysis-measures)
4. [Time Intelligence Measures](#time-intelligence-measures)
5. [Data Quality Measures](#data-quality-measures)
6. [Advanced Calculations](#advanced-calculations)

---

## 1. Expenditure Measures

### Core Spending Metrics

```DAX
// Total Spend
Total Spend = SUM(Transactions[AmountUSD])

// Total Transactions
Total Transactions = COUNTROWS(Transactions)

// Average Transaction Size
Avg Transaction = 
DIVIDE(
    [Total Spend],
    [Total Transactions],
    0
)

// Median Transaction Size
Median Transaction = 
MEDIAN(Transactions[AmountUSD])

// Transaction Size Distribution
Small Transactions = 
CALCULATE(
    [Total Transactions],
    Transactions[AmountUSD] < 1000
)

Medium Transactions = 
CALCULATE(
    [Total Transactions],
    Transactions[AmountUSD] >= 1000 && Transactions[AmountUSD] < 10000
)

Large Transactions = 
CALCULATE(
    [Total Transactions],
    Transactions[AmountUSD] >= 10000
)
```

### Category Analysis

```DAX
// Spend by Category
Spend by Category = 
CALCULATE(
    [Total Spend],
    ALLEXCEPT(Transactions, Transactions[Category])
)

// Top Category
Top Category = 
CALCULATE(
    FIRSTNONBLANK(Transactions[Category], 1),
    TOPN(1, 
        VALUES(Transactions[Category]),
        [Total Spend],
        DESC
    )
)

// Category Percentage of Total
Category % = 
DIVIDE(
    [Total Spend],
    CALCULATE(
        [Total Spend],
        ALL(Transactions[Category])
    ),
    0
)
```

---

## 2. Budget Performance Measures

### Budget Tracking

```DAX
// Total Budget
Total Budget = SUM(BudgetTable[BudgetAmount])

// Total Actual Spend
Total Actual = SUM(Transactions[AmountUSD])

// Budget Variance (Absolute)
Budget Variance = [Total Actual] - [Total Budget]

// Budget Variance (Percentage)
Budget Variance % = 
DIVIDE(
    [Budget Variance],
    [Total Budget],
    0
)

// Budget Utilization Rate
Budget Utilization = 
DIVIDE(
    [Total Actual],
    [Total Budget],
    0
)

// Budget Remaining
Budget Remaining = [Total Budget] - [Total Actual]

// Budget Status
Budget Status = 
SWITCH(
    TRUE(),
    [Budget Utilization] > 1.1, "Critical Over Budget",
    [Budget Utilization] > 1.0, "Over Budget",
    [Budget Utilization] > 0.9, "Warning - Near Limit",
    [Budget Utilization] > 0.75, "On Track",
    "Under Utilized"
)
```

### Budget Analysis by Component

```DAX
// Components Over Budget
Components Over Budget = 
CALCULATE(
    DISTINCTCOUNT(BudgetTable[ComponentID]),
    FILTER(
        VALUES(BudgetTable[ComponentID]),
        [Budget Utilization] > 1.0
    )
)

// Components Under Budget
Components Under Budget = 
CALCULATE(
    DISTINCTCOUNT(BudgetTable[ComponentID]),
    FILTER(
        VALUES(BudgetTable[ComponentID]),
        [Budget Utilization] < 1.0
    )
)

// Total Over Budget Amount
Total Over Budget = 
CALCULATE(
    [Total Actual] - [Total Budget],
    FILTER(
        VALUES(BudgetTable[ComponentID]),
        [Total Actual] > [Total Budget]
    )
)

// Projected Year-End Spend
Projected Year End = 
VAR MonthsElapsed = 
    DATEDIFF(
        MIN(Transactions[Date]),
        MAX(Transactions[Date]),
        MONTH
    )
VAR MonthsInYear = 12
VAR CurrentRunRate = 
    DIVIDE([Total Actual], MonthsElapsed, 0)
RETURN
    CurrentRunRate * MonthsInYear

// Projected Over/Under Budget
Projected Budget Variance = 
[Projected Year End] - [Total Budget]
```

---

## 3. Supplier Analysis Measures

### Supplier Metrics

```DAX
// Total Unique Suppliers
Total Suppliers = DISTINCTCOUNT(Transactions[SupplierID])

// Supplier Spend
Supplier Spend = 
CALCULATE(
    [Total Spend],
    ALLEXCEPT(Transactions, Transactions[SupplierID])
)

// Supplier Transaction Count
Supplier Transactions = 
CALCULATE(
    [Total Transactions],
    ALLEXCEPT(Transactions, Transactions[SupplierID])
)

// Average Spend per Supplier
Avg Spend per Supplier = 
DIVIDE(
    [Total Spend],
    [Total Suppliers],
    0
)

// Average Transaction per Supplier
Avg Transaction per Supplier = 
DIVIDE(
    [Supplier Spend],
    [Supplier Transactions],
    0
)
```

### Supplier Concentration Risk

```DAX
// Top Supplier
Top Supplier = 
CALCULATE(
    FIRSTNONBLANK(Transactions[SupplierName], 1),
    TOPN(1,
        VALUES(Transactions[SupplierID]),
        [Supplier Spend],
        DESC
    )
)

// Top Supplier Amount
Top Supplier Amount = 
CALCULATE(
    [Supplier Spend],
    TOPN(1,
        VALUES(Transactions[SupplierID]),
        [Supplier Spend],
        DESC
    )
)

// Top 5 Supplier Concentration
Top 5 Concentration = 
VAR Top5Spend = 
    CALCULATE(
        [Supplier Spend],
        TOPN(5,
            VALUES(Transactions[SupplierID]),
            [Supplier Spend],
            DESC
        )
    )
RETURN
    DIVIDE(Top5Spend, [Total Spend], 0)

// Top 10 Supplier Concentration
Top 10 Concentration = 
VAR Top10Spend = 
    CALCULATE(
        [Supplier Spend],
        TOPN(10,
            VALUES(Transactions[SupplierID]),
            [Supplier Spend],
            DESC
        )
    )
RETURN
    DIVIDE(Top10Spend, [Total Spend], 0)

// Supplier Risk Score
Supplier Risk Score = 
SWITCH(
    TRUE(),
    [Top 5 Concentration] > 0.8, "High Risk - Heavy Concentration",
    [Top 5 Concentration] > 0.6, "Medium Risk",
    [Top 5 Concentration] > 0.4, "Low Risk",
    "Well Diversified"
)
```

---

## 4. Time Intelligence Measures

### Period-over-Period Analysis

```DAX
// Year-to-Date Spend
YTD Spend = 
TOTALYTD(
    [Total Spend],
    Transactions[Date]
)

// Quarter-to-Date Spend
QTD Spend = 
TOTALQTD(
    [Total Spend],
    Transactions[Date]
)

// Month-to-Date Spend
MTD Spend = 
TOTALMTD(
    [Total Spend],
    Transactions[Date]
)

// Previous Year Spend
Previous Year Spend = 
CALCULATE(
    [Total Spend],
    DATEADD(Transactions[Date], -1, YEAR)
)

// Year-over-Year Growth
YoY Growth = 
[Total Spend] - [Previous Year Spend]

// Year-over-Year Growth %
YoY Growth % = 
DIVIDE(
    [YoY Growth],
    [Previous Year Spend],
    0
)

// Previous Quarter Spend
Previous Quarter Spend = 
CALCULATE(
    [Total Spend],
    DATEADD(Transactions[Date], -1, QUARTER)
)

// Quarter-over-Quarter Growth
QoQ Growth = 
[Total Spend] - [Previous Quarter Spend]

// Quarter-over-Quarter Growth %
QoQ Growth % = 
DIVIDE(
    [QoQ Growth],
    [Previous Quarter Spend],
    0
)
```

### Moving Averages

```DAX
// 3-Month Moving Average
3M Moving Avg = 
VAR CurrentDate = MAX(Transactions[Date])
VAR Last3Months = 
    DATESINPERIOD(
        Transactions[Date],
        CurrentDate,
        -3,
        MONTH
    )
RETURN
    CALCULATE(
        AVERAGEX(
            VALUES(Transactions[Date]),
            [Total Spend]
        ),
        Last3Months
    )

// 6-Month Moving Average
6M Moving Avg = 
VAR CurrentDate = MAX(Transactions[Date])
VAR Last6Months = 
    DATESINPERIOD(
        Transactions[Date],
        CurrentDate,
        -6,
        MONTH
    )
RETURN
    CALCULATE(
        AVERAGEX(
            VALUES(Transactions[Date]),
            [Total Spend]
        ),
        Last6Months
    )

// Rolling 12-Month Total
Rolling 12M Total = 
VAR CurrentDate = MAX(Transactions[Date])
VAR Last12Months = 
    DATESINPERIOD(
        Transactions[Date],
        CurrentDate,
        -12,
        MONTH
    )
RETURN
    CALCULATE(
        [Total Spend],
        Last12Months
    )
```

---

## 5. Data Quality Measures

### Data Validation Metrics

```DAX
// Data Quality Score
Data Quality % = 
DIVIDE(
    CALCULATE(
        COUNTROWS(Transactions),
        Transactions[QualityFlag] = "OK"
    ),
    COUNTROWS(Transactions),
    0
)

// Records with Issues
Records Flagged = 
CALCULATE(
    COUNTROWS(Transactions),
    Transactions[QualityFlag] <> "OK"
)

// Clean Records
Clean Records = 
CALCULATE(
    COUNTROWS(Transactions),
    Transactions[QualityFlag] = "OK"
)

// Critical Issues
Critical Issues = 
CALCULATE(
    COUNTROWS(Transactions),
    Transactions[Priority] = 1
)

// High Priority Issues
High Priority Issues = 
CALCULATE(
    COUNTROWS(Transactions),
    Transactions[Priority] = 2
)

// Issues by Type
Missing Budget Codes = 
CALCULATE(
    COUNTROWS(Transactions),
    Transactions[QualityFlag] = "MISSING_BUDGET_CODE"
)

Invalid Codes = 
CALCULATE(
    COUNTROWS(Transactions),
    Transactions[QualityFlag] = "INVALID_CODE"
)

Missing Suppliers = 
CALCULATE(
    COUNTROWS(Transactions),
    Transactions[QualityFlag] = "MISSING_SUPPLIER"
)
```

### Completeness Metrics

```DAX
// Budget Code Completeness
Budget Code Completeness = 
DIVIDE(
    CALCULATE(
        COUNTROWS(Transactions),
        NOT(ISBLANK(Transactions[BudgetCode]))
    ),
    COUNTROWS(Transactions),
    0
)

// Supplier Completeness
Supplier Completeness = 
DIVIDE(
    CALCULATE(
        COUNTROWS(Transactions),
        NOT(ISBLANK(Transactions[SupplierID]))
    ),
    COUNTROWS(Transactions),
    0
)

// Description Completeness
Description Completeness = 
DIVIDE(
    CALCULATE(
        COUNTROWS(Transactions),
        NOT(ISBLANK(Transactions[Description]))
    ),
    COUNTROWS(Transactions),
    0
)
```

---

## 6. Advanced Calculations

### Statistical Analysis

```DAX
// Standard Deviation
Spend Std Dev = 
STDEV.P(Transactions[AmountUSD])

// Coefficient of Variation
Coefficient of Variation = 
DIVIDE(
    [Spend Std Dev],
    [Avg Transaction],
    0
)

// Outlier Detection (Z-Score > 3)
Outlier Count = 
VAR AvgAmount = [Avg Transaction]
VAR StdDev = [Spend Std Dev]
RETURN
    COUNTROWS(
        FILTER(
            Transactions,
            ABS(Transactions[AmountUSD] - AvgAmount) > (3 * StdDev)
        )
    )

// Percentile Calculations
25th Percentile = 
PERCENTILE.INC(Transactions[AmountUSD], 0.25)

50th Percentile = 
PERCENTILE.INC(Transactions[AmountUSD], 0.50)

75th Percentile = 
PERCENTILE.INC(Transactions[AmountUSD], 0.75)

90th Percentile = 
PERCENTILE.INC(Transactions[AmountUSD], 0.90)
```

### Ranking and Top N

```DAX
// Supplier Rank by Spend
Supplier Rank = 
RANKX(
    ALL(Transactions[SupplierID]),
    [Supplier Spend],
    ,
    DESC,
    DENSE
)

// Category Rank
Category Rank = 
RANKX(
    ALL(Transactions[Category]),
    [Total Spend],
    ,
    DESC,
    DENSE
)

// Is Top 10 Supplier
Is Top 10 Supplier = 
IF([Supplier Rank] <= 10, "Top 10", "Other")

// Cumulative Spend %
Cumulative Spend % = 
VAR CurrentSupplierRank = [Supplier Rank]
VAR TotalSpendAllSuppliers = 
    CALCULATE(
        [Total Spend],
        ALL(Transactions[SupplierID])
    )
VAR CumulativeSpend = 
    CALCULATE(
        [Total Spend],
        FILTER(
            ALL(Transactions[SupplierID]),
            [Supplier Rank] <= CurrentSupplierRank
        )
    )
RETURN
    DIVIDE(CumulativeSpend, TotalSpendAllSuppliers, 0)
```

### Conditional Formatting Helpers

```DAX
// Budget Status Color
Budget Color = 
SWITCH(
    TRUE(),
    [Budget Utilization] > 1.0, "Red",
    [Budget Utilization] > 0.9, "Yellow",
    "Green"
)

// Trend Indicator
Trend Indicator = 
SWITCH(
    TRUE(),
    [QoQ Growth %] > 0.1, "▲ Strong Growth",
    [QoQ Growth %] > 0, "↗ Growth",
    [QoQ Growth %] = 0, "→ Flat",
    [QoQ Growth %] > -0.1, "↘ Decline",
    "▼ Strong Decline"
)

// Performance KPI
Performance KPI = 
VAR Utilization = [Budget Utilization]
VAR Quality = [Data Quality %]
RETURN
    SWITCH(
        TRUE(),
        Utilization > 1.0 OR Quality < 0.9, 0,  // Bad
        Utilization > 0.9 AND Quality > 0.95, 2, // Great
        1  // OK
    )
```

---

## 📊 Measure Organization Best Practices

### Create Measure Groups

Organize measures into logical groups using display folders:

```
_Measures Table
│
├── 📁 Expenditure
│   ├── Total Spend
│   ├── Total Transactions
│   └── Avg Transaction
│
├── 📁 Budget
│   ├── Total Budget
│   ├── Budget Variance
│   └── Budget Utilization
│
├── 📁 Suppliers
│   ├── Total Suppliers
│   ├── Top Supplier
│   └── Supplier Concentration
│
├── 📁 Time Intelligence
│   ├── YTD Spend
│   ├── YoY Growth %
│   └── Moving Averages
│
└── 📁 Data Quality
    ├── Data Quality %
    ├── Records Flagged
    └── Completeness Metrics
```

### Formatting Standards

**Currency Measures:**
- Format: Currency ($)
- Decimal places: 0 or 2
- Symbol: $ (USD)

**Percentage Measures:**
- Format: Percentage
- Decimal places: 0, 1, or 2
- Display: 85% not 0.85

**Whole Numbers:**
- Format: Whole number
- Use 1000 separator: Yes

**Dates:**
- Format: Short Date or Long Date
- Locale: English (United States)

---

## 🔧 Performance Optimization Tips

### Efficient DAX Patterns

**DO:**
- ✅ Use variables (VAR) to store intermediate calculations
- ✅ Filter before calculating (CALCULATE with filters)
- ✅ Use DIVIDE instead of division operator (handles divide by zero)
- ✅ Use SELECTEDVALUE for single-value contexts
- ✅ Leverage time intelligence functions

**DON'T:**
- ❌ Nest CALCULATE functions unnecessarily
- ❌ Use complex iterators (SUMX, FILTER) when simple aggregations work
- ❌ Create calculated columns when measures suffice
- ❌ Use ALL() without understanding context
- ❌ Overuse RELATED in measures

---

## 📝 Notes

- All DAX formulas are generic examples
- Table and column names are placeholders
- No actual field names or values from production system
- Patterns demonstrate Power BI development capability

---

**Author:** Swithun M. Chiziko  
**Purpose:** Portfolio demonstration of Power BI/DAX development skills  
**Context:** Enterprise financial BI system dashboard development
