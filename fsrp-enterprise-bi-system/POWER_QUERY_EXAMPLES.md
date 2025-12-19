# Sample Power Query (M Language) Code Examples
## Enterprise Financial BI System

**Note:** All code samples use generic examples and dummy data. No confidential information is included.

---

## 📋 Table of Contents

1. [Data Extraction](#data-extraction)
2. [Data Transformation](#data-transformation)
3. [Data Quality Validation](#data-quality-validation)
4. [Deduplication Logic](#deduplication-logic)
5. [Currency Conversion](#currency-conversion)
6. [Performance Optimization](#performance-optimization)

---

## 1. Data Extraction

### Extract from Multiple Excel Workbooks

```m
let
    // Define source folder path
    SourceFolder = "C:\Data\Cashbooks\",
    
    // Get all Excel files from folder
    Source = Folder.Files(SourceFolder),
    
    // Filter for Excel files only
    FilteredFiles = Table.SelectRows(Source, 
        each Text.EndsWith([Name], ".xlsx") or Text.EndsWith([Name], ".xls")
    ),
    
    // Add column to extract data from each file
    ExtractData = Table.AddColumn(FilteredFiles, "Data", 
        each Excel.Workbook([Content])),
    
    // Expand the data
    ExpandedData = Table.ExpandTableColumn(ExtractData, "Data", 
        {"Name", "Data", "Kind"}, 
        {"TableName", "TableData", "Kind"}),
    
    // Filter for specific sheet name (e.g., "Transactions")
    FilterSheet = Table.SelectRows(ExpandedData, 
        each [TableName] = "Transactions" and [Kind] = "Sheet"),
    
    // Expand the transaction data
    FinalData = Table.ExpandTableColumn(FilterSheet, "TableData", 
        {"Date", "Amount", "Description", "Reference"}, 
        {"TransactionDate", "Amount", "Description", "Reference"}),
    
    // Add source file name for tracking
    AddSource = Table.AddColumn(FinalData, "SourceFile", 
        each [Name]),
    
    // Remove unnecessary columns
    CleanData = Table.RemoveColumns(AddSource, 
        {"Content", "Extension", "Date accessed", "Date modified", "Attributes", "Folder Path"})
in
    CleanData
```

### Extract Specific Range from Excel

```m
let
    // Load Excel file
    Source = Excel.Workbook(File.Contents("C:\Data\Budget.xlsx"), null, true),
    
    // Get specific sheet
    Sheet = Source{[Item="BudgetCodes",Kind="Sheet"]}[Data],
    
    // Promote headers
    PromotedHeaders = Table.PromoteHeaders(Sheet, [PromoteAllScalars=true]),
    
    // Change column types
    TypedData = Table.TransformColumnTypes(PromotedHeaders,{
        {"BudgetCode", type text},
        {"Component", type text},
        {"Activity", type text},
        {"BudgetAmount", Currency.Type},
        {"StartDate", type date},
        {"EndDate", type date}
    }),
    
    // Filter out null rows
    FilteredRows = Table.SelectRows(TypedData, 
        each [BudgetCode] <> null and [BudgetCode] <> "")
in
    FilteredRows
```

---

## 2. Data Transformation

### Standardize Transaction Data

```m
let
    Source = CashbookRawData,
    
    // Standardize column names
    RenamedColumns = Table.RenameColumns(Source,{
        {"Txn Date", "TransactionDate"},
        {"Amt", "Amount"},
        {"Desc", "Description"},
        {"Ref No", "ReferenceNumber"}
    }),
    
    // Add calculated columns
    AddYear = Table.AddColumn(RenamedColumns, "FiscalYear", 
        each Date.Year([TransactionDate])),
    
    AddQuarter = Table.AddColumn(AddYear, "FYQuarter", 
        each "Q" & Text.From(Date.QuarterOfYear([TransactionDate]))),
    
    AddMonth = Table.AddColumn(AddQuarter, "MonthName", 
        each Date.MonthName([TransactionDate])),
    
    // Clean text fields
    CleanDescription = Table.TransformColumns(AddMonth,{
        {"Description", Text.Upper, type text},
        {"ReferenceNumber", Text.Trim, type text}
    }),
    
    // Extract budget code from description
    ExtractBudgetCode = Table.AddColumn(CleanDescription, "BudgetCode", 
        each 
            let
                // Look for pattern like "BC-12345" in description
                Pattern = Text.Select([Description], {"0".."9", "-", "A".."Z"}),
                Code = if Text.Length(Pattern) >= 5 then Text.Start(Pattern, 8) else null
            in
                Code,
        type text),
    
    // Categorize transaction types
    AddTransactionType = Table.AddColumn(ExtractBudgetCode, "TransactionType",
        each 
            if Text.Contains([Description], "PAYMENT") then "Payment"
            else if Text.Contains([Description], "RECEIPT") then "Receipt"
            else if Text.Contains([Description], "TRANSFER") then "Transfer"
            else "Other",
        type text)
in
    AddTransactionType
```

### Normalize Supplier Names

```m
let
    Source = SupplierRawData,
    
    // Clean supplier names
    CleanNames = Table.TransformColumns(Source,{
        {"SupplierName", each Text.Trim(Text.Upper(_)), type text}
    }),
    
    // Remove common variations
    StandardizeNames = Table.ReplaceValue(
        CleanNames,
        "LTD",
        "LIMITED",
        Replacer.ReplaceText,
        {"SupplierName"}
    ),
    
    StandardizeNames2 = Table.ReplaceValue(
        StandardizeNames,
        "CO.",
        "COMPANY",
        Replacer.ReplaceText,
        {"SupplierName"}
    ),
    
    // Remove duplicate spaces
    RemoveExtraSpaces = Table.TransformColumns(StandardizeNames2,{
        {"SupplierName", each Text.Combine(Splitter.SplitTextByWhitespace()(_), " "), type text}
    }),
    
    // Assign unique supplier ID
    AddIndex = Table.AddIndexColumn(RemoveExtraSpaces, "SupplierID", 1, 1, Int64.Type)
in
    AddIndex
```

---

## 3. Data Quality Validation

### Multi-Level Validation Framework

```m
let
    Source = TransactionData,
    
    // Add validation flags
    AddValidationFlags = Table.AddColumn(Source, "DataQualityFlag",
        each 
            // Priority 1: Critical errors
            if [BudgetCode] = null or [BudgetCode] = "" then "CRITICAL_MISSING_BUDGET_CODE"
            else if [Amount] = null or [Amount] = 0 then "CRITICAL_INVALID_AMOUNT"
            else if [TransactionDate] = null then "CRITICAL_MISSING_DATE"
            
            // Priority 2: High priority issues
            else if not List.Contains(ValidBudgetCodes[BudgetCode], [BudgetCode]) then "HIGH_INVALID_BUDGET_CODE"
            else if [Amount] < 0 then "HIGH_NEGATIVE_AMOUNT"
            else if [SupplierName] = null or [SupplierName] = "" then "HIGH_MISSING_SUPPLIER"
            
            // Priority 3: Medium priority warnings
            else if Text.Length([Description]) < 10 then "MEDIUM_SHORT_DESCRIPTION"
            else if [SubItemCode] = null then "MEDIUM_MISSING_SUBITEM"
            
            // Priority 4: Low priority notes
            else if Text.Contains([Description], "PENDING") then "LOW_PENDING_CLASSIFICATION"
            
            // All clear
            else "OK",
        type text),
    
    // Add priority level
    AddPriority = Table.AddColumn(AddValidationFlags, "Priority",
        each 
            if Text.StartsWith([DataQualityFlag], "CRITICAL") then 1
            else if Text.StartsWith([DataQualityFlag], "HIGH") then 2
            else if Text.StartsWith([DataQualityFlag], "MEDIUM") then 3
            else if Text.StartsWith([DataQualityFlag], "LOW") then 4
            else 0,
        Int64.Type),
    
    // Add validation message
    AddMessage = Table.AddColumn(AddPriority, "ValidationMessage",
        each 
            if [DataQualityFlag] = "OK" then "Data passes all validation checks"
            else "Review required: " & [DataQualityFlag],
        type text)
in
    AddMessage
```

### Cross-Reference Validation

```m
let
    Transactions = TransactionData,
    BudgetMaster = BudgetCodesTable,
    
    // Left join to check if budget code exists
    JoinToBudget = Table.NestedJoin(
        Transactions, {"BudgetCode"},
        BudgetMaster, {"BudgetCode"},
        "BudgetReference",
        JoinKind.LeftOuter
    ),
    
    // Add validation based on join result
    AddValidation = Table.AddColumn(JoinToBudget, "BudgetCodeValid",
        each 
            if Table.RowCount([BudgetReference]) = 0 then false
            else true,
        type logical),
    
    // Expand budget details for valid codes
    ExpandBudget = Table.ExpandTableColumn(AddValidation, "BudgetReference",
        {"Component", "Activity", "BudgetAmount"},
        {"ComponentName", "ActivityName", "BudgetLimit"}),
    
    // Calculate remaining budget
    AddRemainingBudget = Table.AddColumn(ExpandBudget, "BudgetRemaining",
        each 
            if [BudgetCodeValid] then
                [BudgetLimit] - [Amount]
            else
                null,
        type number)
in
    AddRemainingBudget
```

---

## 4. Deduplication Logic

### Identify and Remove Duplicates Across Sources

```m
let
    // Combine data from multiple sources
    Source1 = OperatingCashbook,
    Source2 = ForeignCurrencyCashbook,
    
    // Add source identifier
    AddSource1 = Table.AddColumn(Source1, "DataSource", each "Operating", type text),
    AddSource2 = Table.AddColumn(Source2, "DataSource", each "ForeignCurrency", type text),
    
    // Combine tables
    Combined = Table.Combine({AddSource1, AddSource2}),
    
    // Create unique key for duplicate detection
    AddUniqueKey = Table.AddColumn(Combined, "UniqueKey",
        each 
            Text.From(Date.Year([TransactionDate])) & "-" &
            Text.PadStart(Text.From(Date.Month([TransactionDate])), 2, "0") & "-" &
            Text.PadStart(Text.From(Date.Day([TransactionDate])), 2, "0") & "-" &
            Text.From([Amount]) & "-" &
            Text.Upper(Text.Start([Description], 20)),
        type text),
    
    // Group by unique key to find duplicates
    GroupedRows = Table.Group(AddUniqueKey, {"UniqueKey"}, {
        {"Count", each Table.RowCount(_), Int64.Type},
        {"AllRows", each _, type table},
        {"DataSources", each Text.Combine([DataSource], ","), type text}
    }),
    
    // Mark duplicates
    AddDuplicateFlag = Table.AddColumn(GroupedRows, "IsDuplicate",
        each [Count] > 1,
        type logical),
    
    // For duplicates, keep only from primary source (Operating)
    SelectPrimary = Table.AddColumn(AddDuplicateFlag, "SelectedRow",
        each 
            if [IsDuplicate] then
                Table.SelectRows([AllRows], each [DataSource] = "Operating"){0}
            else
                [AllRows]{0}),
    
    // Expand selected rows
    ExpandSelected = Table.ExpandRecordColumn(SelectPrimary, "SelectedRow",
        {"TransactionDate", "Amount", "Description", "BudgetCode", "DataSource"},
        {"TransactionDate", "Amount", "Description", "BudgetCode", "FinalSource"}),
    
    // Remove temporary columns
    CleanData = Table.RemoveColumns(ExpandSelected, 
        {"AllRows", "DataSources"}),
    
    // Add deduplication note
    AddNote = Table.AddColumn(CleanData, "DeduplicationNote",
        each 
            if [IsDuplicate] then "Duplicate found - kept from " & [FinalSource]
            else "Unique transaction",
        type text)
in
    AddNote
```

### Calculate Duplication Rate

```m
let
    DeduplicatedData = DuplicationQuery,
    
    // Calculate statistics
    TotalRecords = Table.RowCount(DeduplicatedData),
    DuplicateRecords = Table.RowCount(
        Table.SelectRows(DeduplicatedData, each [IsDuplicate] = true)
    ),
    UniqueRecords = TotalRecords - DuplicateRecords,
    
    // Calculate percentages
    DuplicationRate = DuplicateRecords / TotalRecords,
    
    // Create summary table
    Summary = #table(
        {"Metric", "Value", "Percentage"},
        {
            {"Total Records", TotalRecords, 1.0},
            {"Unique Records", UniqueRecords, UniqueRecords / TotalRecords},
            {"Duplicate Records", DuplicateRecords, DuplicationRate}
        }
    ),
    
    // Format percentages
    FormatPercentage = Table.TransformColumns(Summary,{
        {"Percentage", each Number.ToText(_, "P2"), type text}
    })
in
    FormatPercentage
```

---

## 5. Currency Conversion

### Dynamic Exchange Rate Conversion

```m
let
    Source = TransactionData,
    ExchangeRates = ExchangeRateTable,
    
    // Join to exchange rates by date
    JoinRates = Table.NestedJoin(
        Source, {"TransactionDate"},
        ExchangeRates, {"Date"},
        "RateInfo",
        JoinKind.LeftOuter
    ),
    
    // Expand exchange rate
    ExpandRate = Table.ExpandTableColumn(JoinRates, "RateInfo",
        {"MWKtoUSD"},
        {"ExchangeRate"}),
    
    // Handle missing rates (use previous available rate)
    FillMissingRates = Table.FillDown(ExpandRate, {"ExchangeRate"}),
    
    // Convert amount to USD
    AddUSDAmount = Table.AddColumn(FillMissingRates, "AmountUSD",
        each 
            if [Currency] = "MWK" then
                [Amount] / [ExchangeRate]
            else if [Currency] = "USD" then
                [Amount]
            else
                null,
        Currency.Type),
    
    // Add conversion note
    AddConversionNote = Table.AddColumn(AddUSDAmount, "ConversionNote",
        each 
            if [Currency] = "MWK" then
                "Converted from MWK at rate: " & Text.From([ExchangeRate])
            else if [Currency] = "USD" then
                "Already in USD"
            else
                "Unknown currency",
        type text),
    
    // Round to 2 decimal places
    RoundUSD = Table.TransformColumns(AddConversionNote,{
        {"AmountUSD", each Number.Round(_, 2), Currency.Type}
    })
in
    RoundUSD
```

---

## 6. Performance Optimization

### Query Folding Optimization

```m
let
    // Enable query folding by filtering early
    Source = Sql.Database("ServerName", "DatabaseName"),
    
    // Filter at source (folds to SQL WHERE clause)
    FilteredAtSource = Table.SelectRows(Source, 
        each [TransactionDate] >= #date(2024, 1, 1) and 
             [TransactionDate] <= #date(2024, 12, 31) and
             [Amount] > 0
    ),
    
    // Select only needed columns (folds to SQL SELECT)
    SelectedColumns = Table.SelectColumns(FilteredAtSource,
        {"TransactionDate", "Amount", "BudgetCode", "Description"}
    ),
    
    // Aggregate at source if possible (folds to SQL GROUP BY)
    GroupedAtSource = Table.Group(SelectedColumns, {"BudgetCode"}, {
        {"TotalAmount", each List.Sum([Amount]), Currency.Type},
        {"TransactionCount", each Table.RowCount(_), Int64.Type}
    })
in
    GroupedAtSource
```

### Connection-Only Query Pattern

```m
// Master Data Query (Connection Only - Not Loaded to Model)
let
    Source = Excel.Workbook(File.Contents("C:\Data\MasterData.xlsx")),
    AllTransactions = Source{[Item="Transactions",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(AllTransactions),
    TypedData = Table.TransformColumnTypes(PromotedHeaders, /* column types */)
in
    TypedData

// Aggregated Query (Loaded to Model)
let
    Source = MasterDataQuery,  // References connection-only query
    
    // Perform aggregation (only aggregate data loaded)
    Aggregated = Table.Group(Source, 
        {"Category", "FiscalYear"}, 
        {
            {"TotalSpend", each List.Sum([Amount]), Currency.Type},
            {"TransactionCount", each Table.RowCount(_), Int64.Type},
            {"AvgAmount", each List.Average([Amount]), Currency.Type}
        }
    )
in
    Aggregated
```

### Incremental Refresh Pattern

```m
let
    // Use parameters for date range (enables incremental refresh)
    StartDate = #date(2024, 1, 1),
    EndDate = DateTime.Date(DateTime.LocalNow()),
    
    Source = TransactionData,
    
    // Filter to date range
    FilteredDates = Table.SelectRows(Source,
        each [TransactionDate] >= StartDate and 
             [TransactionDate] <= EndDate
    ),
    
    // Add RangeStart and RangeEnd columns (required for Power BI incremental refresh)
    AddRangeStart = Table.AddColumn(FilteredDates, "RangeStart", 
        each StartDate, type date),
    AddRangeEnd = Table.AddColumn(AddRangeStart, "RangeEnd", 
        each EndDate, type date)
in
    AddRangeEnd
```

---

## 📊 Query Performance Tips

### Best Practices Implemented

1. **Early Filtering**
   - Filter rows as early as possible in the query
   - Use query folding to push filters to source system

2. **Column Selection**
   - Remove unnecessary columns early
   - Only select columns needed for analysis

3. **Connection-Only Queries**
   - Use for large source tables
   - Reference in multiple aggregated queries
   - Prevents loading duplicate data

4. **Efficient Joins**
   - Perform joins after filtering
   - Use appropriate join types (inner vs left)
   - Avoid cross joins

5. **Data Type Specification**
   - Always specify correct data types
   - Helps with query folding
   - Improves performance

6. **Aggregation**
   - Aggregate at source when possible
   - Pre-calculate summary tables
   - Reduce row count before loading

---

## 🔧 Error Handling

### Robust Error Handling Pattern

```m
let
    Source = try TransactionData otherwise null,
    
    // Check if source loaded successfully
    CheckSource = if Source = null then
        #table(
            {"Error"},
            {{"Source data could not be loaded"}}
        )
    else
        // Proceed with transformation
        let
            ProcessData = try Table.TransformColumns(Source, /* transformations */) 
                          otherwise Source,
            
            ValidateData = try Table.SelectRows(ProcessData, 
                              each [Amount] <> null) 
                          otherwise ProcessData,
            
            // Add error column to track issues
            AddErrorColumn = Table.AddColumn(ValidateData, "ProcessingError",
                each 
                    try
                        // Validation logic
                        if [BudgetCode] = null then "Missing budget code"
                        else "OK"
                    otherwise "Error during validation",
                type text)
        in
            AddErrorColumn
in
    CheckSource
```

---

## 📝 Notes

- All code samples are sanitized and use generic examples
- No actual budget codes, amounts, or supplier names included
- Patterns demonstrate technical capability without revealing confidential data
- Code follows Power Query best practices for performance and maintainability

---

**Author:** Swithun M. Chiziko  
**Purpose:** Portfolio demonstration of Power Query development skills  
**Context:** Enterprise financial BI system for international development program
