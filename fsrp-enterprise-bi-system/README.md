# Enterprise Financial Reporting & BI System
## World Bank-Funded Agricultural Development Program

![Status](https://img.shields.io/badge/Status-Production-success)
![Excel](https://img.shields.io/badge/Excel-Power%20Query-217346)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811)
![Impact](https://img.shields.io/badge/Time%20Saved-35%25-blue)

---

## 📋 Project Overview

Developed and deployed a comprehensive Business Intelligence and financial reporting system for a **World Bank-funded agricultural development program** in Malawi. The system integrates multiple data sources, automates donor reporting workflows, and provides real-time financial insights to program management and international stakeholders.

**Organization:** AGCOM Malawi (Agricultural Commodity Exchange for Africa)  
**Program:** Multi-million dollar agricultural resilience project  
**Duration:** July 2025 - Present  
**Role:** Finance Intern & BI Developer

---

## 🎯 Business Challenge

### The Problem
- **Manual reporting processes** taking 4-5 days per month
- **Multiple disconnected data sources** (3 cashbook systems)
- **No real-time visibility** into program expenditure
- **Data quality issues** affecting donor reporting
- **Complex budget structure** with 277 budget codes across multiple project components
- **World Bank compliance requirements** for detailed financial reporting

### The Impact
Program management lacked timely insights for decision-making, and World Bank donor reports required extensive manual consolidation with high risk of errors.

---

## 💡 Solution Architecture

### System Overview

Built a comprehensive **Excel-based BI system** with 45 interconnected worksheets, processing **8,000+ transaction records** from multiple sources with automated data integration, validation, and visualization.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INTEGRATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ NBS Cashbook │  │ FCDA Foreign │  │ LCDA Local   │         │
│  │ (Operating)  │  │   Currency   │  │   Currency   │         │
│  │ 6,199 records│  │   38 records │  │ 5,575 records│         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                 │
│                             │                                    │
│                   ┌─────────▼─────────┐                        │
│                   │  Power Query ETL  │                        │
│                   │  - Deduplication  │                        │
│                   │  - Validation     │                        │
│                   │  - Transformation │                        │
│                   └─────────┬─────────┘                        │
└─────────────────────────────┼─────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                    DATA WAREHOUSE LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Supplier    │  │   Budget     │  │  Financial   │        │
│  │  Analysis    │  │  Tracking    │  │  Statements  │        │
│  │ 3,951 records│  │ 398 activities│  │  (SoRP/SoFP) │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Cashier    │  │   Activity   │  │ Data Quality │        │
│  │  Performance │  │   Analysis   │  │   Reports    │        │
│  │ 1,179 records│  │ 837 records  │  │   4 levels   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────┬─────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                   PRESENTATION LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Executive   │  │  Component   │  │  Expenditure │        │
│  │  Dashboard   │  │  Analysis    │  │   Analysis   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Supplier   │  │   Cashier    │  │   Activity   │        │
│  │   Tracking   │  │   Monitor    │  │   Details    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Implementation

### Data Integration & ETL

**Challenge:** Three separate cashbook systems with overlapping data causing potential double-counting.

**Solution:**
- Built **Power Query M** scripts to extract and transform data from multiple Excel workbooks
- Discovered **87-94% duplication** between LCDA and NBS cashbooks through correlation analysis
- Implemented intelligent deduplication logic to prevent double-counting
- Created **data quality validation** framework with 4 priority levels

**Key Technical Decisions:**
```
Data Source Strategy:
├── Primary: NBS Operating Cashbook (authoritative source)
├── Secondary: FCDA Foreign Currency (unique transactions only)
└── Excluded: LCDA Local Currency (prevent duplication)

Rationale: Analysis showed LCDA transactions were already captured 
in NBS, causing 90%+ overlap. Exclusion ensures data integrity.
```

### Power Query Development

**Implemented 15+ Power Query transformations:**

1. **Data Extraction**
   - Multi-workbook consolidation
   - Dynamic column mapping
   - Error handling for missing sources

2. **Data Transformation**
   - Currency conversion (MWK → USD)
   - Date standardization across sources
   - Budget code validation
   - Supplier name normalization

3. **Data Quality**
   - Missing value detection
   - Invalid code flagging
   - Cross-reference validation
   - Duplicate identification

4. **Performance Optimization**
   - Query folding where possible
   - Reduced refresh time by **92%** (25 min → 2 min)
   - Implemented incremental refresh patterns
   - Minimized data loading through connection-only queries

### Dashboard Development

**Created 10+ Interactive Dashboards:**

| Dashboard | Purpose | Key Metrics | Users |
|-----------|---------|-------------|-------|
| **Executive Overview** | Strategic KPIs | Total spend, budget utilization, trends | Senior Management, World Bank |
| **Component Analysis** | Project component tracking | Budget vs. actual by component | Program Managers |
| **Expenditure Analysis** | Detailed spending patterns | Category breakdown, variance analysis | Finance Team |
| **Supplier Intelligence** | Vendor performance | Top suppliers, concentration risk | Procurement Team |
| **Cashier Performance** | Disbursement monitoring | Subsistence tracking, outliers | Operations Team |
| **Activity Tracking** | Budget code monitoring | 277 codes across activities | Finance & Management |
| **Data Quality Report** | Validation dashboard | Exception tracking, priority flags | Data Governance |
| **Financial Statements** | SoRP & SoFP | Donor compliance reporting | World Bank, Management |

**Technical Features:**
- ✅ Dynamic filtering and drill-down capabilities
- ✅ Conditional formatting for variance alerts
- ✅ Exception highlighting (4-level priority system)
- ✅ YTD, QTD, and MTD calculations
- ✅ Budget variance analysis with traffic light indicators
- ✅ Supplier concentration risk metrics
- ✅ Automated data refresh workflows

---

## 📊 Power BI Migration Strategy

Developed comprehensive **Power BI migration guide** (1,887 lines) to transition dashboards from Excel to Power BI for enhanced scalability.

### Planned Power BI Features

**5-Page Dashboard Suite:**

1. **Executive Overview** - Strategic KPIs and trend analysis
2. **Expenditure Deep Dive** - SubItem code analysis with decomposition trees
3. **Supplier Intelligence** - Vendor tracking with custom tooltips
4. **Cashier Performance** - Disbursement monitoring with gauges
5. **Budget Performance** - Waterfall charts and variance analysis

**Technical Approach:**
- 50+ DAX measures for advanced calculations
- Star schema data model for optimal performance
- Row-level security for stakeholder access control
- Incremental refresh for large datasets
- Mobile-responsive design

---

## 🎯 Business Impact & Results

### Quantifiable Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Monthly Reporting Time** | 4-5 days | 3 days | **35% reduction** |
| **Data Accuracy** | Manual errors common | 100% validated | **Eliminated errors** |
| **Data Refresh Time** | 25 minutes | 2 minutes | **92% faster** |
| **Budget Code Tracking** | Manual spreadsheets | Automated dashboard | **Real-time visibility** |
| **Stakeholder Reports** | 5 separate documents | Single integrated system | **Unified reporting** |
| **Decision-Making Speed** | Week-old data | Same-day insights | **Real-time analysis** |

### Critical Business Value

**1. Data Quality Discovery**
- Identified **$7.8M in unbudgeted spending** requiring immediate management attention
- Detected systematic data quality issues before World Bank audit
- Created exception tracking preventing compliance issues

**2. Operational Efficiency**
- Automated previously manual 4-day reporting process
- Freed finance team capacity for strategic analysis
- Enabled monthly (vs quarterly) donor reporting

**3. Strategic Insights**
- Real-time visibility into program expenditure patterns
- Component-level budget tracking across 277 codes
- Supplier concentration risk monitoring
- Cash flow forecasting capabilities

**4. Stakeholder Enablement**
- World Bank staff can self-serve reports
- Program managers have real-time budget status
- Finance team can quickly respond to ad-hoc queries

---

## 💻 Technical Stack

### Core Technologies

**Data Integration:**
- **Excel Power Query (M Language)** - ETL development
- **Power Pivot** - Data modeling
- **DAX** - Advanced calculations

**Visualization:**
- **Excel** - Interactive dashboards (current production)
- **Power BI Desktop** - Migration in progress
- **DAX measures** - 50+ calculated metrics

**Data Sources:**
- **Excel Workbooks** - 3 cashbook systems
- **Tompro Accounting System** - Source data extraction
- **Exchange Rate APIs** - Currency conversion

**Development Tools:**
- **Power Query Editor** - M code development
- **Excel Formula Language** - Complex calculations
- **Git/GitHub** - Version control (planned)

### Technical Skills Demonstrated

**Data Engineering:**
- ✅ Multi-source data integration
- ✅ ETL pipeline development
- ✅ Data quality framework implementation
- ✅ Performance optimization
- ✅ Incremental refresh patterns

**Business Intelligence:**
- ✅ Dashboard design and development
- ✅ KPI definition and tracking
- ✅ Data modeling (star schema)
- ✅ DAX measure development
- ✅ User experience design

**Data Analysis:**
- ✅ Financial analysis and variance reporting
- ✅ Trend analysis and forecasting
- ✅ Exception identification
- ✅ Budget vs actual analysis
- ✅ Supplier performance metrics

**Project Management:**
- ✅ Requirements gathering with stakeholders
- ✅ Phased implementation approach
- ✅ Documentation and training
- ✅ Change management
- ✅ User acceptance testing

---

## 🏗️ System Architecture Details

### Data Model Structure

```
Fact Tables:
├── TransactionDetail (8,000+ records)
│   ├── TransactionID (PK)
│   ├── Date
│   ├── Amount (USD & MWK)
│   ├── Supplier (FK)
│   ├── BudgetCode (FK)
│   ├── Cashier (FK)
│   └── DataQualityFlag
│
├── BudgetAllocation (398 activities)
│   ├── ActivityCode (PK)
│   ├── ComponentID (FK)
│   ├── BudgetAmount
│   └── Financier (FK)
│
Dimension Tables:
├── ChartOfAccounts (124 codes)
├── Suppliers (3,951 unique)
├── ProjectComponents (24 components)
├── Financiers (4 sources)
└── Cashiers (Active staff)
```

### ETL Workflow

```mermaid
graph LR
    A[Source Cashbooks] --> B[Extract]
    B --> C[Validate]
    C --> D[Transform]
    D --> E[Deduplicate]
    E --> F[Enrich]
    F --> G[Load]
    G --> H[Aggregate]
    H --> I[Dashboards]
```

**Process Steps:**

1. **Extract** - Pull data from 3 cashbook sources
2. **Validate** - Check data quality, flag exceptions
3. **Transform** - Standardize formats, convert currencies
4. **Deduplicate** - Remove overlapping transactions
5. **Enrich** - Add budget references, component mapping
6. **Load** - Write to analytical tables
7. **Aggregate** - Pre-calculate summary metrics
8. **Dashboards** - Present insights to users

---

## 📈 Key Features

### 1. Multi-Source Integration

**Challenge:** Integrate 3 separate cashbook systems without double-counting

**Implementation:**
- Analyzed transaction overlap patterns
- Identified 10% duplication system 
- Developed deduplication algorithm
- Implemented primary source logic

**Result:** Single source of truth for financial data

### 2. Data Quality Framework

**4-Level Priority System:**

| Priority | Flag Type | Action Required | Count |
|----------|-----------|-----------------|-------|
| **Critical** | Missing budget code | Immediate correction | 12 |
| **High** | Invalid supplier | Review within 24h | 8 |
| **Medium** | Missing subitem | Review within week | 45 |
| **Low** | Minor formatting | Review monthly | 89 |

**Automated Validation Rules:**
- Budget code existence check
- Supplier name normalization
- Date range validation
- Amount reasonableness checks
- Currency consistency verification

### 3. Budget Variance Tracking

**Real-time Monitoring:**
- 277 budget codes across program
- Component-level rollup (24 components)
- Activity-level detail (398 activities)
- Automatic variance calculation
- Exception highlighting (>10% variance)

**Alert System:**
- 🔴 Over budget (>100% utilization)
- 🟡 Warning (90-100% utilization)
- 🟢 On track (<90% utilization)

### 4. Supplier Intelligence

**Analysis Dimensions:**
- Total spend per supplier (3,951 unique suppliers)
- Transaction frequency
- Average transaction size
- Payment concentration risk
- Category breakdown

**Risk Metrics:**
- Top 5 supplier concentration
- Single-supplier dependencies
- Payment pattern anomalies
- Compliance tracking

### 5. Financial Statements

**Automated Generation:**
- **Statement of Receipts and Payments (SoRP)** - Cash basis
- **Statement of Financial Position (SoFP)** - Assets & liabilities
- **Budget vs Actual Reports** - Variance analysis
- **Cashflow Forecasting** - Liquidity planning

**World Bank Compliance:**
- Follows IPSAS cash basis standards
- Component-level detail as required
- Financier-wise breakdown
- Quarterly submission ready

---

## 🔐 Data Security & Governance

**Access Control:**
- Role-based worksheet protection
- Sensitive data segregation
- Audit trail for changes
- Version control

**Data Privacy:**
- No personal identifying information
- Supplier anonymization in exports
- Compliance with organizational policies

**Quality Assurance:**
- Automated validation checks
- Manual review workflow
- Exception escalation process
- Regular data audits

---

## 📚 Documentation

Created comprehensive system documentation:

1. **User Guide** - Dashboard navigation and interpretation
2. **Technical Specification** - Data model and ETL logic
3. **Power Query Documentation** - M code explanations
4. **Power BI Migration Guide** - 1,887-line implementation guide
5. **Training Materials** - Onboarding for new users
6. **Data Dictionary** - Field definitions and business rules

---

## 🚀 Future Enhancements

### Short-term (Next 3 months)
- [ ] Complete Power BI migration
- [ ] Implement row-level security
- [ ] Add mobile dashboards
- [ ] Deploy to Power BI Service

### Medium-term (6-12 months)
- [ ] Integrate with Tompro accounting system API
- [ ] Add predictive analytics (budget forecasting)
- [ ] Implement automated email alerts
- [ ] Create executive mobile app

### Long-term (12+ months)
- [ ] Machine learning for anomaly detection
- [ ] Natural language query interface
- [ ] Advanced analytics (spend optimization)
- [ ] Integration with other donor systems

---

## 💡 Key Learnings & Challenges

### Technical Challenges Overcome

**1. Data Duplication Mystery**
- **Problem:** Inconsistent transaction counts across sources
- **Investigation:** Built correlation analysis in Power Query
- **Discovery:** 87-94% overlap between LCDA and NBS
- **Solution:** Documented decision to exclude LCDA from operational reporting

**2. Performance Optimization**
- **Problem:** 25-minute refresh time unacceptable for monthly reporting
- **Analysis:** Profiled queries to identify bottlenecks
- **Solution:** Implemented query folding and connection-only queries
- **Result:** 92% improvement (2-minute refresh)

**3. Complex Budget Structure**
- **Problem:** 277 budget codes with nested components
- **Challenge:** Activities contain multiple sub-activities
- **Solution:** Created hierarchical mapping table
- **Result:** Drill-down capability from component to transaction level

**4. Unbudgeted Spending Discovery**
- **Problem:** ~$7.8M in transactions without valid budget codes
- **Action:** Flagged as critical data quality issue
- **Impact:** Required immediate management attention
- **Resolution:** Systematic budget code assignment process implemented

### Stakeholder Management

**World Bank Requirements:**
- Monthly financial reporting (previously quarterly)
- Component-level expenditure detail
- Financier-wise breakdown (4 separate sources)
- Data quality assurance

**Internal Users:**
- Finance team (detailed reconciliation)
- Program managers (budget monitoring)
- Operations team (cashier performance)
- Senior management (executive overview)

**Success Factors:**
- Regular stakeholder meetings
- Iterative development approach
- User training and documentation
- Quick response to feedback

---

## 🎓 Skills Demonstrated

**Technical:**
- Advanced Excel (Power Query, Power Pivot, complex formulas)
- Power BI (DAX, data modeling, visualization design)
- M Language (ETL scripting)
- Data analysis and quality assurance
- Performance optimization

**Business:**
- Financial reporting and analysis
- Donor compliance requirements
- Budget monitoring and variance analysis
- Stakeholder communication
- Requirements gathering

**Project Management:**
- Phased implementation
- Documentation and training
- Change management
- User acceptance testing
- Continuous improvement

---

## 📧 Contact & Collaboration

**Developer:** Swithun M. Chiziko  
**Role:** Finance Intern & BI Developer  
**Organization:** AGCOM Malawi  
**Email:** chizikoswith@gmail.com  
**LinkedIn:** [linkedin.com/in/swithun-chiziko-94a21869](https://linkedin.com/in/swithun-chiziko-94a21869)

---

## 📄 Important Notes

### Confidentiality

This portfolio showcases the **technical architecture and methodology** of the system. All specific financial data, budget amounts, supplier names, and other confidential information have been removed or anonymized.

**What's Included:** System design, technical approach, process improvements  
**What's Excluded:** Actual financial data, real budget codes, supplier details

### Purpose

This project demonstrates:
- ✅ Enterprise-scale BI development capabilities
- ✅ Complex data integration and ETL skills
- ✅ Financial systems and donor reporting knowledge
- ✅ Power Query and DAX proficiency
- ✅ Dashboard design and data visualization
- ✅ Stakeholder management and documentation

---

## 🏆 Recognition

- Successfully deployed system used for World Bank monthly reporting
- Identified critical data quality issues before external audit
- Reduced reporting time by 35%, freeing team capacity
- Enabled real-time budget monitoring for program management
- Created comprehensive documentation for system sustainability

---

**Last Updated:** December 2025  
**Project Status:** ✅ Production | 🔄 Power BI Migration In Progress  
**Impact:** 💼 Enterprise BI System | 🌍 International Development | 💰 Multi-Million Dollar Program
