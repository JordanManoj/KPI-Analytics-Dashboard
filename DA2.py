import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode


edited_df = DataFrameEditor(df)
st.dataframe(edited_df)



st.set_page_config(page_title="Procurement Dashboard", layout="wide")


FILE_PATH = "processed_procurement.csv"   

@st.cache_data
def load_data(path):

    df = pd.read_csv(path)

    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    #DATE DETECTION
    date_col = None

    preferred = [
        "order_date", "order date", "date",
        "orderdate", "invoice_date", "delivery_date"
    ]

    cols_lower = {c: c.lower() for c in df.columns}

    for p in preferred:
        for orig, low in cols_lower.items():
            if p == low:
                date_col = orig
                break
        if date_col:
            break
    if not date_col:
        for orig, low in cols_lower.items():
            if "date" in low:
                date_col = orig
                break
    if date_col:

        df[date_col] = pd.to_datetime(df[date_col],
                                      errors="coerce",
                                      infer_datetime_format=True)

        nat_frac = df[date_col].isna().mean()

    
        if nat_frac > 0.3:
            df[date_col] = pd.to_datetime(df[date_col],
                                          errors="coerce",
                                          dayfirst=True,
                                          infer_datetime_format=True)
            nat_frac = df[date_col].isna().mean()

       
        if nat_frac > 0.3:
            formats = [
                "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
                "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d"
            ]

            for fmt in formats:
                parsed = pd.to_datetime(df[date_col].astype(str),
                                        format=fmt,
                                        errors="coerce")
                if parsed.notna().sum() > 0:
                    df[date_col] = df[date_col].fillna(parsed)

            nat_frac = df[date_col].isna().mean()

        if df[date_col].notna().sum() == 0:
            date_col = None
        else:
            df = df.dropna(subset=[date_col])  

    return df, date_col



df, detected_date_col = load_data(FILE_PATH)

# Sidebar filters

st.sidebar.header("Filters")

# Date filter
if detected_date_col:
    min_date = df[detected_date_col].min()
    max_date = df[detected_date_col].max()

    date_range = st.sidebar.date_input(
        "Order Date Range",
        value=[min_date, max_date]
    )

    if len(date_range) == 2:
        start, end = date_range
        df = df[
            (df[detected_date_col] >= pd.to_datetime(start)) &
            (df[detected_date_col] <= pd.to_datetime(end))
        ]

# Category filter
category_col = None
for col in df.columns:
    if "category" in col.lower():
        category_col = col
        break

if category_col:
    selected = st.sidebar.multiselect(
        "Category",
        df[category_col].unique(),
        default=df[category_col].unique()
    )
    df = df[df[category_col].isin(selected)]

# Supplier filter
supplier_col = None
for col in df.columns:
    if "supplier" in col.lower():
        supplier_col = col
        break

if supplier_col:
    selected = st.sidebar.multiselect(
        "Supplier",
        df[supplier_col].unique(),
        default=df[supplier_col].unique()
    )
    df = df[df[supplier_col].isin(selected)]

st.sidebar.markdown("---")



st.title(" Procurement KPI Dashboard")

col1, col2, col3, col4 = st.columns(4)

# Detect spend + quantity columns
spend_col = None
qty_col = None

for col in df.columns:
    if "unit" in col.lower() and "price" in col.lower():
        spend_col = col
    if "qty" in col.lower() or "quantity" in col.lower():
        qty_col = col


if spend_col and qty_col:
    df["Spend"] = df[spend_col] * df[qty_col]
    total_spend = df["Spend"].sum()
else:
    total_spend = 0

total_orders = len(df)
unique_suppliers = df[supplier_col].nunique() if supplier_col else 0
unique_categories = df[category_col].nunique() if category_col else 0

col1.metric("Total Spend", f"${total_spend:,.2f}")
col2.metric("Total Orders", total_orders)
col3.metric("Suppliers", unique_suppliers)
col4.metric("Categories", unique_categories)

st.markdown("---")


# Tabs on the dashboard
tab1, tab2, tab3, tab4 = st.tabs([
    " Overview",
    " Spend Trend",
    " Category Analysis",
    " Supplier Analysis"
])

# TAB 1 – OVERVIEW


with tab1:

    st.header("Overall Spend Overview")
    num_cols = df.select_dtypes(include=np.number).columns

    if len(num_cols) > 0:
        fig = px.histogram(
            df, x=num_cols[0], nbins=40,
            title=f"Distribution of {num_cols[0]}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    AgGrid(df)

# TAB 2 – SPEND TREND

with tab2:

    st.header("Spend Trend Over Time")

    if detected_date_col and spend_col and qty_col:

        if not pd.api.types.is_datetime64_any_dtype(df[detected_date_col]):
            df[detected_date_col] = pd.to_datetime(df[detected_date_col], errors="coerce")

        if pd.api.types.is_datetime64_any_dtype(df[detected_date_col]):
            df["Month"] = df[detected_date_col].dt.to_period("M").astype(str)
            df["Spend"] = df[spend_col] * df[qty_col]

            monthly = df.groupby("Month", sort=True)["Spend"].sum().reset_index()

            fig = px.line(
                monthly, x="Month", y="Spend",
                markers=True, title="Monthly Spend Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Date column exists but could not be parsed properly.")

    else:
        st.warning("Missing date/spend/quantity columns. Cannot plot trend.")

# TAB 3 – CATEGORY ANALYSIS

with tab3:

    st.header("Category-wise Spend")

    if category_col and spend_col and qty_col:
        df["Spend"] = df[spend_col] * df[qty_col]
        cat = df.groupby(category_col)["Spend"].sum().reset_index()

        fig = px.pie(
            cat,
            names=category_col,
            values="Spend",
            title="Spend by Category",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Category Table")
        AgGrid(cat)

    else:
        st.warning("Category or Spend Columns Missing!")

# TAB 4 – SUPPLIER ANALYSIS

with tab4:

    st.header("Supplier Spend Comparison")

    if supplier_col and spend_col and qty_col:
        df["Spend"] = df[spend_col] * df[qty_col]
        sup = df.groupby(supplier_col)["Spend"].sum().reset_index().sort_values(
            by="Spend", ascending=False
        )

        fig = px.bar(
            sup, x=supplier_col, y="Spend",
            title="Top Supplier Spend", color="Spend"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Supplier Table")
        AgGrid(sup)

    else:
        st.warning("Supplier or Spend Columns Missing!")

st.markdown("---")
st.caption("Interactive Procurement Dashboard – Date-safe Version")



