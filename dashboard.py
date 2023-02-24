import pandas as pd
import streamlit as st
import plotly.express as px


# Set page
st.set_page_config(page_title="Alzheimer's Disease", 
                   layout="wide")

# Irrelevant data to be dropped during parse
drop_col = ["RowId",
            "Datasource",
            "Response",
            "Data_Value_Unit",
            "Data_Value_Type",
            "Data_Value_Footnote_Symbol",
            "Data_Value_Footnote",
            #"Low_Confidence_Limit",
            #"High_Confidence_Limit",
            "Sample_Size",
            "StratificationCategory1",
            "StratificationCategory2",
            "StratificationCategory3",
            "ClassID",
            "TopicID",
            #"QuestionID",
            #"LocationID",
            "Stratification3",
            "ResponseID",
            "StratificationCategoryID1",
            "StratificationID1",
            #"StratificationCategoryID2",
            #"StratificationID2",
            "StratificationCategoryID3",
            "StratificationID3",
            "Report",
            "Geolocation",
            "DataValueTypeID"]


# Main Page Layout
st.title("Data Visualization of Alzheimer's Disease and Healthy Aging Data")
st.markdown("By Jay An, Hope Andrescavage, Patrick Hamill, Khayla Wehman")

# Cache CSV 
@st.cache
def parseCSV():
    df = pd.read_csv('Alzheimers Disease/data.csv', usecols=lambda x: x not in drop_col)
    return df

df = parseCSV()

def clear_selection():
    st.session_state.multiselect = []
    return

# Query Selection 
st.sidebar.header("Sample")

startYearSelector = st.sidebar.multiselect(
    "Select a Start Year:",
    options=sorted(df["YearStart"].unique())
)

endYearSelector = st.sidebar.multiselect(
    "Select an End Year:",
    options=sorted(df["YearEnd"].unique())
)

topicSelector = st.sidebar.multiselect(
    "Select a Categorization:",
    options=sorted(df["Topic"].unique())
)

classSelector = st.sidebar.multiselect(
    "Select a Class:",
    options=sorted(df["Class"].unique())
)

ageRangeSelector = st.sidebar.multiselect(
    "Select an Age Range:",
    options=sorted(df["Stratification1"].unique())
)

if st.sidebar.checkbox("Select All States"):
    LocationSelector = st.sidebar.multiselect(
        "Select Location:",
        options=sorted(df["LocationDesc"].unique()),
        default=sorted(df["LocationDesc"].unique())
    )
else:
    LocationSelector = st.sidebar.multiselect(
    "Select State(s):",
    options=sorted(df["LocationDesc"].unique())
    )

df_selection = df.query(
    "YearStart == @startYearSelector |" +
    "YearEnd == @endYearSelector |" +
    "Topic == @topicSelector | " + 
    "Class == @classSelector |" +
    "Stratification1 == @ageRangeSelector |" +
    "LocationDesc == @LocationSelector"
)

df_selection_Strict = df.query(
    "(YearStart == @startYearSelector & YearEnd == @endYearSelector) &" +
    "(Topic == @topicSelector | Class == @classSelector) &" +
    "Stratification1 == @ageRangeSelector &" +
    "LocationDesc == @LocationSelector"
)

# Graphs

locationBarGraph = px.bar(df_selection_Strict["LocationDesc"].value_counts(),
                          title="State",
                          template="plotly_white").update_layout(xaxis_title="States")

yearStartBarGraph = px.bar(df_selection_Strict["YearStart"].value_counts(),
                           title="Year Start",
                           template="plotly_white").update_xaxes(dtick=1).update_traces(width=.9).update_layout(xaxis_title="Years")

yearEndBarGraph = px.bar(df_selection_Strict["YearEnd"].value_counts(),
                         title="Year End",
                         template="plotly_white").update_xaxes(dtick=1).update_traces(width=.9).update_layout(xaxis_title="Years")

lowConGraph = px.scatter(x = df["Low_Confidence_Limit"], 
                         y = df["Data_Value"],
                         template="plotly_white").update_layout(xaxis_title="Low Confidence Limit", yaxis_title="Data Value")

highConGraph = px.scatter(x = df["High_Confidence_Limit"],
                          y = df["Data_Value"],
                          template="plotly_white").update_layout(xaxis_title="High Confidence Limit", yaxis_title="Data Value")


# Graphs / Data Display
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs((["Preview","Bar Chart", "Data Correlation", "Outlier", "Decision Tree", "Association Rule"]))
with tab1:
    st.markdown("Refined Preview - Use Side Bar")
    st.dataframe(df_selection)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("Age by Class")
        st.write(df.groupby(["Stratification1", "Class"]).size())

    with col2:
        st.markdown("Demographic")
        x = df.groupby(["StratificationCategoryID2", "StratificationID2"]).size().to_frame("Size")
        st.write(x)
   
    with col3:
        st.markdown("Demographic by State")
        st.write(df.groupby(["LocationDesc", "StratificationID2"]).size())
    
    col12, col22 = st.columns(2)
    with col12:
        st.markdown("Topic by Age Group")
        st.write(df.groupby(["Question", "Stratification1"]).size())
    with col22:
        st.markdown("Topic by Demographic")
        st.write(df.groupby(["Question", "StratificationID2"]).size()) 

    with st.expander("???"):
        st.image("https://pics.me.me/train-pepe-3-38637769.png")    
    
with tab2:
    if tab2:
        st.subheader("Refined Strict Search")
        st.markdown("Use Side Bar")
        st.plotly_chart(locationBarGraph, use_container_width=True)
        st.plotly_chart(yearStartBarGraph, use_container_width=True)
        st.plotly_chart(yearEndBarGraph, use_container_width=True)
        st.dataframe(df_selection_Strict)

with tab3:
    if tab3:
        #pass
        df2 = df.corr()
        
        st.plotly_chart(px.imshow(df2, text_auto=True), use_container_width=True)
        st.write(df2)

with tab4:
    if tab4:
        st.subheader("Low Confidence Limit")
        st.plotly_chart(lowConGraph, use_container_width=True)
        st.subheader("High Confidence Limit")
        st.plotly_chart(highConGraph, use_container_width=True)
        
with tab5:
    if tab5:
        st.subheader("Alzheimers Indicators max leaf nodes = 5, max depth = 3")
        st.image("Alzheimers_Indicators_Decision_Tree_1.png", use_column_width=True)
        st.subheader("Alzheimers Indicators max leaf nodes = 7, max depth = 4")
        st.image("Alzheimers_Indicators_Decision_Tree_2.png", use_column_width=True)
        st.subheader("Alzheimers Indicators max leaf nodes = 9, max depth = 5")
        st.image("Alzheimers_Indicators_Decision_Tree_3.png", use_column_width=True)
        st.subheader("Alzheimers Indicators max leaf nodes = 12, max depth = 6")
        st.image("Alzheimers_Indicators_Decision_Tree_4.png", use_column_width=True)
    
with tab6:
    if tab6:
        st.subheader("Distribution of association rule antecedents by support")
        st.image("Count_of_antecedents.png", use_column_width=True)