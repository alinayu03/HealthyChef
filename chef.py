from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import os
import streamlit as st

# Title
st.markdown("## Healthy Chef")

st.divider()

# Nutrition Search
st.markdown("## Nutrition Search")

# Read CSV
df = pd.read_csv('nutrition.csv')

# Search
search_query = st.text_input("Enter a search query")
column_to_search = st.selectbox("Select a column to search", df.columns)

# Nutrition Search Button
if st.button("Search"):
    filtered_rows = df[df[column_to_search].str.contains(
        search_query, case=False)]
    st.write(filtered_rows)

# Display dataset
st.dataframe(df)

# Secret OpenAI API Key
openai_api_key = st.secrets["openai_api_key"]

# User input OpenAI API Key
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Set key
os.environ["OPENAI_API_KEY"] = openai_api_key

st.divider()
# Ingredients input
ingredients = st.text_area("Enter ingredients list")
col1, col2 = st.columns(2)

# LLM setup
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, temperature=0.0)

# Recipe Generator 
col1.markdown("#### New Recipe")
template = """
        Task: Generate Healthy Recipes with Nutrition Facts based on a list of ingredients
        Ingredient List: {ingredients}"""

# Recipe Generator Button
if col1.button("Run", key="prompt_chain_button"):
    with st.spinner("Running"):
        prompt = PromptTemplate(
            input_variables=["ingredients"],
            template=template,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run({"ingredients": ingredients})
        col1.info(output)

# Nutrition Facts Generator
col2.markdown("#### Ingredients List Nutrition Facts")

# Nutrition Facts Button
if col2.button("Run", key="toolkit_agent_button"):
    with st.spinner("Running"):
        agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), df, verbose=True)
        instructions = """Search for the first instance of the ingredient name in the dataset."""
        output = agent.run(f"Calculate the nutrition facts for this list of ingredients: {ingredients}")
        col2.info(output)
