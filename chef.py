from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Set key
os.environ["OPENAI_API_KEY"] = openai_api_key

st.divider()

# init options
meal_type = ""
culture = ""
high_protein = ""
low_carb = ""
sugar_free = ""
low_fat = ""
low_sodium = ""

# Optional Preferences
meal_type = st.radio("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])
culture = st.text_input("Culture")
high_protein = st.checkbox("High-protein")
low_carb = st.checkbox("Low-carb")
sugar_free = st.checkbox("Sugar-free")
low_fat = st.checkbox("Low-fat")
low_sodium = st.checkbox("Low-sodium")

# Ingredients input
ingredients = st.text_area("Enter ingredients list")

# LLM setup
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, temperature=0.0)

st.write(meal_type)

# Recipe Generator 
st.markdown("#### New Recipe")
template = """
        Task: Generate A Healthy Recipe with Nutrition Facts based on a list of ingredients and optional preferences
        Ingredient List: {ingredients}

        Optional Preferences:
        Meal Type: {meal_type}
        Culture: {culture}
        Dietary Restrictions:
        - High-protein: {high_protein}
        - Low-carb: {low_carb}
        - Sugar-free: {sugar_free}
        - Low-fat: {low_fat}
        - Low-sodium: {low_sodium}"""

# Recipe Generator Button
if st.button("Run", key="prompt_chain_button"):
    with st.spinner("Running"):
        prompt = PromptTemplate(
            input_variables=["ingredients", "meal_type", "culture",
                             "high_protein", "low_carb", "sugar_free", "low_fat", "low_sodium"],
            template=template,
        )
        variables = {
            "ingredients": ingredients,
            "meal_type": str(meal_type),
            "culture": culture,
            "high_protein": high_protein,
            "low_carb": low_carb,
            "sugar_free": sugar_free,
            "low_fat": low_fat,
            "low_sodium": low_sodium,
        }
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run(variables)
        st.info(output)

# # Nutrition Facts Generator
# col2.markdown("#### Ingredients List Nutrition Facts")

# # Nutrition Facts Button
# if col2.button("Run", key="toolkit_agent_button"):
#     with st.spinner("Running"):
#         agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), df, verbose=True)
#         instructions = """Search for the first instance of the ingredient name in the dataset."""
#         output = agent.run(f"Calculate the nutrition facts for this list of ingredients: {ingredients}")
#         col2.info(output)
