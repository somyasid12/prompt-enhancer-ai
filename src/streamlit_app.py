import streamlit as st
from format_selector_agent import FormatSelectorAgent
from prompt_optimizer_agent import PromptOptimizerAgent


# Initialize agents

@st.cache_resource(show_spinner=False)
def load_agents():
    format_agent = FormatSelectorAgent()
    optimizer_agent = PromptOptimizerAgent()
    return format_agent, optimizer_agent

format_agent, optimizer_agent = load_agents()


# Streamlit UI

st.set_page_config(
    page_title="Prompt Enhancer",
    page_icon="✨",
    layout="centered",
)

st.title("✨ Prompt Enhancer")
st.markdown(
    """This tool takes a user query, detects the optimal prompt format, 
    and outputs a fully optimized prompt ready for an LLM."""
)

# User input
user_query = st.text_area("Enter your query here:", height=150)

if st.button("Enhance Prompt"):
    if not user_query.strip():
        st.warning("Please enter a query first!")
    else:
        with st.spinner("Processing..."):
            try:
                # Step 1: Format detection
                format_result = format_agent.select_format(user_query)
                
                # Step 2: Prompt optimization (FIXED - only pass format_result)
                optimized_result = optimizer_agent.optimize_prompt(format_result)
                
                # Display results
                st.subheader("Format Detection")
                st.json(format_result)
                
                st.subheader("Optimized Prompt")
                st.json(optimized_result)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your API key configuration.")
