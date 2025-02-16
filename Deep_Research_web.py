import gradio as gr
import openai
import google.generativeai as genai
import os
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
# Set API Keys as environment variables (recommended for security)
# For example, in your terminal:
# export OPENAI_API_KEY="your_openai_key"
# export GOOGLE_GEMINI_API_KEY="your_gemini_key"

# Fallback keys (not recommended for production, use environment variables!)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_FALLBACK_KEY")
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY", "YOUR_GEMINI_FALLBACK_KEY")

# --- Helper Functions ---

def fetch_web_content(url):
    """Fetches and parses web content from a URL using BeautifulSoup."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text content - this is a simplified example
        text_content = soup.get_text()
        return text_content
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def query_openai(prompt, model="gpt-4o-mini"):
    """Queries the OpenAI API (or a compatible API)."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
      response = client.chat.completions.create(
        model=model,
        messages=[
          {"role": "system", "content": "You are a helpful research assistant, skilled in providing comprehensive and well-structured answers based on provided web content and user queries."},
          {"role": "user", "content": prompt}
        ]
      )
      return response.choices[0].message.content
    except openai.APIConnectionError as e:
        return f"OpenAI Connection Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred with OpenAI: {e}"

def query_gemini(prompt):
    """Queries the Google Gemini Pro API."""
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An unexpected error occurred with Google Gemini: {e}"

# --- Core Research Function ---

def deep_research(query, api_choice, urls_to_scrape):
    """
    Performs deep research on a given query.

    Args:
        query: The research question.
        api_choice:  "openai" or "gemini".
        urls_to_scrape: A comma-separated string of URLs to scrape.

    Returns:
        A string containing the research findings.
    """
    # 1. Web Scraping (if URLs are provided)
    scraped_content = ""
    if urls_to_scrape:
        url_list = [url.strip() for url in urls_to_scrape.split(",")]
        for url in url_list:
            content = fetch_web_content(url)
            scraped_content += f"\n\n--- Content from {url} ---\n\n" + content

    # 2. Construct the prompt (combine query and scraped content)
    if scraped_content:
      combined_prompt = f"""
      Here is the user's research query:
      {query}

      Here is some relevant content I found from the web, to assist with the research for answering user's question, do not repeat the contents here verbatim but only use it as additional information to assist with your research:
      {scraped_content}

      Please provide a comprehensive and well-structured answer to the research query, it is important to consider web content to provide the most up-to-date answer:
      """
    else:
      combined_prompt = f"""
        Here is the user's research query:
        {query}
        
        Please provide a comprehensive and well-structured answer to the research query.
      """

    # 3. Query the chosen API
    if api_choice == "openai":
        research_result = query_openai(combined_prompt)
    elif api_choice == "gemini":
        research_result = query_gemini(combined_prompt)
    else:
        return "Invalid API choice."

    return research_result
# --- Gradio Interface ---

if __name__ == '__main__':
    with gr.Blocks() as interface:
        gr.Markdown("# AI-Powered Deep Research Tool")
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Research Query", lines=3)
                api_choice = gr.Radio(label="Choose API", choices=["openai", "gemini"], value="openai")
                urls_input = gr.Textbox(label="URLs to Scrape (comma-separated)", lines=2)
                submit_button = gr.Button("Perform Research")
            with gr.Column():
                output_text = gr.Textbox(label="Research Results", lines=10)

        submit_button.click(
            deep_research,
            inputs=[query_input, api_choice, urls_input],
            outputs=output_text
        )

    interface.launch(share=False, debug=True, server_name="0.0.0.0")