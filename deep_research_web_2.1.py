import os
import requests
from bs4 import BeautifulSoup
import gradio as gr

# --- API Clients ---
try:
    import openai
except ImportError:
    raise ImportError("Please install the openai package: pip install openai")

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Please install the google-generativeai package: pip install google-generativeai")

# --- Configuration ---
# It is recommended to set these as environment variables for security.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_FALLBACK_KEY")
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY", "YOUR_GEMINI_FALLBACK_KEY")

# --- Helper Functions ---
def fetch_web_content(url):
    """
    Fetches and parses web content from a URL using BeautifulSoup.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text content (simplified)
        return soup.get_text(separator="\n")
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def query_openai(prompt, model="gpt-4o-mini"):
    """
    Queries the OpenAI API (or a compatible API) using the new interface.
    """
    openai.api_key = OPENAI_API_KEY  # Set the API key directly
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant skilled in providing comprehensive and well-structured answers based on provided web content and user queries."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except openai.APIConnectionError as e:
        return f"OpenAI Connection Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred with OpenAI: {e}"

def query_gemini(prompt):
    """
    Queries the Google Gemini 2 API.
    """
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
    # Adjust model name as required by your Gemini API setup
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"An unexpected error occurred with Google Gemini: {e}"

# --- Core Research Function ---
def deep_research(query, api_choice, urls_to_scrape):
    """
    Performs deep research on a given query.
    
    🟢 *Parameters:*
      - **query:** The research question.
      - **api_choice:** "openai" or "gemini".
      - **urls_to_scrape:** Comma-separated URLs to scrape for additional context.
      
    🟢 *Process:*
      1. Scrapes provided URLs.
      2. Constructs a prompt combining the user query and scraped content.
      3. Queries the chosen API and returns a detailed response.
    """
    # 1. Web Scraping (if URLs are provided)
    scraped_content = ""
    if urls_to_scrape:
        url_list = [url.strip() for url in urls_to_scrape.split(",") if url.strip()]
        for url in url_list:
            content = fetch_web_content(url)
            scraped_content += f"\n\n--- Content from {url} ---\n\n{content}"

    # 2. Construct the prompt
    if scraped_content:
        combined_prompt = (
            f"Here is the user's research query: {query}\n\n"
            f"Below is some web content to assist your research. Use this as background context, "
            f"but do not repeat it verbatim in your answer:\n\n{scraped_content}\n\n"
            "Please provide a comprehensive, detailed, and well-structured answer."
        )
    else:
        combined_prompt = (
            f"Here is the user's research query: {query}\n\n"
            "Please provide a comprehensive, detailed, and well-structured answer."
        )

    # 3. Query the chosen API
    if api_choice.lower() == "openai":
        research_result = query_openai(combined_prompt)
    elif api_choice.lower() == "gemini":
        research_result = query_gemini(combined_prompt)
    else:
        research_result = "Invalid API choice. Please select either 'openai' or 'gemini'."

    return research_result

# --- Gradio Interface ---
if __name__ == '__main__':
    with gr.Blocks() as interface:
        gr.Markdown("### 🟢 AI-Powered Deep Research Tool")
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Research Query", placeholder="Enter your research query here...", lines=3)
                api_choice = gr.Radio(label="Choose API", choices=["openai", "gemini"], value="openai")
                urls_input = gr.Textbox(label="URLs to Scrape (comma-separated)", placeholder="e.g., https://example.com", lines=2)
                submit_button = gr.Button("Perform Research")
            with gr.Column():
                output_text = gr.Textbox(label="Research Results", placeholder="Your research results will appear here...", lines=15)
        
        submit_button.click(
            fn=deep_research,
            inputs=[query_input, api_choice, urls_input],
            outputs=output_text
        )
    
    # Launch the Gradio interface on all network interfaces (0.0.0.0) for broader accessibility.
    interface.launch(share=False, debug=True, server_name="0.0.0.0")
