https://chatgpt.com/share/67b172ae-d58c-800c-b348-553018499b21

Yes, you can absolutely update the original deep research app by replacing the old “query_openai” method with the latest client-based version. This updated method uses the new OpenAI Python interface (via the OpenAI client) and aligns with the sample code. Here’s an example of how you can integrate the new method into the deep research app:

```python
import os
import requests
from bs4 import BeautifulSoup
import gradio as gr
from openai import OpenAI  # New client-based API

# --- Configuration ---
# Set your API keys and base URL via environment variables for security.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_FALLBACK_KEY")
GOOGLE_GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY", "YOUR_GEMINI_FALLBACK_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

# --- Helper Functions ---
def fetch_web_content(url):
    """Fetches and parses web content from a URL using BeautifulSoup."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator="\n")
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def query_openai(prompt, model="gpt-4o-mini", temperature=0.8):
    """
    Queries the OpenAI API using the new client-based interface.
    """
    try:
        # Initialize the OpenAI client with the base URL and API key
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    except Exception as e:
        return f"Error initializing OpenAI client: {str(e)}"
    
    try:
        # Create the chat completion request
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant skilled in providing comprehensive and well-structured answers based on provided web content and user queries."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during API call: {str(e)}"

def query_gemini(prompt):
    """Queries the Google Gemini 2 API."""
    try:
        import google.generativeai as genai
    except ImportError:
        return "Please install the google-generativeai package: pip install google-generativeai"
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
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
    
    1. Web scraping: Extracts text from provided URLs.
    2. Prompt construction: Merges user query with scraped content.
    3. API Query: Uses either the updated OpenAI or Gemini function.
    """
    scraped_content = ""
    if urls_to_scrape:
        url_list = [url.strip() for url in urls_to_scrape.split(",") if url.strip()]
        for url in url_list:
            content = fetch_web_content(url)
            scraped_content += f"\n\n--- Content from {url} ---\n\n{content}"
    
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
    
    if api_choice.lower() == "openai":
        return query_openai(combined_prompt)
    elif api_choice.lower() == "gemini":
        return query_gemini(combined_prompt)
    else:
        return "Invalid API choice. Please select either 'openai' or 'gemini'."

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
        
        submit_button.click(fn=deep_research, inputs=[query_input, api_choice, urls_input], outputs=output_text)
    
    interface.launch(share=False, debug=True, server_name="0.0.0.0")
```

**Summary:**
- The updated `query_openai` function uses the new OpenAI client interface (as demonstrated in the sample) to create a chat completion.
- It sets up the client with a configurable base URL and API key.
- The rest of the deep research app remains unchanged, and you can seamlessly integrate this updated function.

Feel free to update your code with this version. This should resolve the error regarding the deprecated usage and allow your deep research app to function with the latest OpenAI API methods.
