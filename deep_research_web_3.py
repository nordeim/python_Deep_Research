import os
import requests
from bs4 import BeautifulSoup
import gradio as gr
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import concurrent.futures
import time
import urllib.parse

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# --- Helper Functions ---
def fetch_web_content(url):
    """Fetches and parses web content, handling errors robustly."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content more effectively, removing script and style tags
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()  # Remove these tags

        text = soup.get_text(separator="\n", strip=True)
        return text
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def query_openai(prompt, model="gpt-4-turbo-preview", temperature=0.7):
    """Queries OpenAI using the client-based API."""
    try:
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    except Exception as e:
        return f"Error initializing OpenAI client: {str(e)}"

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during OpenAI API call: {str(e)}"

def query_gemini(prompt):
    """Queries Google Gemini."""
    try:
        genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')  # Use gemini-pro
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"An unexpected error occurred with Google Gemini: {e}"

def generate_search_queries(base_query, num_queries, llm_choice, follow_up_questions=""):
    """Generates multiple search queries using the LLM."""
    prompt = (
        f"Generate {num_queries} different search queries related to: '{base_query}'. "
        f"Make them diverse and suitable for finding relevant information. "
        f"{follow_up_questions if follow_up_questions else ''} "
        "Return the queries as a numbered list, one query per line, without any additional text."
    )
    if llm_choice == "openai":
        response = query_openai(prompt)
    else:
        response = query_gemini(prompt)

    if response.startswith("Error"):
        return [response]
    queries = response.split("\n")  # Split into individual queries
    # Remove numbering and any extra whitespace
    cleaned_queries = [q.strip().replace(f"{i+1}.", "").strip() for i, q in enumerate(queries) if q.strip()]

    return cleaned_queries[:num_queries]  # Ensure we don't return more than requested

def perform_google_search(query):
    """Performs a Google search and returns the top URLs."""
    try:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract URLs - Updated to use a more robust selector
        search_results = soup.select('a[href^="/url?q="]')
        urls = [urllib.parse.urlparse(urllib.parse.unquote(result['href'].replace('/url?q=', '').split('&')[0])).netloc for result in search_results]

        # Remove duplicates and empty strings, then return only the top 5 unique URLs
        unique_urls = []
        for url in urls:
            if url and url not in unique_urls:
                unique_urls.append(url)
            if len(unique_urls) == 5:
                break
        return unique_urls

    except requests.exceptions.RequestException as e:
        print(f"Error during Google search: {e}")
        return []

def scrape_urls_concurrently(urls):
    """Scrapes multiple URLs concurrently."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        future_to_url = {executor.submit(fetch_web_content, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error: {exc}"
    return results

def research_iteration(base_query, llm_choice, num_search_queries, follow_up_questions=""):
    """Performs a single iteration of the research process."""
    search_queries = generate_search_queries(base_query, num_search_queries, llm_choice, follow_up_questions)
    if not search_queries or search_queries[0].startswith("Error"):
        return {"error": "Failed to generate search queries."}

    all_scraped_content = {}

    # Use concurrent.futures to perform searches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {executor.submit(perform_google_search, query): query for query in search_queries}
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                urls = future.result()
                scraped_content = scrape_urls_concurrently(urls)
                all_scraped_content.update(scraped_content)
            except Exception as exc:
                print(f"Query {query} generated an exception: {exc}")

    # Combine scraped content for LLM summary
    combined_content = "\n\n".join([f"--- Content from {url} ---\n{content}" for url, content in all_scraped_content.items()])

     # Generate follow-up questions
    follow_up_prompt = (
        f"Based on the following content, generate 3-5 follow-up research questions to further explore the topic: '{base_query}'.\n\n"
        f"{combined_content[:10000] if combined_content else 'No content available.'}\n\n"  # Limit content length
        "Return the questions as a numbered list, one question per line, without any additional text."
    )
    if llm_choice == "openai":
        new_follow_up_questions = query_openai(follow_up_prompt)
        summary = query_openai(f"Summarize the following content concisely, focusing on key insights related to '{base_query}':\n\n{combined_content[:10000]}")
    else:
        new_follow_up_questions = query_gemini(follow_up_prompt)
        summary = query_gemini(f"Summarize the following content concisely, focusing on key insights related to '{base_query}':\n\n{combined_content[:10000]}")


    return {
        "search_queries": search_queries,
        "scraped_content": all_scraped_content,
        "summary": summary,
        "follow_up_questions": new_follow_up_questions,
    }

def deep_research(query, llm_choice, depth, num_search_queries, urls_to_scrape=""):
    """Performs deep research with multiple iterations."""
    # Initialize the report
    report = f"# Deep Research Report: {query}\n\n"
    follow_up_questions = ""

    # Manual URL scraping (if provided)
    if urls_to_scrape:
        urls = [url.strip() for url in urls_to_scrape.split(",") if url.strip()]
        manual_scraped_content = scrape_urls_concurrently(urls)
        report += "## Manually Provided URLs\n\n"
        for url, content in manual_scraped_content.items():
            report += f"### Content from {url}\n\n{content}\n\n"

        # Initial summary of manual URLs
        if llm_choice == "openai":
            initial_summary = query_openai(f"Summarize the following content concisely, focusing on key insights related to '{query}':\n\n" +
                                          "\n\n".join(manual_scraped_content.values()))
        else:
            initial_summary = query_gemini(f"Summarize the following content concisely, focusing on key insights related to '{query}':\n\n" +
                                         "\n\n".join(manual_scraped_content.values()))

        report += f"## Initial Summary of Manually Provided URLs\n\n{initial_summary}\n\n"

    # Iterative research
    for i in range(depth):
        report += f"## Iteration {i+1}\n\n"
        iteration_results = research_iteration(query, llm_choice, num_search_queries, follow_up_questions)

        if "error" in iteration_results:
            report += f"**Error:** {iteration_results['error']}\n\n"
            break

        report += f"### Search Queries Used:\n\n" + "\n".join([f"*   {q}" for q in iteration_results['search_queries']]) + "\n\n"

        report += f"### Scraped Content Summary:\n\n{iteration_results['summary']}\n\n"
        report += f"### Follow-Up Questions:\n\n{iteration_results['follow_up_questions']}\n\n"

        follow_up_questions = iteration_results['follow_up_questions']  # Prepare for next iteration

    return report

# --- Gradio Interface ---
if __name__ == '__main__':
    with gr.Blocks() as interface:
        gr.Markdown("### 🟢 AI-Powered Deep Research Tool")
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Research Query", placeholder="Enter your research query here...", lines=3)
                llm_choice = gr.Radio(label="Choose LLM", choices=["openai", "gemini"], value="openai")
                depth_input = gr.Slider(label="Depth (Iterations)", minimum=1, maximum=5, value=2, step=1)
                num_queries_input = gr.Slider(label="Number of Search Queries per Iteration", minimum=1, maximum=10, value=3, step=1)
                urls_input = gr.Textbox(label="URLs to Scrape (comma-separated)", placeholder="e.g., https://example.com", lines=2)

                submit_button = gr.Button("Perform Research")
            with gr.Column():
                output_text = gr.Textbox(label="Research Results", placeholder="Your research results will appear here...", lines=15)

        submit_button.click(
            fn=deep_research,
            inputs=[query_input, llm_choice, depth_input, num_queries_input, urls_input],
            outputs=output_text
        )

    interface.launch(share=False, debug=True, server_name="0.0.0.0")
  
