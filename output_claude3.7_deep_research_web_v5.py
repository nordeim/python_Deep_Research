I'll dive deep into how an ideal deep research tool should function, explore the existing code, and then create an improved version.

## How a Good Deep Research Tool Should Work
```
<think>
```
To analyze how an ideal deep research tool should work, I need to understand the core aspects of effective research procedures. Let me break this down systematically:

1. Query analysis and planning
2. Search methodology and web crawling
3. Content evaluation and extraction
4. Information synthesis and summarization
5. Citation and sourcing
6. User experience considerations

For each component, I'll explore best practices and potential improvements.
```
</think>
```

A good deep research tool should function through these key steps:

1. **Query Analysis**: First understand the research question's scope, identify key concepts, and develop a structured research plan.

2. **Diversified Search Strategy**: Generate multiple search queries approaching the topic from different angles to overcome search engine biases and capture diverse perspectives.

3. **Intelligent Content Retrieval**: Prioritize authoritative sources while filtering out low-quality content, and handle various content formats effectively.

4. **Contextual Understanding**: Recognize the relevance of information to the research question, extracting meaningful insights rather than just text.

5. **Synthesis and Organization**: Organize findings thematically rather than just by source, identifying patterns, contradictions, and knowledge gaps.

6. **Iterative Refinement**: Use initial findings to inform subsequent searches, progressively diving deeper into the topic.

7. **Clear Attribution**: Maintain transparent source tracking with proper citations throughout the research process.

## Analysis of the Current Implementation
```
<think>
```
Now I'll examine the existing code to identify its strengths and limitations:

The current implementation includes:
- Environment setup and API integration (OpenAI and Gemini)
- Web content fetching using crawl4ai
- Search query generation
- Google search execution
- URL scraping functionality
- Research iteration logic
- Basic summarization

Let me analyze each component in detail to determine what needs improvement.
```
</think>
```

The current implementation has several strengths but also limitations:

- **Basic Query Processing**: No structured analysis of the initial research question
- **Limited Search Strategy**: Simple Google searches without filtering for source quality
- **Minimal Content Evaluation**: No prioritization of sources by relevance or credibility
- **Basic Summarization**: Simplistic content synthesis without thematic organization
- **Incomplete Citation**: URLs are included but without proper attribution
- **Error Handling**: Some basic handling but needs improvement for robustness
- **Interface**: The Gradio interface is incomplete in the provided code

## Creating an Improved Implementation

Based on this analysis, I'll create an enhanced version of the deep research tool with significant improvements in each area.

```python
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
import asyncio
import json
import re
import logging
import tldextract
from datetime import datetime
from crawl4ai import AsyncWebCrawler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Default models
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"

# Domain credibility ratings
CREDIBILITY_RATINGS = {
    "edu": 0.9,
    "gov": 0.9,
    "org": 0.7,
    "wikipedia.org": 0.75,
    "medium.com": 0.5,
    "blogspot.com": 0.3,
    "wordpress.com": 0.3,
}

# --- Helper Functions ---
async def fetch_web_content_crawl4ai(url):
    """Fetches web content using crawl4ai with improved error handling."""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if not result.markdown and not result.text:
                return {
                    "content": f"No content extracted from {url}", 
                    "error": True, 
                    "status": "No content"
                }
            
            # Extract metadata
            domain = tldextract.extract(url).registered_domain
            metadata = {
                "url": url,
                "domain": domain,
                "title": result.title if hasattr(result, 'title') else None,
                "timestamp": datetime.now().isoformat(),
                "credibility_score": get_domain_credibility_score(domain)
            }
            
            return {
                "content": result.markdown or result.text,
                "metadata": metadata,
                "error": False,
                "status": "Success"
            }
    except Exception as e:
        logger.error(f"Error fetching URL with crawl4ai: {url} - {str(e)}")
        return {
            "content": f"Error fetching URL: {str(e)}",
            "error": True,
            "status": str(e),
            "metadata": {"url": url}
        }

def get_domain_credibility_score(domain):
    """Estimates the credibility of a domain based on TLD and known sites."""
    # Check for exact domain match
    if domain in CREDIBILITY_RATINGS:
        return CREDIBILITY_RATINGS[domain]
    
    # Check TLD
    tld = domain.split('.')[-1] if '.' in domain else ''
    if tld in CREDIBILITY_RATINGS:
        return CREDIBILITY_RATINGS[tld]
    
    # Default score for unknown domains
    return 0.5

def query_openai(prompt, model=DEFAULT_OPENAI_MODEL, temperature=0.7, system_message=None):
    """Queries OpenAI using the client-based API with enhanced error handling."""
    if not system_message:
        system_message = "You are a helpful research assistant."
    
    start_time = time.time()
    retries = 0
    max_retries = 3
    
    while retries < max_retries:
        try:
            client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            response = completion.choices[0].message.content.strip()
            logger.info(f"OpenAI query completed in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            retries += 1
            wait_time = 2 ** retries  # Exponential backoff
            logger.warning(f"OpenAI API error (attempt {retries}/{max_retries}): {str(e)}")
            
            if "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                time.sleep(wait_time)
            elif retries < max_retries:
                logger.warning(f"Retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                return f"Error during OpenAI API call: {str(e)}"
    
    return "Failed to get response after maximum retries"

def query_gemini(prompt, model=DEFAULT_GEMINI_MODEL, temperature=0.7):
    """Queries Google Gemini with error handling."""
    start_time = time.time()
    retries = 0
    max_retries = 3
    
    while retries < max_retries:
        try:
            genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 64,
            }
            model_instance = genai.GenerativeModel(model, generation_config=generation_config)
            response = model_instance.generate_content(prompt)
            logger.info(f"Gemini query completed in {time.time() - start_time:.2f}s")
            return response.text.strip()
            
        except Exception as e:
            retries += 1
            wait_time = 2 ** retries  # Exponential backoff
            logger.warning(f"Gemini API error (attempt {retries}/{max_retries}): {str(e)}")
            
            if retries < max_retries:
                logger.warning(f"Retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                return f"An unexpected error occurred with Google Gemini: {e}"
    
    return "Failed to get response after maximum retries"

def analyze_research_query(query, llm_choice):
    """Analyzes the research query to identify key concepts and create a research plan."""
    prompt = f"""
    Please analyze the following research query: "{query}"
    
    1. Identify the main topics and key concepts that need to be researched
    2. Break down the query into subtopics or aspects that should be explored
    3. Suggest potential perspectives or angles to consider
    4. Identify any potential challenges in researching this topic
    5. Create a brief research plan with 3-5 main areas to focus on
    
    Format your response as a structured analysis with clear sections.
    """
    
    if llm_choice == "openai":
        response = query_openai(prompt, system_message="You are an expert research methodologist who helps plan comprehensive research strategies.")
    else:
        response = query_gemini(prompt)
    
    return response

def generate_search_queries(base_query, num_queries, llm_choice, research_analysis="", follow_up_questions=""):
    """Generates multiple search queries using the LLM with improved diversity."""
    prompt = f"""
    Generate {num_queries} different search queries related to the research topic: '{base_query}'
    
    Research analysis:
    {research_analysis}
    
    Follow-up questions to consider:
    {follow_up_questions}
    
    Guidelines for creating effective search queries:
    1. Make them diverse to cover different aspects of the topic
    2. Use different phrasings and synonyms to capture various results
    3. Include some specific, targeted queries and some broader queries
    4. Consider including queries that might surface contrasting viewpoints
    5. Include academic or technical terminology where appropriate
    6. Some queries should focus on finding recent information
    7. Some queries should focus on finding authoritative sources
    
    Return ONLY the numbered list of search queries, one query per line, without any additional text.
    """
    
    if llm_choice == "openai":
        response = query_openai(prompt, temperature=0.8)
    else:
        response = query_gemini(prompt, temperature=0.8)

    if response.startswith("Error") or "Failed to get response" in response:
        return [response]
    
    # Extract queries from the response
    queries = []
    lines = response.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove numbering and any extra characters
        cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
        if cleaned_line and cleaned_line not in queries:
            queries.append(cleaned_line)
    
    # Ensure we return the requested number of queries
    return queries[:num_queries]

def perform_google_search(query, num_results=8):
    """Performs a Google search and returns the top URLs with improved filtering."""
    try:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}&num={num_results}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract URLs
        search_results = soup.select('a[href^="/url?q="]')
        
        urls = []
        for result in search_results:
            url = urllib.parse.unquote(result['href'].replace('/url?q=', '').split('&')[0])
            
            # Filter out unwanted URLs
            if url and should_include_url(url):
                urls.append(url)
        
        # Remove duplicates
        unique_urls = []
        for url in urls:
            if url not in unique_urls:
                unique_urls.append(url)
                if len(unique_urls) >= num_results:
                    break
        
        return unique_urls

    except requests.exceptions.RequestException as e:
        logger.error(f"Error during Google search: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during search: {e}")
        return []

def should_include_url(url):
    """Determines if a URL should be included in research results."""
    # Skip ad results, shopping results, and certain low-value domains
    excluded_patterns = [
        'google.com/aclk',
        'google.com/shopping',
        'amazon.com/s?',
        'ebay.com/sch',
        'youtube.com/shorts',
        'instagram.com/p/',
        'pinterest.com/pin',
        'doubleclick.net',
        '/search?',
        'googleadservices',
    ]
    
    for pattern in excluded_patterns:
        if pattern in url:
            return False
    
    return True

async def scrape_urls_concurrently(urls):
    """Scrapes multiple URLs concurrently using crawl4ai with improved handling."""
    results = {}
    
    # Skip URLs that are obviously not going to work
    filtered_urls = [url for url in urls if is_valid_url(url)]
    
    if not filtered_urls:
        return results

    # Create tasks for all URLs
    tasks = []
    for url in filtered_urls:
        task = asyncio.create_task(fetch_web_content_crawl4ai(url))
        tasks.append((url, task))
    
    # Wait for all tasks to complete
    for url, task in tasks:
        try:
            result = await task
            results[url] = result
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            results[url] = {
                "content": f"Error: {str(e)}",
                "error": True,
                "status": str(e),
                "metadata": {"url": url}
            }
    
    return results

def is_valid_url(url):
    """Checks if a URL is valid and likely to be scrapable."""
    if not url.startswith(('http://', 'https://')):
        return False
        
    # Skip URLs that are likely to cause issues
    problematic_patterns = [
        '.pdf', '.jpg', '.png', '.gif', '.mp4', '.zip', '.doc', '.docx', '.xls', '.xlsx',
        'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com',
        'apple.com/itunes', 'play.google.com',
    ]
    
    for pattern in problematic_patterns:
        if pattern in url.lower():
            return False
            
    return True

def extract_relevant_content(scraped_data, query):
    """Extracts and prioritizes the most relevant content from scraped data."""
    extracted_results = []
    
    for url, data in scraped_data.items():
        if data.get("error", False):
            continue
            
        content = data.get("content", "")
        if not content or len(content) < 100:  # Skip very short content
            continue
            
        metadata = data.get("metadata", {"url": url})
        
        # Calculate relevance score based on query terms
        relevance_score = calculate_relevance(content, query)
        
        # Combine with credibility score
        credibility_score = metadata.get("credibility_score", 0.5)
        final_score = (relevance_score * 0.7) + (credibility_score * 0.3)
        
        # Truncate very long content
        max_content_length = 10000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"
        
        extracted_results.append({
            "url": url,
            "content": content,
            "metadata": metadata,
            "score": final_score
        })
    
    # Sort by score, highest first
    extracted_results.sort(key=lambda x: x["score"], reverse=True)
    
    return extracted_results

def calculate_relevance(content, query):
    """Calculate the relevance of content to the query."""
    query_terms = query.lower().split()
    content_lower = content.lower()
    
    # Count occurrences of query terms
    term_counts = {term: content_lower.count(term) for term in query_terms}
    
    # Calculate a simple relevance score
    score = sum(term_counts.values()) / (len(content_lower.split()) + 1)
    
    # Boost score if query terms appear early in the content
    for term in query_terms:
        if term in content_lower[:500]:
            score *= 1.2
    
    return min(score * 10, 1.0)  # Normalize to 0-1 range

def generate_content_summary(content_items, query, llm_choice):
    """Generates a comprehensive summary of the content with improved structure."""
    if not content_items:
        return "No relevant content was found to summarize."
        
    # Prepare the content for summarization
    combined_content = ""
    for i, item in enumerate(content_items[:5]):  # Focus on top 5 results
        source_info = f"Source {i+1}: {item['url']} (credibility: {item['metadata'].get('credibility_score', 'unknown')})"
        combined_content += f"\n\n--- {source_info} ---\n{item['content'][:8000]}"
    
    prompt = f"""
    Analyze and synthesize the following research content related to: "{query}"
    
    {combined_content}
    
    Create a comprehensive research summary that:
    1. Identifies the main findings and key points
    2. Organizes information thematically, not just by source
    3. Highlights areas of consensus across sources
    4. Notes contradictions or different perspectives
    5. Evaluates the strength of evidence for major claims
    6. Identifies any obvious gaps in the information
    
    Structure your summary with clear sections and proper attribution to sources when stating specific facts.
    """
    
    if llm_choice == "openai":
        summary = query_openai(prompt, temperature=0.3, system_message="You are an expert research analyst who excels at synthesizing information from multiple sources into coherent summaries.")
    else:
        summary = query_gemini(prompt, temperature=0.3)
    
    return summary

def generate_follow_up_questions(content_items, base_query, llm_choice):
    """Generates follow-up questions based on the research findings."""
    if not content_items:
        return "Unable to generate follow-up questions due to lack of content."
        
    # Prepare the content for analysis
    combined_content = ""
    for i, item in enumerate(content_items[:3]):  # Focus on top 3 results
        combined_content += f"\n\n--- Content from {item['url']} ---\n{item['content'][:5000]}"
    
    prompt = f"""
    Based on the following research content about "{base_query}":
    
    {combined_content}
    
    Generate 3-5 insightful follow-up questions that would:
    1. Address gaps in the current information
    2. Explore important aspects not covered in the existing content
    3. Help deepen understanding of contradictions or complex aspects
    4. Explore practical implications or applications
    5. Consider alternative perspectives or approaches
    
    Return ONLY the numbered list of questions, one per line, without any additional text.
    """
    
    if llm_choice == "openai":
        response = query_openai(prompt, temperature=0.7)
    else:
        response = query_gemini(prompt, temperature=0.7)
    
    # Clean up the response
    if response.startswith("Error") or "Failed to get response" in response:
        return response
        
    questions = []
    lines = response.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove numbering and any extra characters
        cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
        if cleaned_line and cleaned_line not in questions:
            questions.append(cleaned_line)
    
    return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

async def research_iteration(base_query, llm_choice, num_search_queries, research_analysis="", follow_up_questions=""):
    """Performs a single iteration of the research process with improved methodology."""
    # Generate search queries
    search_queries = generate_search_queries(base_query, num_search_queries, llm_choice, research_analysis, follow_up_questions)
    if not search_queries or search_queries[0].startswith("Error"):
        return {"error": "Failed to generate search queries.", "details": search_queries[0] if search_queries else "Unknown error"}

    # Collect all URLs from searches
    all_urls = []
    
    # Use concurrent.futures to perform searches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {executor.submit(perform_google_search, query): query for query in search_queries}
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                urls = future.result()
                all_urls.extend(urls)
                logger.info(f"Found {len(urls)} URLs for query: {query}")
            except Exception as exc:
                logger.error(f"Query {query} generated an exception: {exc}")
    
    # Remove duplicates while preserving order
    unique_urls = []
    for url in all_urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    if not unique_urls:
        return {"error": "No URLs were found from the search queries."}
    
    # Scrape content from URLs
    scraped_content = await scrape_urls_concurrently(unique_urls[:15])  # Limit to top 15 URLs
    
    # Extract and prioritize relevant content
    relevant_content = extract_relevant_content(scraped_content, base_query)
    
    if not relevant_content:
        return {"error": "No relevant content could be extracted from the URLs."}
    
    # Generate summary
    summary = generate_content_summary(relevant_content, base_query, llm_choice)
    
    # Generate follow-up questions
    new_follow_up_questions = generate_follow_up_questions(relevant_content, base_query, llm_choice)

    return {
        "search_queries": search_queries,
        "urls": unique_urls,
        "relevant_content": relevant_content,
        "summary": summary,
        "follow_up_questions": new_follow_up_questions,
    }

async def deep_research(query, llm_choice, depth, num_search_queries, urls_to_scrape=""):
    """Performs deep research with multiple iterations and improved methodology."""
    # Initialize the report
    report = f"# Deep Research Report: {query}\n\n"
    
    # Initial query analysis
    logger.info(f"Analyzing research query: {query}")
    research_analysis = analyze_research_query(query, llm_choice)
    report += f"## Research Analysis\n\n{research_analysis}\n\n"
    
    # Track all sources for final citation
    all_sources = []
    follow_up_questions = ""
    
    # Manual URL scraping (if provided)
    if urls_to_scrape:
        report += "## Analysis of Provided URLs\n\n"
        urls = [url.strip() for url in urls_to_scrape.split(",") if url.strip()]
        
        if urls:
            logger.info(f"Scraping {len(urls)} manually provided URLs")
            manual_scraped_content = await scrape_urls_concurrently(urls)
            
            # Extract and prioritize relevant content
            manual_relevant_content = extract_relevant_content(manual_scraped_content, query)
            
            if manual_relevant_content:
                # Add sources to tracking
                all_sources.extend([item["url"] for item in manual_relevant_content])
                
                # Generate summary of manual URLs
                initial_summary = generate_content_summary(manual_relevant_content, query, llm_choice)
                report += f"### Summary of Provided Sources\n\n{initial_summary}\n\n"
                
                # Generate initial follow-up questions
                follow_up_questions = generate_follow_up_questions(manual_relevant_content, query, llm_choice)
                report += f"### Initial Questions for Further Research\n\n{follow_up_questions}\n\n"
            else:
                report += "No relevant content could be extracted from the provided URLs.\n\n"
    
    # Iterative research
    for i in range(depth):
        logger.info(f"Starting research iteration {i+1} of {depth}")
        report += f"## Research Iteration {i+1}\n\n"
        
        # Perform research iteration
        iteration_results = await research_iteration(
            query, 
            llm_choice, 
            num_search_queries, 
            research_analysis, 
            follow_up_questions
        )

        if "error" in iteration_results:
            report += f"**Error:** {iteration_results['error']}\n\n"
            if "details" in iteration_results:
                report += f"**Details:** {iteration_results['details']}\n\n"
            break

        # Add sources to tracking
        if "relevant_content" in iteration_results:
            all_sources.extend([item["url"] for item in iteration_results["relevant_content"]])

        # Report search queries
        report += f"### Search Queries Used\n\n" + "\n".join([f"* {q}" for q in iteration_results['search_queries']]) + "\n\n"
        
        # Report findings
        report += f"### Key Findings\n\n{iteration_results['summary']}\n\n"
        
        # Update follow-up questions for next iteration
        follow_up_questions = iteration_results['follow_up_questions']
        report += f"### Follow-Up Questions\n\n{follow_up_questions}\n\n"
    
    # Final summary and synthesis
    if all_sources:
        report += "## Sources Referenced\n\n"
        for i, source in enumerate(all_sources):
            report += f"{i+1}. {source}\n"
        
        # Generate final synthesis if we have multiple iterations
        if depth > 1:
            final_synthesis_prompt = f"""
            Create a final synthesis of the research on "{query}" based on all the information collected.
            
            Focus on:
            1. The most important findings and insights
            2. How different sources or iterations complemented each other
            3. Any remaining uncertainties or areas for further research
            4. Practical implications or applications of the findings
            
            Keep your synthesis concise but comprehensive.
            """
            
            if llm_choice == "openai":
                final_synthesis = query_openai(final_synthesis_prompt, temperature=0.3, system_message="You are an expert research synthesizer who excels at creating final summaries of complex research topics.")
            else:
                final_synthesis = query_gemini(final_synthesis_prompt, temperature=0.3)
                
            report += f"\n## Final Synthesis\n\n{final_synthesis}\n\n"
    
    return report

# --- Gradio Interface ---
def gradio_research_handler(query, llm_choice, depth, num_queries, urls):
    """Non-async handler for the Gradio interface that runs the async function."""
    if not query:
        return "Please provide a research query."
    
    if not OPENAI_API_KEY and llm_choice == "openai":
        return "OpenAI API key is not set. Please check your .env file."
        
    if not GOOGLE_GEMINI_API_KEY and llm_choice == "gemini":
        return "Google Gemini API key is not set. Please check your .env file."
    
    try:
        # Convert inputs to appropriate types
        depth = int(depth)
        num_queries = int(num_queries)
        
        # Use asyncio.run to run the async function in a synchronous context
        result = asyncio.run(deep_research(query, llm_choice, depth, num_queries, urls))
        return result
    except Exception as e:
        logger.error(f"Error in research_handler: {str(e)}")
        return f"An error occurred while performing research: {str(e)}"

if __name__ == '__main__':
    with gr.Blocks(title="AI Deep Research Tool") as interface:
        gr.Markdown("# 🔍 AI-Powered Deep Research Tool")
        gr.Markdown("This tool performs comprehensive research on your topic using AI and web searches.")
        
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Research Query", 
                    placeholder="Enter your research topic or question here...", 
                    lines=3
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        llm_choice = gr.Radio(
                            label="AI Model", 
                            choices=["openai", "gemini"], 
                            value="openai"
                        )
                        
                    with gr.Column(scale=1):
                        depth_input = gr.Slider(
                            label="Research Depth (Iterations)", 
                            minimum=1, 
                            maximum=5, 
                            value=2, 
                            step=1
                        )
                
                num_queries_input = gr.Slider(
                    label="Search Queries per Iteration", 
                    minimum=3, 
                    maximum=15, 
                    value=5, 
                    step=1
                )
                
                urls_input = gr.Textbox(
                    label="Optional: Specific URLs to Analyze (comma-separated)", 
                    placeholder="https://example.com, https://another-site.org", 
                    lines=2
                )
                
                submit_button = gr.Button("Start Deep Research", variant="primary")
                
            with gr.Column(scale=2):
                output_text = gr.Markdown(
                    label="Research Report",
                    value="Your research results will appear here..."
                )
        
        submit_button.click(
            fn=gradio_research_handler, 
            inputs=[query_input, llm_choice, depth_input, num_queries_input, urls_input], 
            outputs=output_text
        )
        
        gr.Markdown("""
        ## How to Use
        1. Enter your research topic or question in the Research Query box
        2. Select your preferred AI model
        3. Adjust the research depth (more iterations = more comprehensive but slower)
        4. Optionally, provide specific URLs you want analyzed
        5. Click 'Start Deep Research' and wait for results
        
        ## Tips
        - Be specific in your query for more focused results
        - Higher depth values provide more thorough research but take longer
        - You can provide URLs you already know are relevant to include in the analysis
        """)
    
    # Launch the interface
    interface.launch(share=False)
```

Would you like me to explain or break down the code?
