https://chatgpt.com/share/67b172ae-d58c-800c-b348-553018499b21

Below is the updated version of the OpenAI query function along with a brief explanation. The migration guide indicates that the previous approach of creating a client with `openai.OpenAI(api_key=...)` is no longer supported. Instead, you should set the API key directly on the module (using `openai.api_key = ...`) and then call the new endpoint using `openai.ChatCompletion.create(...)`.

Here's the updated code snippet for the OpenAI query function:

```python
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
```

**Key changes and explanation:**

- **API Key Setting:**  
  Instead of creating a client instance, the API key is now set on the `openai` module using `openai.api_key = OPENAI_API_KEY`.

- **Direct API Call:**  
  The function now directly calls `openai.ChatCompletion.create(...)` rather than accessing a sub-object via a client. This adheres to the new interface in `openai>=1.0.0`.

- **Error Handling:**  
  The error handling remains similar, ensuring that connection issues and unexpected errors are captured and returned with a descriptive message.

Integrate this updated function into your overall code, and the error regarding the unsupported usage should be resolved. Let me know if you encounter any further issues!
