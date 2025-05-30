import json
import trafilatura
import requests
from duckduckgo_search import DDGS
from ollama import Client

ollama = Client()

# 🔍 Funktion som søger og henter ren tekst
def search_web(query: str) -> str:
    print(f"[!] Søger på nettet: {query}")
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        urls = [r['href'] for r in results]

    content_list = []
    for url in urls:
        try:
            html = requests.get(url, timeout=10).text
            clean_text = trafilatura.extract(html)
            if clean_text:
                content_list.append(clean_text)
        except Exception as e:
            print(f"Kunne ikke hente {url}: {e}")

    return "\n\n".join(content_list[:2])  # Begræns længde

# 🧰 Fortæl Ollama hvilke funktioner den må bruge
functions = [
    {
        "name": "search_web",
        "description": "Søger på nettet og returnerer nyttig tekst",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Det spørgsmål eller emne der skal søges efter"
                }
            },
            "required": ["query"]
        }
    }
]

# 🤖 Start en samtale med LLM
def chat_with_llm(question: str):
    messages = [{"role": "user", "content": question}]

    response = ollama.chat(
        model="llama3",
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    # 📌 Hvis modellen vil bruge en funktion
    if "function_call" in response:
        name = response["function_call"]["name"]
        args = json.loads(response["function_call"]["arguments"])

        if name == "search_web":
            result = search_web(args["query"])
            messages.append(response)  # tilføj modelens call
            messages.append({
                "role": "function",
                "name": name,
                "content": result
            })

            # 🔁 Giv svaret fra funktionen til LLM og få endeligt svar
            final_response = ollama.chat(
                model="llama3",
                messages=messages
            )
            return final_response["message"]["content"]
    else:
        return response["message"]["content"]

# ✅ Eksempel
if __name__ == "__main__":
    query = input("Hvad vil du spørge om? ")
    svar = chat_with_llm(query)
    print("\n🤖 Svar fra LLM:\n", svar)