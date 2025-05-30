import json
import uuid
import trafilatura
import requests
from duckduckgo_search import DDGS
from ollama import Client

ollama = Client()

def search_web(query: str) -> str:
    print(f"[TOOL] search_web() kaldt med query: {query}")
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        urls = [r['href'] for r in results]

    content_list = []
    for url in urls:
        try:
            print(f"[TOOL] Henter URL: {url}")
            html = requests.get(url, timeout=10).text
            clean_text = trafilatura.extract(html)
            if clean_text:
                print(f"[TOOL] Ekstraherede {len(clean_text)} tegn")
                content_list.append(clean_text)
        except Exception as e:
            print(f"[TOOL] Kunne ikke hente {url}: {e}")

    snippet = "\n\n".join(content_list[:2])
    print(f"[TOOL] Returnerer snippet på {len(snippet)} tegn")
    
    # Skriv snippet til fil
    with open("snippet.txt", "w", encoding="utf-8") as f:
        f.write(snippet)
        print("[TOOL] Skrev snippet til snippet.txt")
    
    return snippet

def chat_with_llm(question: str):
    print(f"[MAIN] Bruger spørger: {question}")
    messages = [{"role": "user", "content": question}]

    # Første kald: lad LLM beslutte om den vil bruge search_web
    print("[MAIN] Sender til LLM med tools=[search_web]")
    response = ollama.chat(
        model="llama3.3",
        messages=messages,
        tools=[search_web]
    )

    tool_calls = getattr(response.message, "tool_calls", []) or []
    if tool_calls:
        call = tool_calls[0]
        name = call.function.name
        args = call.function.arguments
        tool_id = str(uuid.uuid4())
        print(f"[MAIN] LLM vil kalde tool: {name} med args {args}")

        # Udfør værktøjet og skriv snippet til fil (gøres i search_web)
        result = search_web(**args)

        # Byg assistant-tool_call-besked
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args
                    }
                }
            ]
        })
        # Byg tool-svar
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": name,
            "content": result
        })

        # Skriv hele prompten til fil til inspektion
        with open("final_prompt.txt", "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
            print("[MAIN] Skrev prompt-beskeder til final_prompt.txt")

        # Andet kald: generér endeligt svar
        print("[MAIN] Sender tool-output tilbage til LLM for endeligt svar")
        final_response = ollama.chat(
            model="llama3.3",
            messages=messages
        )
        answer = final_response.message.content
    else:
        print("[MAIN] LLM svarede uden at kalde tool")
        answer = response.message.content

    print(f"[MAIN] Endeligt svar: {answer}")
    return answer

if __name__ == "__main__":
    query = input("Hvad vil du spørge om? ")
    svar = chat_with_llm(query)
    print("\n🤖 Svar fra LLM:\n", svar)