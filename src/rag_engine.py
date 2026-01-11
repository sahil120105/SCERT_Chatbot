# src/rag_engine.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class MarathiRAG:
    def __init__(self, db_path=r"qdrant_db", collection_name="scert_bot"):

        base_dir = Path(__file__).resolve().parent
        paf = base_dir / ".." / db_path

        # Load the same model used in ingestion
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')
        self.client = QdrantClient(path=paf)
        self.collection = collection_name
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def retrieve(self, query, top_k=20):
        """Searches the vector DB for relevant chunks."""
        # 'query:' prefix is required for E5 models
        query_vector = self.embedder.encode(f"query: {query}").tolist()
        
        # Use query_points (Robust Fallback)
        # This is the direct API call which is less likely to break than .search()
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k
        ).points  # Note: We access .points attribute here

        return results

    def generate_answer(self, user_query):
        """Orchestrates the RAG flow."""
        
        # 1. Retrieve Context
        hits = self.retrieve(user_query)
        
        if not hits:
            return "Maaf kara, mala yabadal mahiti sapadli nahi. (Sorry, I couldn't find info on this.)", []

        # 2. Build Context String & Collect Sources
        context_parts = []
        sources = []
        unique_urls = set()

        for hit in hits:
            url = hit.payload.get('source_url', '#')
            text = hit.payload.get('text', '')
            section = hit.payload.get('context', '')
            
            # Format: [Source URL] Content...
            context_parts.append(f"Source: {url}\nSection: {section}\nContent: {text}")
            
            if url not in unique_urls and url != "https://www.maa.ac.in/":
                unique_urls.add(url)
                sources.append(url)

        context_str = "\n\n---\n\n".join(context_parts)

        # 3. Construct the Prompt
        # We explicitly tell Gemini to handle the mixed language.
        system_prompt = f"""
        You are an official academic assistant for **SCERT Maharashtra** (State Council of Educational Research and Training).
        Your goal is to help students, teachers, and officials find accurate information from the website.

        ### GUIDELINES:
        1. **Direct Answer:** Start answering the question immediately. **NEVER** say "Based on the context", "According to the document", or "The provided text says". Just state the facts.
        2. **Tone:** Be professional, polite, and official. Use clear, simple language.
        3. **Structure:** - Use **bullet points** for lists (dates, fees, names).
           - Use **Bold** for key entities like Names, Phone Numbers, and Deadlines.
           - If the answer is long, break it into small paragraphs.
        4. **Language:** Strict strict adherence to the user's language. 
           - If User asks in **Marathi**, answer in **Marathi**.
           - If User asks in **English**, answer in **English**.
           - If User asks in **Hinglish**, answer in **natural mixed Marathi/English**.
        5. **Missing Info:** If the answer is NOT in the 'Background Information' below, politely say: "Sorry, I currently do not have that specific information. Please check the official contact section." (Translate this to Marathi if needed). Do NOT make up facts.
        6. **Links:** If you mention a form, PDF, or page, explicitly tell the user to "Click the link below" (The system will handle the actual URL display, you just reference it).
        7. **LANGUAGE MATCHING (HIGHEST PRIORITY):** - **DETECT** the language of the `USER QUESTION` below.
           - **ANSWER ONLY** in that **EXACT SAME LANGUAGE**.
           - **SCENARIO:** If the user asks in **English**, but the `BACKGROUND INFORMATION` is in **Marathi**, you **MUST TRANSLATE** the facts into English for your answer. Do not output Marathi text to an English user.
        8. **GUIDE TO RELEVANT PAGES:** If user asks for some information or where to find something guide him/her to the relevant page on the SCERT website using the source URL.
        ### BACKGROUND INFORMATION:
        {context_str}

        ### USER QUESTION:
        {user_query}
        """

        # 4. Generate Response
        response = self.model.generate_content(system_prompt)
        return response.text, sources