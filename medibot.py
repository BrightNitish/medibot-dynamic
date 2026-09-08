import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ==========================================
# 1. SETUP NVIDIA API CLIENT
# ==========================================
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# ==========================================
# 2. AGENTIC QUERY OPTIMIZER (NEMOTRON)
# ==========================================
def optimize_pubmed_query(user_query, client):
    """
    Agentic layer converting natural language into PubMed search terms.
    Strips Nemotron reasoning tokens to prevent search failures.
    """
    system_prompt = (
        "You are an API middleware that outputs raw search tokens for PubMed. "
        "Strict Rule: Do NOT explain your thought process. Do NOT include 'Thinking process' or greetings. "
        "Remove all conversational filler, stopwords, and punctuation. "
        "Extract ONLY core medical entities, conditions, demographics, and drugs. "
        "Join entities with ' AND '. "
        "Example output: Type 1 Diabetes AND Type 2 Diabetes AND Diagnosis\n"
        "Return EXCLUSIVELY the keyword query string."
    )

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract PubMed query keywords for: {user_query}"}
            ],
            temperature=0.0,
            max_tokens=60
        )
        raw_output = response.choices[0].message.content.strip()

        # Discard reasoning lines/tokens from Nemotron
        lines = [line.strip() for line in raw_output.split("\n") if line.strip()]
        filtered_lines = [
            l for l in lines
            if not any(stop in l.lower() for stop in ["thinking", "analyze", "user query", "goal:", "here is", "output:"])
        ]

        optimized_query = filtered_lines[-1] if filtered_lines else raw_output
        return optimized_query.replace('"', '').replace("'", "")

    except Exception as e:
        print(f"Optimizer fallback: {e}")
        return user_query

# ==========================================
# 3. REAL-TIME PUBMED E-UTILITIES
# ==========================================
def fetch_pubmed_realtime(query, max_results=3):
    try:
        refined_query = f"{query} AND hasabstract[text]"
        safe_query = urllib.parse.quote(refined_query)

        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={safe_query}&retmode=json&retmax={max_results}"

        req = urllib.request.Request(search_url, headers={'User-Agent': 'MediBot/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            id_list = data.get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return "", {}

        id_str = ",".join(id_list)
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_str}&retmode=xml"

        fetch_req = urllib.request.Request(fetch_url, headers={'User-Agent': 'MediBot/1.0'})
        with urllib.request.urlopen(fetch_req) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        live_context = ""
        live_sources = {}

        for article in root.findall('.//PubmedArticle'):
            title = article.find('.//ArticleTitle')
            title_text = title.text if title is not None else "No Title"

            abstract_texts = article.findall('.//AbstractText')
            abstract_content = " ".join([elem.text for elem in abstract_texts if elem.text])

            pmid = article.find('.//PMID')
            if pmid is not None and abstract_content:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid.text}/"
                live_context += f"Paper Title: {title_text}\nFindings: {abstract_content}\n\n"
                live_sources[url] = title_text

        return live_context, live_sources
    except Exception as e:
        print(f"PubMed API Error: {e}")
        return "", {}

# ==========================================
# 4. STREAMLIT UI & INTERFACE
# ==========================================
def main():
    st.set_page_config(page_title="MediBot AI Engine", page_icon="🩺", layout="centered")

    # Theming & CSS
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, 0.2); }
        [data-testid="stChatInput"] { border-radius: 20px !important; border: 1px solid rgba(128, 128, 128, 0.3) !important; }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #3b82f6;'>MediBot 🩺</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8; font-size: 1.1em;'>Dynamic Clinical Evidence Synthesis Engine</p>", unsafe_allow_html=True)
    st.divider()

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
        st.header("💡 Live Capabilities")
        st.markdown("""
        - 🌐 **Real-Time PubMed Queries**
        - 🧬 **Agentic Query Optimization**
        - 📑 **Dynamic Citation Extraction**
        """)
        st.divider()
        st.info("📚 Connected to NCBI Entrez E-Utilities API.")
        st.caption("Developed for educational & clinical research evaluation.")

    # Chat Memory
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am **MediBot**. Ask any medical query, and I will search **PubMed live** to generate an evidence-grounded response with dynamic citations."}
        ]

    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🩺"
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Ask a medical question..."):
        st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("🔬 Optimizing search & retrieving real-time PubMed records..."):
            optimized_keywords = optimize_pubmed_query(prompt, client)
            st.info(f"🔍 **Optimized Agentic Search:** `{optimized_keywords}`")

            live_context, sources_dict = fetch_pubmed_realtime(optimized_keywords, max_results=3)

            if not live_context.strip() or not sources_dict:
                final_output = "❌ **Evidence Not Found:** I cannot find sufficient verified clinical data or papers for this query in current digital medical publication registries."
            else:
                final_prompt = f"""
                You are a senior clinical research AI. Synthesize a professional, structured answer using ONLY the context provided below.

                Rules:
                1. Split into logical sections using bold titles.
                2. Use clear bullet points. Avoid walls of text.
                3. Do not say "Based on the provided abstract". State clinical findings directly.
                4. Maintain an objective, scientific tone. Never fabricate data.

                Context:
                {live_context}

                User Query: {prompt}

                Clinical Synthesis:
                """

                try:
                    response = client.chat.completions.create(
                        model="nvidia/nemotron-3.5-lightning-30b-a3b",
                        messages=[{"role": "user", "content": final_prompt}],
                        temperature=0.0
                    )
                    result_text = response.choices[0].message.content

                    # Format dynamic clickable citations
                    link_section = "\n\n---\n### 🔗 Verified Clinical Literature Sources:\n"
                    for url, title in sources_dict.items():
                        link_section += f"- [{title}]({url})\n"

                    final_output = result_text + link_section
                except Exception as e:
                    final_output = f"⚠️ Inference API Connection Timeout: {e}"

        st.chat_message("assistant", avatar="🩺").markdown(final_output)
        st.session_state.messages.append({"role": "assistant", "content": final_output})

if __name__ == "__main__":
    main()