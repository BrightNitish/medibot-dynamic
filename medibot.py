import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables instantly
load_dotenv()

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
from openai import OpenAI

# ==========================================
# SETUP NVIDIA API CLIENT
# ==========================================
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") 

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY 
)

# ==========================================
# PURE REAL-TIME PUBMED GLOBAL API (FIXED)
# ==========================================
def fetch_pubmed_realtime(query, max_results=5): # Fetching 5 to guarantee we get good abstracts
    try:
        # Proper URL encoding with PubMed's official abstract filter
        refined_query = f"{query} AND hasabstract[text]"
        safe_query = urllib.parse.quote(refined_query)
        
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={safe_query}&retmode=json&retmax={max_results}"
        
        with urllib.request.urlopen(search_url) as response:
            data = json.loads(response.read().decode())
            id_list = data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return "", {}
        
        id_str = ",".join(id_list)
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_str}&retmode=xml"
        
        with urllib.request.urlopen(fetch_url) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        live_context = ""
        live_sources = {}
        
        valid_count = 0
        for article in root.findall('.//PubmedArticle'):
            if valid_count >= 3: # Keep context size manageable for LLaMA
                break
                
            title = article.find('.//ArticleTitle')
            title_text = title.text if title is not None else "No Title"
            
            abstract_texts = article.findall('.//AbstractText')
            abstract_content = " ".join([elem.text for elem in abstract_texts if elem.text])
            
            pmid = article.find('.//PMID')
            if pmid is not None and abstract_content:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid.text}/"
                live_context += f"Paper Title: {title_text}\nFindings: {abstract_content}\n\n"
                live_sources[url] = title_text
                valid_count += 1
                
        return live_context, live_sources
    except Exception as e:
        print(f"PubMed API Error: {e}") # This will show error in terminal if it fails
        return "", {}

# ==========================================
# STREAMLIT APPLICATION INTERFACE
# ==========================================
def main():
    st.set_page_config(page_title="MediBot AI Engine", page_icon="🩺", layout="centered")
    
    # Clean UI Title Setup
    st.title("MediBot 🩺") 
    st.markdown("### *Dynamic Clinical Evidence Synthesis Engine*")
    st.write("---")

    # Chat history session storage allocation
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am **MediBot**. Enter any medical condition or query, and I will search **PubMed live** to generate an evidence-grounded response with dynamic citations. How can I assist you today?"}
        ]

    # Render conversations cleanly
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🩺"
        st.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

    # Core user interface action block
    if prompt := st.chat_input("Ask a medical question..."):
        st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("🔬 Querying live PubMed Central databases & synthesizing insights..."):
            context = ""
            sources_dict = {}

            # Pure Live Global PubMed Extraction (No Local FAISS interference anymore!)
            live_context, live_sources = fetch_pubmed_realtime(prompt, max_results=3)
            if live_context:
                context += "Live Academic Publications:\n" + live_context
                sources_dict.update(live_sources)

            # Synthesizer Engine Prompting
            if not context.strip() or len(sources_dict) == 0:
                final_output = "❌ **Evidence Not Found:** I cannot find sufficient verified clinical data or papers for this query in current digital medical publication registries."
            else:
                final_prompt = f"""
                You are a senior clinical research AI. Synthesize a reader-friendly, highly professional, and structured answer based ONLY on the provided context below.
                
                CRITICAL VISUAL & TEXTUAL RULES:
                1. Split your answer into clear logical sections using bold headers.
                2. Use clean markdown bullet points to make the data instantly readable. Avoid massive blocks of texts.
                3. Do NOT say things like "Based on the provided abstract" or "In the paper titled...". Talk directly about the clinical findings.
                4. If the context does not contain relevant insights to answer the question, state exactly: "I cannot find sufficient verified clinical evidence for this query in recent PubMed publications."
                5. Never fabricate data. Keep the tone completely objective and scientific.

                Context Data:
                {context}

                User Query: {prompt}

                Clinical Synthesis:
                """

                try:
                    response = client.chat.completions.create(
                        model="meta/llama-3.1-8b-instruct", 
                        messages=[{"role": "user", "content": final_prompt}],
                        temperature=0.0
                    )
                    result_text = response.choices[0].message.content
                    
                    if "I cannot find sufficient verified clinical evidence" in result_text:
                        final_output = "❌ **Evidence Not Found:** I cannot find sufficient verified clinical data for this query in recent PubMed publications."
                    else:
                        # Append source references cleanly with proper formatting
                        link_section = "\n\n---\n### 🔗 Verified Clinical Literature Sources:\n"
                        for url, title in sources_dict.items():
                            link_section += f"- [{title}]({url})\n"
                        final_output = result_text + link_section
                except Exception as e:
                    final_output = f"⚠️ Inference API Connection Timeout: {e}"

        # Push clean markdown to the UI interface layer
        st.chat_message('assistant', avatar="🩺").markdown(final_output)
        st.session_state.messages.append({'role': 'assistant', 'content': final_output})

if __name__ == "__main__":
    main()