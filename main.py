import streamlit as st
import asyncio
import os
import time
import json
from datetime import datetime
from litellm import acompletion, completion_cost
from tavily import TavilyClient


# --- 1. SETUP & ASYNC HELPERS ---
def get_or_create_event_loop():
    """
    Streamlit runs in a dedicated thread. This helper ensures we use the
    existing event loop for that thread or create a new one to avoid
    'RuntimeError: There is no current event loop'.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise


st.set_page_config(page_title="Council 2026", layout="wide", page_icon="⚖️")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stTextArea textarea { font-size: 16px; }
    .stStatus { border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; }
    </style>
""", unsafe_allow_html=True)

# Create history directory
HISTORY_DIR = "research_history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# --- 2. DEFAULT CONFIGURATION ---
DEFAULT_PERSONAS = [
    {"name":   "The Skeptic",
     "prompt": "You are a critical skeptic. Scrutinize the search results for bias, logical fallacies, and missing data. Assume the provided data is misleading."},
    {"name":   "The Futurist",
     "prompt": "You are a visionary futurist. Extrapolate the long-term societal and economic implications of this topic 10 years into the future."},
    {"name":   "The Analyst",
     "prompt": "You are a data-driven pragmatist. Focus strictly on hard numbers, costs, and implementation feasibility. Ignore speculation."}
]

DEFAULT_MODELS = [
    {"model": "upstage/solar-pro-3:free", "assigned_personas": ["The Analyst"], "use_search": True},
    {"model": "arcee-ai/trinity-large-preview:free", "assigned_personas": ["The Skeptic"], "use_search": True},
]

# --- 3. SESSION STATE INITIALIZATION ---
if "active_models" not in st.session_state:
    st.session_state.active_models = DEFAULT_MODELS
if "available_personas" not in st.session_state:
    st.session_state.available_personas = DEFAULT_PERSONAS
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "tavily_key" not in st.session_state:
    st.session_state.tavily_key = ""
if "chair_model" not in st.session_state:
    st.session_state.chair_model = "arcee-ai/trinity-mini:free"


# --- 4. CORE ENGINE FUNCTIONS ---

async def perform_web_search(query, tavily_key):
    """
    Performs a single global search using Tavily to ground all models.
    """
    if not tavily_key:
        return None

    try:
        tavily = TavilyClient(api_key=tavily_key)
        # qna_search is optimized for direct answers suitable for LLM context
        response = await asyncio.to_thread(tavily.qna_search, query=query)
        return response
    except Exception as e:
        return f"Search Error: {str(e)}"


async def fetch_llm_response(model, system_prompt, user_prompt, api_key, meta_data):
    """
    Generic Async LLM caller using LiteLLM.
    meta_data contains {'index', 'persona_name', etc} for UI updates.
    """
    start_time = time.time()
    try:
        # Route to OpenRouter if not already specified
        full_model = f"openrouter/{model}" if "openrouter" not in model and "/" in model else model

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await acompletion(
            model=full_model,
            messages=messages,
            api_key=api_key
        )
        duration = time.time() - start_time

        # Safe cost calculation
        try:
            cost = completion_cost(response) or 0.0
        except:
            cost = 0.0

        return {
            "meta":    meta_data,
            "content": response.choices[0].message.content,
            "cost":    cost,
            "time":    duration,
            "success": True
        }
    except Exception as e:
        return {
            "meta":    meta_data,
            "content": f"Error: {str(e)}",
            "cost":    0.0,
            "time":    0.0,
            "success": False
        }


def save_report_to_disk(query, final_report, analysis_pass, results):
    """Saves the full session to a local Markdown file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join([c if c.isalnum() else "_" for c in query[:30]])
    filename = f"{HISTORY_DIR}/research_{timestamp}_{safe_query}.md"

    content = f"# Research Report: {query}\n"
    content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    content += f"## 🏆 Final Synthesis\n{final_report}\n\n"
    content += f"## 📊 Conflict Analysis (Internal)\n{analysis_pass}\n\n"
    content += "---\n## 🔍 Expert Deliberations\n"

    # Sort results by index
    results.sort(key=lambda x: x['meta']['index'])
    for res in results:
        meta = res['meta']
        status = "(Web Search Used)" if meta.get('used_search') else "(Internal Knowledge Only)"
        content += f"### Model: {meta['model']} | Persona: {meta['persona_name']} {status}\n{res['content']}\n\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename


# --- 5. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚖️ Council Configuration")

    # API KEYS
    with st.expander("🔑 API Keys", expanded=True):
        st.session_state.api_key = st.text_input("OpenRouter API Key", value=st.session_state.api_key, type="password")
        st.session_state.tavily_key = st.text_input("Tavily API Key (Optional)", value=st.session_state.tavily_key,
                                                    type="password", help="Needed for Deep Research")

    st.markdown("---")

    # MODEL MANAGEMENT
    st.subheader("🤖 Active Agents")

    persona_options = [p['name'] for p in st.session_state.available_personas]

    for i, model_entry in enumerate(st.session_state.active_models):
        with st.container(border=True):
            c1, c2 = st.columns([0.8, 0.2])
            with c1:
                st.markdown(f"**Agent Group {i + 1}**")
            with c2:
                if st.button("🗑️", key=f"del_m_{i}"):
                    st.session_state.active_models.pop(i)
                    st.rerun()

            # Model Inputs
            new_model = st.text_input("Model ID", model_entry['model'], key=f"m_name_{i}", label_visibility="collapsed",
                                      placeholder="Model ID")
            st.session_state.active_models[i]['model'] = new_model

            # Persona Selection
            selected = st.multiselect("Assign Personas", persona_options,
                                      default=model_entry.get('assigned_personas', []), key=f"m_pers_{i}")
            st.session_state.active_models[i]['assigned_personas'] = selected

            # Search Toggle
            use_search = st.checkbox("🌐 Enable Web Search", value=model_entry.get('use_search', True),
                                     key=f"search_{i}")
            st.session_state.active_models[i]['use_search'] = use_search

    if st.button("➕ Add New Agent Group", use_container_width=True):
        st.session_state.active_models.append({"model": "", "assigned_personas": [], "use_search": True})
        st.rerun()

    st.markdown("---")

    # ADVANCED SETTINGS
    with st.expander("⚙️ Advanced Settings"):
        st.session_state.chair_model = st.text_input("Synthesis Model", value=st.session_state.chair_model)

        st.markdown("### JSON Backup")
        config_export = {
            "api_key":            st.session_state.api_key,
            "tavily_key":         st.session_state.tavily_key,
            "chair_model":        st.session_state.chair_model,
            "active_models":      st.session_state.active_models,
            "available_personas": st.session_state.available_personas
        }
        st.download_button("Download Config", json.dumps(config_export, indent=4), "council_config.json",
                           "application/json", use_container_width=True)

        uploaded_file = st.file_uploader("Load Config", type=["json"])
        if uploaded_file and st.button("Apply Config"):
            try:
                data = json.load(uploaded_file)
                st.session_state.api_key = data.get("api_key", "")
                st.session_state.tavily_key = data.get("tavily_key", "")
                st.session_state.chair_model = data.get("chair_model", "")
                if "active_models" in data:
                    st.session_state.active_models = data["active_models"]
                if "available_personas" in data:
                    st.session_state.available_personas = data["available_personas"]
                st.rerun()
            except:
                st.error("Invalid JSON")

# --- 6. MAIN UI ---
st.title("Model Council")
st.markdown("#### Deep Research & Multi-Persona Synthesis")

user_query = st.text_area("Enter your research topic:", height=100,
                          placeholder="e.g. Impact of Quantum Computing on Cybersecurity in 2026")

if st.button("🚀 Run Council", use_container_width=True):
    if not st.session_state.api_key:
        st.error("⚠️ OpenRouter API Key is missing.")

    total_agents = sum(len(m['assigned_personas']) for m in st.session_state.active_models)
    if total_agents == 0:
        st.error("⚠️ No personas assigned to any models.")
    else:
        loop = get_or_create_event_loop()

        # PREPARE TASK MAP (Flatten Models * Personas into a list of tasks)
        task_map = []
        current_idx = 0
        st.markdown("### 📡 Live Deliberations")

        cols = st.columns(3)
        placeholders = []

        for m_entry in st.session_state.active_models:
            model_id = m_entry['model']
            use_search = m_entry.get('use_search', True)

            for p_name in m_entry['assigned_personas']:
                if not model_id:
                    continue
                p_prompt = next((p['prompt'] for p in st.session_state.available_personas if p['name'] == p_name), "")

                task_map.append({
                    "index":         current_idx,
                    "model":         model_id,
                    "persona_name":  p_name,
                    "system_prompt": p_prompt,
                    "use_search":    use_search
                })

                # Create UI Card
                with cols[current_idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{p_name}**")
                        st.caption(f"*{model_id}*")
                        if use_search:
                            st.caption("🌐 *Web Enabled*")
                        else:
                            st.caption("🧠 *Training Data Only*")

                        ph = st.empty()
                        ph.info("⏳ Queueing...")
                        placeholders.append(ph)

                current_idx += 1

        # STATUS BAR
        status_container = st.status("🚀 Initializing Council Protocol...", expanded=True)


        async def run_process():
            # 1. WEB SEARCH (Conditional)
            web_context = ""
            if st.session_state.tavily_key and any(t['use_search'] for t in task_map):
                status_container.update(label="🌍 Browsing the web (Tavily)...", state="running")
                raw_search = await perform_web_search(user_query, st.session_state.tavily_key)
                if raw_search:
                    web_context = f"### REAL-TIME WEB SEARCH RESULTS:\n{raw_search}\n\n"
                    st.toast("✅ Web Context Acquired")

            # 2. EXECUTION (Streaming Updates)
            status_container.update(label=f"📡 Experts are deliberating ({len(task_map)} threads)...", state="running")

            tasks = []
            for t in task_map:
                # Inject search results only if this specific model requested it
                prompt = f"{web_context}### USER QUERY:\n{user_query}" if t[
                    'use_search'] else f"### USER QUERY:\n{user_query}"

                tasks.append(
                    fetch_llm_response(t['model'], t['system_prompt'], prompt, st.session_state.api_key, t)
                )

            completed_results = []
            # Use as_completed to update UI instantly
            for future in asyncio.as_completed(tasks):
                result = await future
                completed_results.append(result)
                idx = result['meta']['index']

                with placeholders[idx].container():
                    if result['success']:
                        st.success(f"Done ({result['time']:.1f}s)")
                        with st.expander("View Output"):
                            st.markdown(result['content'])
                    else:
                        st.error("Failed")
                        st.error(result['content'])

            # 3. ANALYSIS PASS (Conflict Detection)
            status_container.update(label="🔍 Analyzing conflicts...", state="running")
            expert_dump = ""
            for r in completed_results:
                if r['success']:
                    src = "WEB SEARCH" if r['meta']['use_search'] else "INTERNAL KNOWLEDGE"
                    expert_dump += f"\n\n--- SOURCE: {r['meta']['model']} [{src}] | ROLE: {r['meta']['persona_name']} ---\n{r['content']}"

            analyst_prompt = f"""
            Analyze these findings. 
            QUERY: {user_query}
            DATA: {expert_dump}
            TASK: 
            1. List Consensus.
            2. List Conflicts.
            3. Highlight differences between Web-Search models vs Internal-Knowledge models.
            """

            analyst_res = await fetch_llm_response(st.session_state.chair_model, "You are a lead analyst.",
                                                   analyst_prompt, st.session_state.api_key, {"index": -1})

            # 4. WRITING PASS (Final Report)
            status_container.update(label="✍️ Finalizing report...", state="running")
            writer_prompt = f"""
            Write a final report. 
            QUERY: {user_query}
            ANALYSIS: {analyst_res['content']}
            DATA: {expert_dump}
            """

            final_res = await fetch_llm_response(st.session_state.chair_model, "You are an expert technical writer.",
                                                 writer_prompt, st.session_state.api_key, {"index": -2})

            status_container.update(label="✅ Research Complete", state="complete", expanded=False)
            return completed_results, analyst_res, final_res


        # Run the full pipeline
        results, analyst, final = loop.run_until_complete(run_process())

        # DISPLAY FINAL RESULTS
        st.divider()
        saved_path = save_report_to_disk(user_query, final['content'], analyst['content'], results)
        st.success(f"📄 Report saved to `{saved_path}`")

        tab1, tab2 = st.tabs(["🏆 Final Report", "📊 Conflict Analysis"])
        with tab1:
            st.markdown(final['content'])
        with tab2:
            st.info("Internal conflict analysis used to generate the final report.")
            st.markdown(analyst['content'])

        total_cost = sum(r['cost'] for r in results) + analyst['cost'] + final['cost']
        st.caption(f"Total Session Cost: ${total_cost:.5f}")