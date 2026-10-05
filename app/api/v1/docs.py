from fastapi import APIRouter, BackgroundTasks
import os
import requests
import json
import time

from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/karmansingh/Desktop/work/ai_interview/ai/.env")

router = APIRouter()

DOCS_DIR = "/Users/karmansingh/Desktop/work/ai_interview/backend/docs_content"

def _get_groq_key() -> str:
    from app.core.key_manager import key_manager
    return key_manager.get_groq_key() or os.environ.get("GROQ_API_KEY", "")

def _get_groq_model() -> str:
    from app.core.key_manager import key_manager
    return key_manager.get_groq_model()

def get_seo_topics():
    prompt = """
    You are an expert tech blog optimizer. Give me a JSON list of 7 deep Master Themes to document today.
    Mandatory master themes list: ["Backend Engineering", "Frontend Engineering", "Machine Learning", "Deep Learning", "Database", "Generative AI", "RAG Systems"]
    
    For EACH Master Theme:
    Provide 2 "subtopic_folders" (e.g., "SQL", "NoSQL" for Database, or "React", "Vue" for Frontend).
    Inside each sub_folder, provide 2 "inner_folders" (e.g., "State Management", "Performance").
    Inside those, provide 2 Articles each.
    
    Return STRICTLY this structure:
    [
       {
         "topic": "Database",
         "subtopic_folders": [
            {
               "folder": "SQL Databases",
               "inner_folders": [
                  {
                     "name": "Indexing & Querying",
                     "articles": [
                        {"name": "B-Tree Indexes in RDBMS", "slug": "db-btree-indexes"},
                        {"name": "Optimizing Joins in SQL", "slug": "db-sql-joins"}
                     ]
                  }
               ]
            }
         ]
       }
    ]
    """
    headers = {
        "Authorization": f"Bearer {_get_groq_key()}",
        "Content-Type": "application/json"
    }
    data = {
        "model": _get_groq_model(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        content = res.json()['choices'][0]['message']['content']
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        return json.loads(content)
    except Exception as e:
        print(f"Failed to fetch themes: {e}")
        try:
            print(f"Groq raw res: {res.text}")
        except:
             pass
        return []

def generate_seo_article(topic, subtopic):
    prompt = f"""
    Write a 2000-word, highly authoritative, detailed technical guide on '{subtopic}' for the category '{topic}'.
    The guide must act as premium documentation. 
    
    Structure it meticulously:
    1. **Meta Title:** [1 sentence SEO title]
       **Meta Description:** [2 sentence description]
       ---
    
    2. # {subtopic} Complete Guide
    
    3. ## 🔬 Introduction & Core Concepts
       Provide an absolute deep dive into WHAT this is, why it matters, and high-level architecture.
    
    4. ## 🛠️ Practical step-by-step Implementation
       Include at least 2 complete, copy-pasteable scripts (using ```language) containing setups, environments, and inline comments describing functionality.
    
    5. ## ⚖️ Scalability, Latency & Error Troubleshooting
       Dissect how this handles failure loads, limits, and mitigation strategies. At least 4 detailed subheadings here.
    
    6. ## 💡 Best Practices & Security Benchmarks
    
    Ensure perfect premium Markdown. Do NOT use HTML tags. Keep it looking expert-level, actionable engineering content.
    """
    headers = {
        "Authorization": f"Bearer {_get_groq_key()}",
        "Content-Type": "application/json"
    }
    data = {
        "model": _get_groq_model(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Failed article: {e}")
        return None

def run_daily_cron():
    print("🚀 Purging current docs and seeding from FastAPI background task...")
    
    import shutil
    if os.path.exists(DOCS_DIR):
        print("Purging current docs index folders...")
        for t in os.listdir(DOCS_DIR):
             t_path = os.path.join(DOCS_DIR, t)
             if os.path.isdir(t_path):
                 shutil.rmtree(t_path)

    topics_list = get_seo_topics()
    if not topics_list:
        print("No topics returned. Aborting.")
        return

    for item in topics_list:
        topic_name = item.get('topic', 'General')
        topic_path = os.path.join(DOCS_DIR, topic_name)

        if 'subtopic_folders' not in item:
            continue

        for group in item['subtopic_folders']:
            folder_name = group.get('folder', 'General')
            group_path = os.path.join(topic_path, folder_name)

            for inner in group.get('inner_folders', []):
                inner_name = inner.get('name', 'General')
                final_path = os.path.join(group_path, inner_name)
                os.makedirs(final_path, exist_ok=True)

                for art in inner.get('articles', []):
                    art_name = art['name']
                    slug_name = art['slug']
                    file_path = os.path.join(final_path, f"{slug_name}.md")

                    print(f"Generating for {topic_name} -> {folder_name} -> {inner_name} -> {art_name}")
                    content = generate_seo_article(topic_name, art_name)

                    if content:
                        with open(file_path, "w") as f:
                            f.write(content)
                    time.sleep(10)

# @router.post("/seed")
# def trigger_seeding(background_tasks: BackgroundTasks):
#     background_tasks.add_task(run_daily_cron)
#     return {"status": "success", "message": "Recursive volume Seeding operation pushed into background buffer tasks."}
