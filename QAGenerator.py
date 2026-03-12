import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# 1. Configuration & Authentication
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
input_path = 'data_jp/jp_data_summarized.jsonl'
output_path = 'data_jp/jp_data_with_questions.jsonl'

def generate_japanese_questions(content):
    """
    Sends summarize_content to OpenAI and retrieves 5 related questions 
    along with token usage data.
    """
    prompt = f"""
    以下の要約内容に基づいて、ユーザーが次に尋ねそうな関連質問を正確に5つ作成してください。
    出力は必ず以下のJSON形式のみで返してください：
    {{"questions": ["質問1", "質問2", "質問3", "質問4", "質問5"]}} 
    
    要約内容:
    {content}
    """
    #
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたは、ユーザーの入力に基づいて関連する質問を生成する専門のアシスタントです。必ずJSON形式で回答してください。"},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        res_data = json.loads(response.choices[0].message.content)
        questions = res_data.get("questions", [])[:5]
        usage = response.usage 
        
        return questions, usage
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return [], None

# 2. Main Execution
total_prompt_tokens = 0
total_completion_tokens = 0
total_all_tokens = 0
start_time = time.time()

print("--- Start processing ---")

try:
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            if not line.strip():
                continue
                
            data = json.loads(line)
            content = data.get('summarize_content', '')
            
            if content:
                # Add a small log every 10 rows so you know it hasn't frozen
                if (i + 1) % 10 == 0:
                    print(f"Processing row {i+1}...")
                
                questions, usage = generate_japanese_questions(content)
                data['related_questions'] = questions
                
                if usage:
                    total_prompt_tokens += usage.prompt_tokens
                    total_completion_tokens += usage.completion_tokens
                    total_all_tokens += usage.total_tokens
            
            # Write row-by-row to save memory
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

except FileNotFoundError:
    print(f"Error: The file at {input_path} was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# 3. Final Report
end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "="*40)
print("STATISTICS REPORT")
print(f"Execution time: {elapsed_time:.2f} seconds")
print("-" * 40)
print(f"Total Prompt Tokens:     {total_prompt_tokens}")
print(f"Total Completion Tokens: {total_completion_tokens}")
print(f"Total Combined Tokens:   {total_all_tokens}")
print("="*40)
print(f"Done! File saved to: {output_path}")