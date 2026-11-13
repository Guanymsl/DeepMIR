import csv
import json
import time
import os

from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from common import read_captions

MODEL = 'gpt-4.1'
CAPTIONS_PATH = Path('test.csv')
OUTPUT_DIR = Path('outputs/prompts')
RETRIES = 5

CONCISE_PROMPT = '''
You are an expert music caption editor.

Rewrite the provided music caption into a concise,
information-dense prompt for text-to-music generation.

Requirements:
- Preserve the original genre, instruments, vocals,
  rhythm, timbre, mood, and musical relationships.
- Remove redundant adjectives, repeated descriptions,
  vague metaphors, and non-audible contextual details.
- Prioritize concrete, audible musical information.
- Preserve temporal events and changes when specified.
- Do not introduce new musical details.
- Do not replace specific instruments with generic terms.
- Do not change the original meaning or certainty.
- Use one clear, natural English sentence when possible.

Return only the rewritten caption.
'''

DETAILED_PROMPT = '''
You are an expert in descriptive music captioning.

Rewrite the provided caption into a detailed, precise,
and coherent prompt for text-to-music generation.

This method is inspired by descriptive recaptioning.

Requirements:
- Preserve all supported musical information.
- Clearly describe genre and overall musical character.
- Describe the instruments and their individual roles.
- Describe rhythm, groove, tempo, and rhythmic patterns
  only when explicitly supported.
- Describe timbre, dynamics, texture, and atmosphere
  using evidence from the original caption.
- Clarify relationships between musical elements.
- Preserve temporal progression and transitions.
- Organize information into a coherent description.
- Do not invent instruments, vocals, BPM, genre,
  production techniques, or musical events.
- Do not expand generic terms into unsupported details.
- Avoid excessive metaphors and repetitive adjectives.
- Prefer one or two detailed, natural English sentences.

Return only the rewritten caption.
'''

PLAN_PROMPT = '''
You are an expert music analyst.

Extract a structured music plan from the provided
caption. This plan will be used to construct a
text-to-music generation prompt.

Extract only information supported by the caption.

Fields:
- genre: stated musical genre or style
- instruments: instruments explicitly mentioned
- vocals: presence and characteristics of vocals
- rhythm: tempo, beat, groove, rhythmic characteristics
- timbre: sound qualities and textures
- mood: emotional character or atmosphere
- musical_relationships: roles and interactions
  between musical elements
- temporal_events: explicitly stated changes,
  transitions, or progression

Requirements:
- Do not infer unstated instruments or vocals.
- Do not invent BPM, key, time signature, or structure.
- Do not turn metaphors into unsupported sound events.
- Preserve uncertainty and ambiguous descriptions.
- Use null for unspecified scalar fields.
- Use empty lists for unspecified list fields.
- Return only valid JSON.

JSON structure:
{
    "genre": null,
    "instruments": [],
    "vocals": null,
    "rhythm": null,
    "timbre": [],
    "mood": null,
    "musical_relationships": [],
    "temporal_events": []
}
'''

PLANNED_PROMPT = '''
You are an expert text-to-music prompt writer.

Construct a coherent music generation prompt using
ONLY the provided structured music plan.

Requirements:
- Treat the plan as the sole source of information.
- Preserve the genre and explicitly stated instruments.
- Describe the roles and interactions of instruments.
- Integrate rhythm, timbre, and mood when available.
- Preserve explicitly described temporal progression.
- Do not invent musical details to fill missing fields.
- Do not infer vocals, BPM, or additional instruments.
- Do not turn unspecified information into negatives.
- Organize the description logically:
  overall style, main elements, supporting elements,
  rhythmic relationships, and atmosphere.
- Avoid repetitive adjectives and unnecessary metaphors.
- Write one or two natural English sentences.
- Do not mention the plan or its JSON fields.

Return only the final music caption.
'''

def call_llm(client, model, system_prompt, content, json_mode=False):
    kwargs = {
        'model': model,
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': content}
        ]
    }

    if json_mode:
        kwargs['response_format'] = {'type': 'json_object'}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()

def rewrite_caption(client, model, caption):
    #print('\nStarting Concise...', flush=True)
    concise = call_llm(client, model, CONCISE_PROMPT, caption)
    #print('Concise done!', flush=True)

    #print('Starting Detailed...', flush=True)
    detailed = call_llm(client, model, DETAILED_PROMPT, caption)
    #print('Detailed done!', flush=True)

    #print('Starting Planning...', flush=True)
    plan_text = call_llm(
        client, model, PLAN_PROMPT, caption, json_mode=True
    )
    plan = json.loads(plan_text)
    #print('Planning done!', flush=True)

    #print('Starting Planned...', flush=True)
    planned = call_llm(
        client,
        model,
        PLANNED_PROMPT,
        json.dumps(plan, ensure_ascii=False)
    )
    #print('Planned done!', flush=True)

    return {
        'concise': concise,
        'detailed': detailed,
        'planned': planned,
        'plan': plan
    }

def load_results():
    path = OUTPUT_DIR / 'rewrites.json'

    if not path.exists():
        return {}

    with open(path, encoding='utf-8') as f:
        return json.load(f)

def save_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / 'rewrites.json'
    temp_path = OUTPUT_DIR / 'rewrites.tmp'

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    temp_path.replace(path)

def save_captions(rows, results):
    for mode in ['concise', 'detailed', 'planned']:
        path = OUTPUT_DIR / f'{mode}.csv'

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'caption'])

            for sample_id, _ in rows:
                sample_id = str(sample_id)

                if sample_id not in results:
                    continue

                writer.writerow([
                    sample_id,
                    results[sample_id][mode]
                ])

def main():
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        timeout=60.0,
        max_retries=2
    )

    rows = list(read_captions(CAPTIONS_PATH))
    results = load_results()

    for sample_id, caption in tqdm(rows, desc='Rewrite'):
        sample_id = str(sample_id)

        for attempt in range(RETRIES):
            try:
                rewritten = rewrite_caption(client, MODEL, caption)
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    raise
                print(f'Retrying: {e}')
                time.sleep(2 ** attempt)

        results[sample_id] = {
            'original': caption,
            **rewritten
        }

        save_results(results)

    save_captions(
        list(read_captions(CAPTIONS_PATH)),
        results
    )

    print('Output:', OUTPUT_DIR)

if __name__ == '__main__':
    main()
