import os
import json
from datasets import Dataset

def build_dataset_from_folders(base_path: str = r"C:\github\Ai_Capstone_Project\data") -> Dataset:
    """
    Recursively traverse the directories under base_path and build a HuggingFace Dataset
    in a prompt-response format. Each example consists of a prompt (character description + emotion + personality)
    and a response (the dialogue line).
    """
    examples = []
    char_attrs_by_story = {}
    # Load character definitions to map characters to their attributes
    # The character definitions file is expected at base_path\characters\Characters__definitions.json
    char_defs_path = os.path.join(base_path, "characters", "Characters__definitions.json")
    if os.path.isfile(char_defs_path):
        with open(char_defs_path, "r", encoding="utf-8") as f:
            char_defs_data = json.load(f)
        # char_defs_data is a dict mapping story name to a list of character attribute dicts
        for story_name, char_list in char_defs_data.items():
            story_map = {}
            for char in char_list:
                name = char.get("이름") or char.get("name")
                if not name:
                    continue
                # Build character description from attributes
                occupation = char.get("직업군_분야")        # Occupation field
                roles = char.get("직업군_역할")             # Roles (list)
                social_status = char.get("공통계층")       # Social status
                era = char.get("시대")                    # Era
                age = char.get("나이")                    # Age category
                desc_parts = []
                if era:
                    desc_parts.append(f"{era} 시대")
                if social_status:
                    desc_parts.append(f"{social_status} 신분")
                if occupation:
                    desc_parts.append(occupation)
                if roles:
                    if isinstance(roles, list):
                        # Join multiple roles with '및' (and) for readability
                        role_text = " 및 ".join(str(r) for r in roles) if len(roles) > 1 else str(roles[0])
                    else:
                        role_text = str(roles)
                    if role_text:
                        desc_parts.append(f"{role_text} 역할")
                if age:
                    desc_parts.append(age)
                # Combine parts into a descriptive phrase
                char_description = " ".join(desc_parts)
                if not char_description:
                    char_description = name  # Fallback to name if no attributes
                else:
                    # Include the character's name at the beginning for clarity
                    char_description = f"{name}은(는) " + char_description + "이다."
                story_map[name] = char_description
            char_attrs_by_story[story_name] = story_map

    # Traverse through all files and folders under base_path
    for root, dirs, files in os.walk(base_path):
        # Skip definitions and characters directories (already processed above)
        if os.path.basename(root).lower() in ["definitions", "characters"]:
            continue
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            if "definitions" in fname.lower():
                continue  # skip definition JSON files
            file_path = os.path.join(root, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue  # skip files that cannot be read or parsed
            # Determine story key from filename (without extension) for character lookup
            story_key = os.path.splitext(fname)[0]
            if story_key not in char_attrs_by_story and "_" in story_key:
                alt_key = story_key.replace("_", " ")
                if alt_key in char_attrs_by_story:
                    story_key = alt_key
            # Case 1: JSON data is a dict of character names to a list of utterances
            if isinstance(data, dict):
                for char_name, utterances in data.items():
                    if not isinstance(utterances, list):
                        continue
                    # Find character description if available
                    char_desc = None
                    if story_key in char_attrs_by_story:
                        char_map = char_attrs_by_story[story_key]
                        if char_name in char_map:
                            char_desc = char_map[char_name]
                    if not char_desc:
                        char_desc = char_name  # Fallback to character name
                    # Iterate through each utterance (dialogue line or monologue) for the character
                    for entry in utterances:
                        if not isinstance(entry, dict) or "text" not in entry:
                            continue
                        text = entry["text"]
                        # Gather personality traits and emotions from the entry
                        traits_list = entry.get("traits") or entry.get("성격") or []
                        emotion_list = entry.get("emotion") or entry.get("감정") or []
                        if isinstance(traits_list, str):
                            traits_list = [traits_list]
                        if isinstance(emotion_list, str):
                            emotion_list = [emotion_list]
                        # Join multiple traits or emotions with comma
                        traits_str = ", ".join(traits_list) if traits_list else ""
                        emotion_str = ", ".join(emotion_list) if emotion_list else ""
                        # Build prompt: "캐릭터 설명: {char_desc}\n감정: {emotion_str}\n성격: {traits_str}"
                        prompt_parts = []
                        prompt_parts.append(f"캐릭터 설명: {char_desc}")
                        prompt_parts.append(f"감정: {emotion_str}" if emotion_str else "감정: ")
                        prompt_parts.append(f"성격: {traits_str}" if traits_str else "성격: ")
                        prompt_text = "\n".join(prompt_parts)
                        response_text = text
                        examples.append({"prompt": prompt_text, "response": response_text})
            # Case 2: JSON data is a list of dialogues (each with speaker, text, etc.)
            elif isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    char_name = entry.get("character") or entry.get("speaker") or entry.get("name")
                    text = entry.get("text")
                    if not text or not char_name:
                        continue
                    # Determine character description
                    char_desc = None
                    if story_key in char_attrs_by_story:
                        char_map = char_attrs_by_story[story_key]
                        if char_name in char_map:
                            char_desc = char_map[char_name]
                    if not char_desc:
                        char_desc = char_name
                    traits_list = entry.get("traits") or entry.get("성격") or []
                    emotion_list = entry.get("emotion") or entry.get("감정") or []
                    if isinstance(traits_list, str):
                        traits_list = [traits_list]
                    if isinstance(emotion_list, str):
                        emotion_list = [emotion_list]
                    traits_str = ", ".join(traits_list) if traits_list else ""
                    emotion_str = ", ".join(emotion_list) if emotion_list else ""
                    prompt_parts = [f"캐릭터 설명: {char_desc}"]
                    prompt_parts.append(f"감정: {emotion_str}" if emotion_str else "감정: ")
                    prompt_parts.append(f"성격: {traits_str}" if traits_str else "성격: ")
                    prompt_text = "\n".join(prompt_parts)
                    response_text = text
                    examples.append({"prompt": prompt_text, "response": response_text})
    # Create a HuggingFace Dataset from the collected examples
    dataset = Dataset.from_list(examples)
    return dataset

if __name__ == "__main__":
    # If run as a script, build the dataset and print a summary
    ds = build_dataset_from_folders()
    print(f"Built dataset with {len(ds)} examples.")
    if len(ds) > 0:
        print("Sample prompt-response pair:")
        sample = ds[0]
        print(f"Prompt:\n{sample['prompt']}")
        print(f"Response:\n{sample['response']}")
