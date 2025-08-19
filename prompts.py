"""
prompts.py - Prompt templates and message formatting
"""

from typing import List, Dict


class PromptManager:
    """Manages prompt templates and formatting"""

    SYSTEM_PROMPT = """As a medical professional, please analyze the following medical notes carefully and extract relevant information in the specified JSON format.
Do not include any values that are not mentioned in the notes.
Ensure all numeric values are properly formatted as numbers, not strings.
Only include fields that have actual data from the notes.
For gender field, use only: "Male", "Female", "Other", or "Unknown".
For symptoms field, every word is written using lowercase letters, and each symptom name which consisted of multiple words are connected by underscores. For example: "symptoms": ["fever", "cough", "sore_throat", "chest_pain", "diarrhea", "runny_nose", "rash", "sneezing", "loss_of_taste_smell", "frequent_urination", "night_sweats", "weight_loss", "facial_pain", "pale_skin"]
For visit_motivation field, only capture the disease name and capitalize every word. For example: "Chief Complaint: Diabetes" is extracted as "Diabetes".  And "Evaluation of Influenza (Flu)" is extracted as "Influenza (Flu)".
visit_motivation is not equal to symptom unless explicitly listed as a symptom.
"""

    PROMPT_TEMPLATE = """
For each entry:
- Accurately identify and extract patient data, visit motivation, symptoms, and vital signs.
- Only include keys that have corresponding information in the notes, omitting any keys that are not mentioned.
- Ensure all numeric values are numbers, not strings.
- Follow this exact JSON structure:
```json
{
  "patient_info": {
    "age": <int>,
    "gender": <string: "Male", "Female", "Other", or "Unknown">
  },
  "visit_motivation": "<string>",
  "symptoms": ["<string>"],
  "vital_signs": {
    "heart_rate": {
      "value": <int>,
      "unit": "bpm"
    },
    "oxygen_saturation": {
      "value": <float>,
      "unit": "%"
    },
    "cholesterol_level": {
      "value": <float>,
      "unit": "mg/dL"
    },
    "glucose_level": {
      "value": <float>,
      "unit": "mg/dL"
    },
    "temperature": {
      "value": <float>,
      "unit": "°C"
    },
    "respiratory_rate": {
      "value": <int>,
      "unit": "breaths/min"
    },
    "blood_pressure": {
      "systolic": {
        "value": <int>,
        "unit": "mmHg"
      },
      "diastolic": {
        "value": <int>,
        "unit": "mmHg"
      }
    }
  }
}
```

Remember:
- Only include fields that are explicitly mentioned in the notes
- Do not invent or assume any values
- Ensure proper JSON formatting

Here are the clinical notes to review:
{medical_notes}
"""

    @classmethod
    def create_message(cls, medical_notes: str) -> List[Dict[str, str]]:
        """Create a formatted message for the LLM"""
        filled_prompt = cls.PROMPT_TEMPLATE.replace("{medical_notes}", medical_notes)
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": filled_prompt}
        ]

    @classmethod
    def create_retry_message(cls, medical_notes: str, previous_error: str = None) -> List[Dict[str, str]]:
        """Create a message for retry attempts with error context"""
        base_prompt = cls.PROMPT_TEMPLATE.replace("{medical_notes}", medical_notes)

        if previous_error:
            retry_prompt = f"""Previous attempt failed with error: {previous_error}
            
Please ensure the response is valid JSON and follows the exact structure specified.
{base_prompt}"""
        else:
            retry_prompt = base_prompt

        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": retry_prompt}
        ]