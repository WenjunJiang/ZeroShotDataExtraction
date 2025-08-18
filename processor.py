"""
processor.py - Response processing and validation utilities
"""

import json
import re
import logging
from typing import Dict, Any
from pydantic import ValidationError

from models import OutputSchema, MedicalRecord

logger = logging.getLogger(__name__)


class ResponseProcessor:
    """Handles cleaning and parsing of LLM responses"""

    @staticmethod
    def extract_json_str(text: str) -> str:
        """
        Return a JSON object string from a model response that may contain prose or code fences.
        Strategy: raw-JSON → ```json ... ``` → first {...} block.
        """
        # 1) Raw JSON
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # 2) Fenced ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if m:
            return m.group(1)

        # 3) Any ``` ... ```
        m = re.search(r"```+\s*(\{.*?\})\s*```+", text, flags=re.S)
        if m:
            return m.group(1)

        # 4) First {...} span
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                json.loads(candidate)  # verify
                return candidate
            except:
                pass

        raise ValueError("No JSON object found in model output.")

    @staticmethod
    def parse_and_validate(response: str) -> OutputSchema:
        """Parse JSON and validate with Pydantic"""
        try:
            # Extract JSON from response
            json_str = ResponseProcessor.extract_json_str(response)

            # Parse JSON
            data = json.loads(json_str)

            # Remove None values and empty dictionaries
            data = ResponseProcessor._remove_empty_values(data)

            # Validate with Pydantic
            record = OutputSchema(**data)
            return record

        except ValueError as e:
            # From extract_json_str
            logger.error(f"Failed to extract JSON: {e}")
            logger.debug(f"Response text: {response[:500]}...")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.debug(f"Attempted to parse: {json_str[:500] if 'json_str' in locals() else 'N/A'}...")
            raise
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            logger.debug(f"Data that failed validation: {data if 'data' in locals() else 'N/A'}")
            raise

    @staticmethod
    def _remove_empty_values(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove None values and empty dictionaries"""
        if not isinstance(data, dict):
            return data

        cleaned = {}
        for key, value in data.items():
            if value is None:
                continue
            elif isinstance(value, dict):
                cleaned_dict = ResponseProcessor._remove_empty_values(value)
                if cleaned_dict:  # Only add non-empty dictionaries
                    cleaned[key] = cleaned_dict
            elif isinstance(value, list) and len(value) > 0:
                # Clean list items if they are strings
                if all(isinstance(item, str) for item in value):
                    # Remove empty strings and strip whitespace
                    cleaned_list = [item.strip() for item in value if item and item.strip()]
                    if cleaned_list:
                        cleaned[key] = cleaned_list
                else:
                    cleaned[key] = value
            elif not isinstance(value, (dict, list)):
                # For strings, strip whitespace
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        cleaned[key] = stripped
                else:
                    cleaned[key] = value

        return cleaned

    @staticmethod
    def extract_error_info(error: Exception) -> str:
        """Extract readable error information for retry context"""
        if isinstance(error, ValueError):
            return f"JSON extraction failed: {str(error)}"
        elif isinstance(error, json.JSONDecodeError):
            return f"Invalid JSON format at position {error.pos}: {error.msg}"
        elif isinstance(error, ValidationError):
            errors = []
            for err in error.errors():
                field = ' -> '.join(str(x) for x in err['loc'])
                errors.append(f"{field}: {err['msg']}")
            return f"Validation errors: {'; '.join(errors[:3])}"  # Limit to first 3 errors
        else:
            return str(error)