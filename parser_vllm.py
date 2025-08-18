"""
parser.py - Main medical notes parser with retry logic
"""

import json
import logging
from typing import Dict, Any, List, Optional
from time import time, sleep

import pandas as pd
import vllm
from vllm import SamplingParams
from pydantic import ValidationError

from config import LLMConfig, RetryConfig
from prompts import PromptManager
from processor import ResponseProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalNotesParser:
    """Main class for parsing medical notes with retry logic"""

    def __init__(self, llm_config: LLMConfig = None, retry_config: RetryConfig = None):
        """
        Initialize the parser with configuration

        Args:
            llm_config: Configuration for LLM
            retry_config: Configuration for retry logic
        """
        self.llm_config = llm_config or LLMConfig()
        self.retry_config = retry_config or RetryConfig()
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the VLLM model"""
        logger.info("Initializing LLM...")
        try:
            # self.llm = vllm.LLM(
            #     model=self.llm_config.model_dir,
            #     tensor_parallel_size=self.llm_config.tensor_parallel_size,
            #     gpu_memory_utilization=self.llm_config.gpu_memory_utilization,
            #     max_model_len=self.llm_config.max_model_len,
            #     trust_remote_code=self.llm_config.trust_remote_code,
            #     dtype=self.llm_config.dtype,
            #     enforce_eager=self.llm_config.enforce_eager,
            # )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _get_sampling_params(self, temperature_offset: float = 0.0) -> SamplingParams:
        """Get sampling parameters with optional temperature adjustment"""
        return SamplingParams(
            n=self.llm_config.n_samples,
            temperature=min(self.llm_config.temperature + temperature_offset, 1.0),
            top_p=self.llm_config.top_p,
            seed=self.llm_config.seed,
            max_tokens=self.llm_config.max_tokens
        )

    def parse_single_note_with_retry(
            self,
            medical_note: str,
            note_id: str = None
    ) -> Dict[str, Any]:
        """
        Parse a single medical note with retry logic

        Args:
            medical_note: The medical note text to parse
            note_id: Optional identifier for the note

        Returns:
            Dictionary containing parsing results
        """
        last_error = None

        for attempt in range(self.retry_config.max_retries):
            try:
                # Create message (with error context on retry)
                if attempt > 0 and last_error:
                    message = PromptManager.create_retry_message(
                        medical_note,
                        ResponseProcessor.extract_error_info(last_error)
                    )
                else:
                    message = PromptManager.create_message(medical_note)

                # Adjust temperature for retries
                temp_offset = attempt * self.retry_config.temperature_increment
                sampling_params = self._get_sampling_params(temp_offset)

                # Get LLM response
                logger.debug(f"Attempt {attempt + 1} for note {note_id}")
                output = self.llm.chat(
                    messages=[message],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )

                response_text = output[0].outputs[0].text
                # response_text = 'Here is the extracted JSON:\n\n```\n{\n  "patient_info": {\n  "age": 29, \n  "gender": "male",\n  "visit_motivation": "Influenza (Flu)" : "Influenza (Flu)"\n}\n```\n\n\nNote that I converted the blood pressure value from the original note to 1 decimal. \n\n````\n{\n  "patient_info": {\n  "age": 29, \n  "gender": "male" | "female" | "other" | "unknown"\n},\n  "visit_motivation": "Allergies",\n  "symptoms": [\n    "runny_nose", \n    "itchy_eyes", \n    "blurred_vision", \n    "wheezing"\n  ],\n  "vital_signs": {\n    "heart_rate": {"value": 13, "unit": "breaths/min"},\n    "oxygen_saturation": {"value": 100, "unit": "%"},\n    "cholesterol_level": {"value": 99, "unit": "mg/dL"},\n    "glucose_level": {"value": 99, "unit": "mg/dL"},\n    "temperature": {"value": 36.6, "unit": "°C"},\n    "respiratory_rate": {"value": 13, "unit": "breaths/min"},\n    "blood_pressure": {\n      "systolic": {"value": 120, "unit": "mmHg"},\n      "diastolic": {"value": 80, "unit": "mmHg"}\n    }\n  }\n}```'

                # Parse and validate
                medical_record = ResponseProcessor.parse_and_validate(response_text)

                # Convert to dict for storage
                return {
                    'id': note_id,
                    'success': True,
                    'data': medical_record.dict(exclude_none=True),
                    'attempts': attempt + 1,
                    'raw_response': response_text
                }

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for note {note_id}: {e}")

                if attempt < self.retry_config.max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = self.retry_config.retry_delay
                    if self.retry_config.exponential_backoff:
                        delay *= (2 ** attempt)
                    sleep(delay)
                else:
                    # Final attempt failed
                    logger.error(f"All attempts failed for note {note_id}")
                    return {
                        'id': note_id,
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'attempts': attempt + 1,
                        'raw_response': response_text if 'response_text' in locals() else None
                    }

            except Exception as e:
                logger.error(f"Unexpected error for note {note_id}: {e}")
                return {
                    'id': note_id,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'attempts': attempt + 1,
                    'raw_response': None
                }

    def parse_batch(
            self,
            df: pd.DataFrame,
            note_column: str = 'Note',
            id_column: str = 'ID',
            batch_size: int = None
    ) -> pd.DataFrame:
        """
        Parse a batch of medical notes from a DataFrame

        Args:
            df: DataFrame containing medical notes
            note_column: Name of the column containing notes
            id_column: Name of the ID column
            batch_size: Optional batch size for processing

        Returns:
            DataFrame with parsed results
        """
        logger.info(f"Processing {len(df)} medical notes...")

        results = []
        start_time = time()

        # Process in batches if specified
        if batch_size:
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                batch_results = self._process_batch_chunk(batch_df, note_column, id_column)
                results.extend(batch_results)
                logger.info(f"Processed {min(i + batch_size, len(df))}/{len(df)} notes")
        else:
            results = self._process_batch_chunk(df, note_column, id_column)

        elapsed = (time() - start_time) / 60
        logger.info(f"Processing completed in {elapsed:.2f} minutes")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Merge with original DataFrame
        output_df = df[[id_column]].copy()
        output_df = output_df.merge(
            results_df[['id', 'success', 'data', 'attempts']],
            left_on=id_column,
            right_on='id',
            how='left'
        )
        output_df = output_df.drop('id', axis=1)  # Remove duplicate id column

        # Add error information for failed parses
        if 'error' in results_df.columns:
            error_df = results_df[results_df['success'] == False][['id', 'error', 'error_type']]
            output_df = output_df.merge(
                error_df,
                left_on=id_column,
                right_on='id',
                how='left'
            )
            if 'id' in output_df.columns:
                output_df = output_df.drop('id', axis=1)

        # Log statistics
        success_rate = results_df['success'].mean() * 100
        avg_attempts = results_df['attempts'].mean()
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Average attempts: {avg_attempts:.2f}")

        return output_df

    def _process_batch_chunk(
            self,
            df: pd.DataFrame,
            note_column: str,
            id_column: str
    ) -> List[Dict[str, Any]]:
        """Process a chunk of the DataFrame"""
        # Prepare all messages
        messages = []
        ids = []

        for _, row in df.iterrows():
            messages.append(PromptManager.create_message(row[note_column]))
            ids.append(row[id_column])

        # First attempt: batch processing
        sampling_params = self._get_sampling_params()
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=True
        )

        results = []
        retry_queue = []

        # Process batch results
        for i, output in enumerate(outputs):
            try:
                response_text = output.outputs[0].text
                # response_text = 'Here is the extracted information in the specified JSON format:\n\n```json\n{\n  "patient_info": {\n    "age": 51,\n    "gender": "Male"\n  },\n  "visit_motivation": "Tuberculosis (TB)",\n  "symptoms": [\n    "Fever",\n    "Cough",\n    "Nausea",\n    "Joint pain",\n    "Sneezing",\n    "Frequent urination",\n    "Blurred vision",\n    "Weight loss",\n    "Painful urination",\n    "Facial pain",\n    "Difficulty concentrating"\n  ],\n  "vital_signs": {\n    "heart_rate": {\n      "value": 19,\n      "unit": "breaths/min"\n    },\n    "oxygen_saturation": {\n      "value": null,\n      "unit": null\n    },\n    "cholesterol_level": {\n      "value": 125.7,\n      "unit": "mg/dL"\n    },\n    "glucose_level": {\n      "value": 70.7,\n      "unit": "mg/dL"\n    },\n    "temperature": {\n      "value": null,\n      "unit": null\n    },\n    "respiratory_rate": {\n      "value": 19,\n      "unit": "breaths/min"\n    },\n    "blood_pressure": {\n      "systolic": {\n        "value": 92,\n        "unit": "mmHg"\n      },\n      "diastolic": {\n        "value": 74,\n        "unit": "mmHg"\n      }\n    }\n  }\n}\n```\n\nNote that I\'ve omitted the `oxygen_saturation` and `temperature` vital signs as they were not mentioned in the notes.'
                medical_record = ResponseProcessor.parse_and_validate(response_text)

                results.append({
                    'id': ids[i],
                    'success': True,
                    'data': medical_record.dict(exclude_none=True),
                    'attempts': 1,
                    'raw_response': response_text
                })
            except Exception as e:
                logger.warning(f"Initial parsing failed for {ids[i]}: {e}")
                retry_queue.append((ids[i], df.iloc[i][note_column]))

        # Process retries individually
        for note_id, medical_note in retry_queue:
            result = self.parse_single_note_with_retry(medical_note, note_id)
            results.append(result)

        return results