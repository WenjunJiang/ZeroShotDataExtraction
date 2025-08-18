"""
parser.py - Main medical notes parser with retry logic (Ollama version)
"""

import json
import logging
from typing import Dict, Any, List, Optional
from time import time, sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pydantic import ValidationError

from ollama import Client as OllamaClient
from config import LLMConfig, RetryConfig
from prompts import PromptManager
from processor import ResponseProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalNotesParser:
    """Main class for parsing medical notes with retry logic (Ollama backend)."""

    def __init__(self, llm_config: LLMConfig = None, retry_config: RetryConfig = None):
        """
        Initialize the parser with configuration

        Args:
            llm_config: Configuration for LLM
            retry_config: Configuration for retry logic
        """
        self.llm_config = llm_config or LLMConfig()
        self.retry_config = retry_config or RetryConfig()

        # Resolve model + host from your existing config (keeps backward compat with model_dir)
        self.model_name = self.llm_config.model_name
        self.host = self.llm_config.host
        # Concurrency for "batch" (Ollama has no multi-chat batch API)
        self.max_concurrency = self.llm_config.max_concurrency

        self.client: Optional[OllamaClient] = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the Ollama client and perform a lightweight check."""
        logger.info("Initializing Ollama client...")
        try:
            self.client = OllamaClient(host=self.host)

            # Optional: verify model is available (non-fatal if not)
            try:
                _ = self.client.show(self.model_name)
                logger.info(f"Model '{self.model_name}' is available.")
            except Exception:
                logger.warning(
                    f"Model '{self.model_name}' not found locally yet. "
                    "Make sure to `ollama pull <model>` before running."
                )

            logger.info("Ollama client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise

    # ---- Sampling/options mapping ----
    def _get_sampling_options(self, temperature_offset: float = 0.0) -> Dict[str, Any]:
        """
        Map your LLMConfig to Ollama 'options' dict.
        vLLM -> Ollama:
            temperature -> temperature
            top_p       -> top_p
            seed        -> seed
            max_tokens  -> num_predict
        """
        max_temp = 1.0 # 1.0 for ollama, 2.0 for vllm
        temperature = max(0.0, min(self.llm_config.temperature + temperature_offset, max_temp))
        options: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": self.llm_config.top_p,
            "seed": self.llm_config.seed,
            "num_predict": self.llm_config.max_tokens,
        }
        # You can add more knobs here if you use them in LLMConfig:
        # options.update({"top_k": ..., "repeat_penalty": ..., "stop": [...]})
        return options

    # ---- Single chat call ----
    def _chat_once(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """
        Call Ollama chat for a single conversation (messages = [{'role', 'content'}, ...]).
        Returns the assistant text.
        """
        if not self.client:
            raise RuntimeError("Ollama client not initialized")
        resp = self.client.chat(model=self.model_name, messages=messages, options=options)
        # Response shape: {'model': ..., 'message': {'role': 'assistant', 'content': '...'}, ...}
        return resp["message"]["content"]

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
                options = self._get_sampling_options(temp_offset)

                logger.debug(f"Attempt {attempt + 1} for note {note_id}")
                # NOTE: message is already a list of role/content dicts
                # Ollama does not support n>1 samples in a single call; we take one best sample.
                response_text = self._chat_once(messages=message, options=options)

                # Parse and validate
                medical_record = ResponseProcessor.parse_and_validate(response_text)

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

        results: List[Dict[str, Any]] = []
        start_time = time()

        # Process in chunks if specified
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
        success_rate = results_df['success'].mean() * 100 if len(results_df) else 0.0
        avg_attempts = results_df['attempts'].mean() if len(results_df) else 0.0
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Average attempts: {avg_attempts:.2f}")

        return output_df

    def _process_batch_chunk(
        self,
        df: pd.DataFrame,
        note_column: str,
        id_column: str
    ) -> List[Dict[str, Any]]:
        """
        Process a chunk of the DataFrame.

        Ollama does not support multi-conversation batching in a single call,
        so we execute chats concurrently via a small thread pool.
        """
        # Prepare all messages
        messages_list: List[List[Dict[str, str]]] = []
        ids: List[Any] = []

        for _, row in df.iterrows():
            messages_list.append(PromptManager.create_message(row[note_column]))
            ids.append(row[id_column])

        options = self._get_sampling_options()

        results: List[Dict[str, Any]] = []
        retry_queue: List[tuple] = []

        # Run a concurrent "first attempt" pass
        def _worker(msgs: List[Dict[str, str]]) -> str:
            return self._chat_once(messages=msgs, options=options)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {pool.submit(_worker, m): i for i, m in enumerate(messages_list)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    response_text = fut.result()
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

        # Process retries individually (with the single-note retry logic)
        for note_id, medical_note in retry_queue:
            result = self.parse_single_note_with_retry(medical_note, note_id)
            results.append(result)

        return results
