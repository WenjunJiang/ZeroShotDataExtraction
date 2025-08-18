"""
utils.py - Utility functions for medical notes processing
"""

import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from config import LLMConfig, RetryConfig
from parser import MedicalNotesParser

logger = logging.getLogger(__name__)


def load_and_process_medical_notes(
    df: pd.DataFrame,
    note_column: str = 'Note',
    id_column: str = 'ID',
    llm_config: LLMConfig = None,
    retry_config: RetryConfig = None,
    batch_size: int = None
) -> pd.DataFrame:
    """
    High-level function to process medical notes from a DataFrame

    Args:
        df: DataFrame containing medical notes
        note_column: Name of the column containing medical notes
        id_column: Name of the ID column
        llm_config: LLM configuration
        retry_config: Retry configuration
        batch_size: Optional batch size for processing

    Returns:
        DataFrame with parsed medical records
    """
    parser = MedicalNotesParser(llm_config, retry_config)
    return parser.parse_batch(df, note_column, id_column, batch_size)


def save_results(
    results_df: pd.DataFrame,
    output_path: str,
    include_raw: bool = False
) -> None:
    """
    Save parsing results to file

    Args:
        results_df: DataFrame with parsing results
        output_path: Path to save the results
        include_raw: Whether to include raw responses
    """
    if not include_raw and 'raw_response' in results_df.columns:
        results_df = results_df.drop('raw_response', axis=1)

    if output_path.endswith('.json'):
        results_df.to_json(output_path, orient='records', indent=2)
    else:
        results_df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")


def extract_successful_records(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only successfully parsed records

    Args:
        results_df: DataFrame with parsing results

    Returns:
        DataFrame with only successful records
    """
    return results_df[results_df['success'] == True].copy()


def extract_failed_records(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only failed parsing records for debugging

    Args:
        results_df: DataFrame with parsing results

    Returns:
        DataFrame with only failed records
    """
    return results_df[results_df['success'] == False].copy()


def get_parsing_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about parsing results

    Args:
        results_df: DataFrame with parsing results

    Returns:
        Dictionary with statistics
    """
    total = len(results_df)
    successful = results_df['success'].sum()
    failed = total - successful

    stats = {
        'total_records': total,
        'successful': successful,
        'failed': failed,
        'success_rate': (successful / total * 100) if total > 0 else 0,
        'average_attempts': results_df['attempts'].mean() if 'attempts' in results_df else 0
    }

    # Add error breakdown if available
    if 'error_type' in results_df.columns:
        error_counts = results_df[results_df['success'] == False]['error_type'].value_counts().to_dict()
        stats['error_breakdown'] = error_counts

    return stats


def flatten_medical_record(record_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested medical record dictionary for easier analysis

    Args:
        record_dict: Nested medical record dictionary

    Returns:
        Flattened dictionary
    """
    flattened = {}

    # Patient info
    if 'patient_info' in record_dict:
        flattened['patient_age'] = record_dict['patient_info'].get('age')
        flattened['patient_gender'] = record_dict['patient_info'].get('gender')

    # Visit motivation and symptoms
    flattened['visit_motivation'] = record_dict.get('visit_motivation')
    flattened['symptoms'] = ', '.join(record_dict.get('symptoms', []))

    # Vital signs
    if 'vital_signs' in record_dict:
        vitals = record_dict['vital_signs']

        # Simple vital signs
        for vital in ['heart_rate', 'oxygen_saturation', 'cholesterol_level',
                     'glucose_level', 'temperature', 'respiratory_rate']:
            if vital in vitals:
                flattened[f'{vital}_value'] = vitals[vital].get('value')
                flattened[f'{vital}_unit'] = vitals[vital].get('unit')

        # Blood pressure
        if 'blood_pressure' in vitals:
            bp = vitals['blood_pressure']
            if 'systolic' in bp:
                flattened['bp_systolic_value'] = bp['systolic'].get('value')
                flattened['bp_systolic_unit'] = bp['systolic'].get('unit')
            if 'diastolic' in bp:
                flattened['bp_diastolic_value'] = bp['diastolic'].get('value')
                flattened['bp_diastolic_unit'] = bp['diastolic'].get('unit')

    return flattened


def create_analysis_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame suitable for analysis from parsing results

    Args:
        results_df: DataFrame with parsing results

    Returns:
        DataFrame with flattened medical records
    """
    successful = extract_successful_records(results_df)

    if len(successful) == 0:
        logger.warning("No successful records to analyze")
        return pd.DataFrame()

    # Flatten all successful records
    flattened_records = []
    for _, row in successful.iterrows():
        if isinstance(row['data'], dict):
            flattened = flatten_medical_record(row['data'])
            flattened['ID'] = row.get('ID', row.name)
            flattened_records.append(flattened)

    return pd.DataFrame(flattened_records)


def validate_batch_results(results_df: pd.DataFrame) -> List[str]:
    """
    Validate batch processing results and return warnings

    Args:
        results_df: DataFrame with parsing results

    Returns:
        List of warning messages
    """
    warnings = []

    # Check success rate
    stats = get_parsing_statistics(results_df)
    if stats['success_rate'] < 80:
        warnings.append(f"Low success rate: {stats['success_rate']:.1f}%")

    # Check for high retry rates
    if stats['average_attempts'] > 2:
        warnings.append(f"High average attempts: {stats['average_attempts']:.1f}")

    # Check for specific error patterns
    if 'error_breakdown' in stats:
        for error_type, count in stats['error_breakdown'].items():
            if count > len(results_df) * 0.1:  # More than 10% of records
                warnings.append(f"Frequent error type: {error_type} ({count} occurrences)")

    return warnings