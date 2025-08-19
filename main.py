"""
Main Jupyter Notebook for Medical Notes Processing
"""

# %% [markdown]
# # Medical Notes Parser with Pydantic Validation
# 
# This notebook demonstrates how to use the modular medical notes parser with retry logic and Pydantic validation.

# %% [markdown]
# ## 1. Setup and Imports

# %%
# Import standard libraries
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from config import LLMConfig, RetryConfig
from parser import MedicalNotesParser
from utils import (
    load_and_process_medical_notes,
    save_results,
    extract_successful_records,
    extract_failed_records,
    get_parsing_statistics,
    create_analysis_dataframe,
    validate_batch_results
)


# %% [markdown]
# ## 2. Load Your Data

# %%
# Load your medical notes data
# Replace with your actual data file
train_df = pd.read_csv('medical-note-extraction-h-2-o-gen-ai-world-ny/train.csv')

# Display basic information about the dataset
print(f"Dataset shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"\nFirst few rows:{train_df.head(2)}" )
train_sample_df = train_df.head(1000)

# %% [markdown]
# ## 3. Configure Settings

# %%
# Configure LLM settings
llm_config = LLMConfig()

# Configure retry settings
retry_config = RetryConfig(
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True,
    temperature_increment=0.1  # Increase temperature on each retry
)

print("Configuration loaded successfully")

# %% [markdown]
# ## 4. Process Medical Notes

# %%
# Option 1: Use the high-level function
results_df = load_and_process_medical_notes(
    train_sample_df,
    note_column='Note',
    id_column='ID',
    llm_config=llm_config,
    retry_config=retry_config,
    batch_size=10  # Process in batches of 10
)

# %% [markdown]
# ### Alternative: Use the parser directly for more control

# %%
# Option 2: Use the parser directly
# parser = MedicalNotesParser(llm_config, retry_config)

# # Process all notes
# results_df = parser.parse_batch(
#     train_sample_df,
#     note_column='Note',
#     id_column='ID',
#     batch_size=10
# )
#
# # Or process a single note
# single_note = train_sample_df.iloc[0]['Note']
# single_result = parser.parse_single_note_with_retry(
#     single_note,
#     note_id='TEST_001'
# )
# print("Single note result:", single_result)

# %% [markdown]
# ## 5. Analyze Results

# %%
# Get parsing statistics
stats = get_parsing_statistics(results_df)
print("Parsing Statistics:")
print(f"  Total records: {stats['total_records']}")
print(f"  Successful: {stats['successful']}")
print(f"  Failed: {stats['failed']}")
print(f"  Success rate: {stats['success_rate']:.1f}%")
print(f"  Average attempts: {stats['average_attempts']:.2f}")

if 'error_breakdown' in stats:
    print("\nError breakdown:")
    for error_type, count in stats['error_breakdown'].items():
        print(f"  {error_type}: {count}")

# %%
# Validate results and check for warnings
warnings = validate_batch_results(results_df)
if warnings:
    print("⚠️ Warnings detected:")
    for warning in warnings:
        print(f"  - {warning}")
else:
    print("✅ No warnings detected")

# %% [markdown]
# ## 6. Extract and Explore Successful Records

# %%
# Extract successful records
successful_df = extract_successful_records(results_df)
print(f"Successfully parsed {len(successful_df)} out of {len(results_df)} records")

# Display a sample parsed record
if len(successful_df) > 0:
    sample_record = successful_df.iloc[0]['data']
    print("\nSample parsed record:")
    print(json.dumps(sample_record, indent=2))

# %%
# Create analysis-ready dataframe with flattened records
analysis_df = create_analysis_dataframe(results_df)
print(f"Analysis dataframe shape: {analysis_df.shape}")
print("\nColumns in analysis dataframe:")
print(analysis_df.columns.tolist())

# Display summary statistics for vital signs
if 'heart_rate_value' in analysis_df.columns:
    print("\nVital Signs Summary:")
    vital_columns = [col for col in analysis_df.columns if '_value' in col]
    for col in vital_columns:
        if analysis_df[col].notna().any():
            print(f"  {col}:")
            print(f"    Mean: {analysis_df[col].mean():.2f}")
            print(f"    Std: {analysis_df[col].std():.2f}")
            print(f"    Min: {analysis_df[col].min():.2f}")
            print(f"    Max: {analysis_df[col].max():.2f}")

# %% [markdown]
# ## 7. Handle Failed Records

# %%
# Extract and analyze failed records
failed_df = extract_failed_records(results_df)

if len(failed_df) > 0:
    print(f"Failed to parse {len(failed_df)} records")
    print("\nSample of failed records:")
    for idx, row in failed_df.head(3).iterrows():
        print(f"\nID: {row.get('ID', idx)}")
        print(f"Error Type: {row.get('error_type', 'Unknown')}")
        print(f"Error: {row.get('error', 'No error message')[:200]}...")
        print(f"Attempts: {row.get('attempts', 0)}")
else:
    print("All records parsed successfully!")

# %% [markdown]
# ## 8. Save Results

# %%
# Save all results
save_results(results_df, 'parsed_medical_notes.csv', include_raw=False)

# Save only successful records
save_results(successful_df, 'successful_parses.csv', include_raw=False)

# Save failed records for debugging
if len(failed_df) > 0:
    save_results(failed_df, 'failed_parses.csv', include_raw=True)

# Save analysis-ready dataframe
if len(analysis_df) > 0:
    analysis_df.to_csv('medical_records_analysis.csv', index=False)
    print("Analysis dataframe saved to medical_records_analysis.csv")

# %% [markdown]
# ## 9. Export to JSON Format (Optional)

# %%
# Export successful records to JSON for use in other applications
if len(successful_df) > 0:
    json_records = []
    for _, row in successful_df.iterrows():
        record = {
            'id': row.get('ID', row.name),
            'medical_record': row['data']
        }
        json_records.append(record)

    with open('medical_records.json', 'w') as f:
        json.dump(json_records, f, indent=2)

    print(f"Exported {len(json_records)} records to medical_records.json")






# %% [markdown]
# ## 10. Advanced: Custom Validation Example

# %%
# Example of using the models directly for validation
# from models import OutputSchema, PatientInfo, VitalSigns, HeartRate, Temperature
#
# # Create a medical record manually
# try:
#     test_record = MedicalRecord(
#         patient_info=PatientInfo(
#             age=45,
#             gender="male"
#         ),
#         visit_motivation="Routine checkup",
#         symptoms=["headache", "fatigue"],
#         vital_signs=VitalSigns(
#             heart_rate=VitalValue(value=72, unit="bpm"),
#             temperature=VitalValue(value=37.2, unit="°C")
#         )
#     )
#     print("✅ Valid medical record created:")
#     print(test_record.dict(exclude_none=True))
# except Exception as e:
#     print(f"❌ Validation error: {e}")

# %% [markdown]
# ## 11. Batch Processing with Progress Monitoring

# %%
# For large datasets, process with progress monitoring
# def process_with_progress(df, chunk_size=5):
#     """Process dataframe in chunks with progress updates"""
#     from tqdm.notebook import tqdm
#
#     parser = MedicalNotesParser(llm_config, retry_config)
#     all_results = []
#
#     for i in tqdm(range(0, len(df), chunk_size), desc="Processing batches"):
#         chunk = df.iloc[i:i+chunk_size]
#         chunk_results = parser.parse_batch(
#             chunk,
#             note_column='Note',
#             id_column='ID'
#         )
#         all_results.append(chunk_results)
#
#     return pd.concat(all_results, ignore_index=True)

# Uncomment to use with progress bar
# results_with_progress = process_with_progress(train_sample_df, chunk_size=5)

# %% [markdown]
# ## Summary
# 
# This notebook demonstrated:
# 1. Loading and configuring the medical notes parser
# 2. Processing medical notes with retry logic
# 3. Analyzing parsing results and statistics
# 4. Handling successful and failed records
# 5. Exporting results in various formats
# 6. Advanced usage with custom validation
# 
# The modular design allows for easy customization and extension of the parsing pipeline.