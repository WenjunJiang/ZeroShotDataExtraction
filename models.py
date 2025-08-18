"""
models.py - Pydantic v2 models for medical records validation
"""

from typing import Optional, List, Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict, StringConstraints

# Reusable types
Number = Annotated[float, Field(ge=0, description="Non-negative numeric value.")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]

# --- Leaf measurement types (required *within* the object if the object is present) ----

class HeartRate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: Number = Field(..., description="Heart beats per minute.")
    unit: Literal["bpm"] = Field(..., description='Unit; must be exactly "bpm".')

class OxygenSaturation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: Number = Field(..., description="Peripheral capillary oxygen saturation.")
    unit: Literal["%"] = Field(..., description='Unit; must be exactly "%".')

class Cholesterol(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: Number = Field(..., description="Total cholesterol measurement.")
    unit: Literal["mg/dL"] = Field(..., description='Unit; must be exactly "mg/dL".')

class Glucose(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: Number = Field(..., description="Blood glucose measurement.")
    unit: Literal["mg/dL"] = Field(..., description='Unit; must be exactly "mg/dL".')

class Temperature(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: Number = Field(..., description="Body temperature in Celsius.")
    unit: Literal["°C"] = Field(..., description='Unit; must be exactly "°C".')

class RespiratoryRate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: Number = Field(..., description="Breaths per minute.")
    unit: Literal["breaths/min"] = Field(..., description='Unit; must be exactly "breaths/min".')

class Pressure(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: Number = Field(..., description="Blood pressure value in millimeters of mercury.")
    unit: Literal["mmHg"] = Field(..., description='Unit; must be exactly "mmHg".')

class BloodPressure(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    systolic: Pressure = Field(..., description="Systolic arterial pressure.")
    diastolic: Pressure = Field(..., description="Diastolic arterial pressure.")

# --- Higher-level sections (OPTIONAL fields) ------------------------------------------

Gender = Literal["Male", "Female", "Other", "Unknown"]

class PatientInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    age: Optional[Annotated[int, Field(ge=0, le=130, description="Age in years.")]] = None
    gender: Optional[Gender] = Field(None, description="Gender label.")

class VitalSigns(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    heart_rate: Optional[HeartRate] = Field(None, description="Heart rate measurement.")
    oxygen_saturation: Optional[OxygenSaturation] = Field(None, description="Oxygen saturation (SpO₂).")
    cholesterol_level: Optional[Cholesterol] = Field(None, description="Total cholesterol level.")
    glucose_level: Optional[Glucose] = Field(None, description="Blood glucose level.")
    temperature: Optional[Temperature] = Field(None, description="Body temperature.")
    respiratory_rate: Optional[RespiratoryRate] = Field(None, description="Respiratory rate.")
    blood_pressure: Optional[BloodPressure] = Field(None, description="Arterial blood pressure (systolic/diastolic).")

class OutputSchema(BaseModel):
    """Top-level patient summary (all sections optional)."""
    model_config = ConfigDict(extra="forbid", strict=True)
    patient_info: Optional[PatientInfo] = Field(None, description="Basic demographics.")
    visit_motivation: Optional[NonEmptyStr] = Field(None, description="Reason for encounter/visit.")
    symptoms: Optional[List[NonEmptyStr]] = Field(None, description="List of patient-reported symptoms.")
    vital_signs: Optional[VitalSigns] = Field(None, description="Container for vital sign measurements.")

# Alias for backward compatibility with existing code
MedicalRecord = OutputSchema