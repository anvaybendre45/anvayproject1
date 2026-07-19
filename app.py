import os
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Initialize FastAPI App
app = FastAPI(title="DawaDost AI Brain Platform Core")

# Enable CORS (Cross-Origin Resource Sharing) 
# This allows your frontend (e.g., running on localhost:3000) to safely talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Pydantic Schemas for API Validation
# -------------------------------------------------------------------
class InventoryRequest(BaseModel):
    chemist_name: str = "Chemist"
    message: str = Field(..., example="Need 5 boxes of pantocid and 2 calpol")
    pincode: str = Field(..., example="400001")

class OrderItem(BaseModel):
    brand: str
    qty: int

class PharmacyOrderList(BaseModel):
    items: List[OrderItem]

# -------------------------------------------------------------------
# Core Brain Logic Integration
# -------------------------------------------------------------------
BRAND_TO_GENERIC = {
    "pantocid": "Pantoprazole",
    "pan-d": "Pantoprazole + Domperidone",
    "calpol": "Paracetamol",
    "crocin": "Paracetamol",
    "augmentin": "Amoxicillin Clavulanate",
    "avil": "Pheniramine Maleate"
}

class InventoryEngine:
    def __init__(self):
        self.client = genai.Client()
        self.model_id = "gemini-2.5-flash"

    def parse_text(self, text: str) -> Dict[str, int]:
        system_instruction = "Extract medications and quantities. If brand name is used, extract it exactly."
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=f"Extract this order: {text}",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=PharmacyOrderList,
                    temperature=0.0
                ),
            )
            parsed_json = json.loads(response.text)
            mapped_orders = {}
            for item in parsed_json.get("items", []):
                brand_clean = item["brand"].lower().strip()
                generic_name = BRAND_TO_GENERIC.get(brand_clean, item["brand"].title())
                mapped_orders[generic_name] = mapped_orders.get(generic_name, 0) + item["qty"]
            return mapped_orders
        except Exception:
            return {}

    def get_weather(self, pincode: str) -> Dict[str, Any]:
        # Simulating active external environment parameters
        return {"current_season": "Pre-Monsoon", "rain_forecast_next_7_days": True, "avg_temp_celsius": 32}

    def compute_recommendations(self, history: Dict[str, int], environment: Dict[str, Any]) -> List[Dict[str, Any]]:
        recommendations = []
        if environment.get("rain_forecast_next_7_days") and environment.get("current_season") == "Pre-Monsoon":
            if "Pantoprazole" in history or "Pantoprazole + Domperidone" in history:
                recommendations.append({
                    "generic_name": "Ofloxacin + Ornidazole (Generic Enteric Antibiotic)",
                    "reason": "Monsoon onset predicted in your area within 7 days. Waterborne infections rise by 35%.",
                    "suggested_quantity": 10,
                    "price_status": "Standard (Lock rate before surge)"
                })
        return recommendations

# Initialize Engine Instance
engine = InventoryEngine()

# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "engine": "Gemini 2.5 Flash Online"}

@app.post("/api/predict")
def predict_inventory(payload: InventoryRequest):
    # 1. Parse using Gemini
    extracted_stock = engine.parse_text(payload.message)
    
    # 2. Grab hyper-local metrics
    environment = engine.get_weather(payload.pincode)
    
    # 3. Calculate alerts
    alerts = engine.compute_recommendations(extracted_stock, environment)
    
    # Return everything cleanly as structured JSON data for the website
    return {
        "chemist_name": payload.chemist_name,
        "pincode": payload.pincode,
        "extracted_inventory": extracted_stock,
        "upcoming_environmental_risks": environment,
        "smart_recommendations": alerts
    }

if __name__ == "__main__":
    import uvicorn
    # Start server on http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)