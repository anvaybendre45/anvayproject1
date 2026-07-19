import os
import json
import streamlit as st
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# -------------------------------------------------------------------
# 1. STREAMLIT APP CONFIG & STYLING
# -------------------------------------------------------------------
st.set_page_config(
    page_title="DawaDost Smart Predictor Dashboard",
    page_icon="💊",
    layout="wide"
)

st.title("💊 DawaDost AI Inventory Engine")
st.markdown("### Small-Scale Fullstack Prototype for Mom-and-Pop Pharmacies")
st.write("---")

# -------------------------------------------------------------------
# 2. DATA SCHEMAS & DICTIONARIES
# -------------------------------------------------------------------
class OrderItem(BaseModel):
    brand: str = Field(description="The brand name or generic drug molecule name mentioned.")
    qty: int = Field(description="The numeric quantity ordered. Default to 5 if not explicitly mentioned.")

class PharmacyOrderList(BaseModel):
    items: List[OrderItem]

BRAND_TO_GENERIC = {
    "pantocid": "Pantoprazole",
    "pan-d": "Pantoprazole + Domperidone",
    "calpol": "Paracetamol",
    "crocin": "Paracetamol",
    "augmentin": "Amoxicillin Clavulanate",
    "avil": "Pheniramine Maleate"
}

# -------------------------------------------------------------------
# 3. CORE AI ENGINE CLASS
# -------------------------------------------------------------------
class InventoryEngine:
    def __init__(self):
        # Initializing client natively via environment variable
        self.client = genai.Client()
        self.model_id = "self.model_id = "gemini-3.5-flash"

    def parse_text(self, text: str) -> Dict[str, int]:
        system_instruction = "Extract medications and quantities. If a brand name is used, extract it exactly."
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
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
            return {}

    def fetch_weather(self, pincode: str) -> Dict[str, Any]:
        # Simulating external public health & climate parameters
        return {
            "pincode": pincode,
            "current_season": "Pre-Monsoon",
            "rain_forecast_next_7_days": True,
            "avg_temp_celsius": 32
        }

    def compute_recommendations(self, history: Dict[str, int], environment: Dict[str, Any]) -> List[Dict[str, Any]]:
        recommendations = []
        # Rule Validation: Impending Monsoon Vector Checks
        if environment.get("rain_forecast_next_7_days") and environment.get("current_season") == "Pre-Monsoon":
            if "Pantoprazole" in history or "Pantoprazole + Domperidone" in history:
                recommendations.append({
                    "generic_name": "Ofloxacin + Ornidazole",
                    "category": "Generic Enteric Antibiotic",
                    "reason": "Monsoon onset predicted in this pincode within 7 days. Waterborne gastrointestinal issues typically surge by 35%.",
                    "suggested_quantity": 10,
                    "price_status": "Standard Rate (Lock in before regional distributor price surge)"
                })
        return recommendations

# Initialize Engine (Cached so it stays fast across interactions)
@st.cache_resource
def get_engine():
    return InventoryEngine()

# Verify API key is present
if "GEMINI_API_KEY" not in os.environ:
    st.warning("⚠️ `GEMINI_API_KEY` environment variable missing. Please export it in your terminal before running.")

engine = get_engine()

# -------------------------------------------------------------------
# 4. STREAMLIT FRONTEND USER INTERFACE
# -------------------------------------------------------------------
# Create side-by-side columns for input configuration
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 Input Data Feed")
    
    chemist_name = st.text_input("Chemist Name", value="Ramesh Kumar")
    pincode = st.text_input("Store Pincode", value="400001", max_chars=6)
    
    message_input = st.text_area(
        "Messy WhatsApp Log / Inbound Request",
        value="Hey team need to order 5 boxes of pantocid, also send over 2 packs of calpol asap.",
        height=150
    )
    
    process_btn = st.button("🚀 Process & Run Engine", type="primary")

with col2:
    st.subheader("📊 Output & Smart Diagnostics Dashboard")
    
    if process_btn:
        with st.spinner("Processing text via Gemini Flash 2.5 & matching local data arrays..."):
            
            # 1. Execute AI Pipeline Components
            extracted_stock = engine.parse_text(message_input)
            weather_data = engine.fetch_weather(pincode)
            alerts = engine.compute_recommendations(extracted_stock, weather_data)
            
            # 2. Display Parsed Metrics Layer
            st.markdown("#### 1. Gemini Structured Inventory Extraction")
            if extracted_stock:
                # Convert dict to clean key-value columns for UI display
                st.json(extracted_stock)
            else:
                st.info("No tracked items or generic mapping matches discovered in the phrase input.")
                
            st.markdown("---")
            
            # 3. Display Location/Environmental Layer
            st.markdown("#### 2. Hyper-local Predictive Vector Feeds")
            w_col1, w_col2, w_col3 = st.columns(3)
            w_col1.metric("Current Season", weather_data["current_season"])
            w_col2.metric("Rain Forecast (7D)", "🌧️ High Risk" if weather_data["rain_forecast_next_7_days"] else "☀️ Low Risk")
            w_col3.metric("Avg Temperature", f"{weather_data['avg_temp_celsius']} °C")
            
            st.markdown("---")
            
            # 4. Display Actionable Smart Upsell Recommendations Cards
            st.markdown("#### 3. Smart High-Margin Generic Recommendations (Data Moat)")
            if alerts:
                for alert in alerts:
                    with st.container(border=True):
                        st.markdown(f"📦 **Suggested Asset:** `{alert['generic_name']}` ({alert['category']})")
                        st.info(f"💡 **Why:** {alert['reason']}")
                        st.success(f"📈 **Target Inventory Allocation:** Pre-book **{alert['suggested_quantity']} boxes**")
                        st.caption(f"💰 **Pricing Status:** {alert['price_status']}")
                        
                        # Simulated interactive interface logic toggle
                        if st.button(f"Auto-Draft WhatsApp Alert for {chemist_name}", key=alert['generic_name']):
                            st.toast("WhatsApp integration alert payload successfully copied to outbound outbox buffer!")
            else:
                st.success("✅ Analytics complete: Local store inventory looks well-optimized for the upcoming seasonal metrics.")
    else:
        st.info("Fill out the pharmacy details on the left and click **'Process & Run Engine'** to simulate the live end-to-end fullstack execution cycle.")
