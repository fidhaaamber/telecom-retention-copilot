import json, os
import httpx
from api.schemas import RecommendationResponse

RULES = "Approved actions: service-quality investigation; plan and billing review; proactive care callback; equipment review. Never promise discounts, contact customers, or state unobserved facts."

def rule_recommendation(customer, probability, drivers):
    facts = [f"Customer ID: {customer.get('CustomerID', 'not supplied')}", f"Predicted churn probability: {probability:.1%}"] + [f"{d['feature']} {d['direction']}" for d in drivers[:3]]
    text = " ".join(facts)
    if any("Dropped" in d["feature"] or "Blocked" in d["feature"] for d in drivers): action = "Request a service-quality investigation"
    elif any("Overage" in d["feature"] or "Revenue" in d["feature"] for d in drivers): action = "Offer an analyst-led plan and billing review"
    elif any("Equipment" in d["feature"] for d in drivers): action = "Offer an analyst-led equipment review"
    else: action = "Queue an analyst-led retention review"
    return RecommendationResponse(recommended_action=action, rationale=text, observed_facts=facts, limitations="Decision support only. Validate account context and obtain analyst approval before customer contact or offers.")

async def recommend(customer, probability, drivers):
    fallback = rule_recommendation(customer, probability, drivers)
    if os.getenv("USE_OLLAMA", "false").lower() != "true": return fallback
    prompt = f"{RULES} Return JSON with recommended_action, rationale, observed_facts, limitations, requires_analyst_approval. Facts: {json.dumps(customer)}. Risk: {probability}; Drivers: {drivers}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"), json={"model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"), "prompt": prompt, "format": "json", "stream": False})
            r.raise_for_status()
        result = RecommendationResponse.model_validate(json.loads(r.json()["response"]))
        result.requires_analyst_approval = True
        return result
    except Exception:
        return fallback
