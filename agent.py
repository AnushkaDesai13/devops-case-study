"""
agent.py
Simple Agentic AI for food delivery partner assignment and decision-making.
"""

from model import predict_delivery_time

# Simulated pool of available delivery partners
PARTNER_POOL = [
    {"id": "P1", "name": "Rahul",  "rating": 4.9, "distance_km": 1.2, "available": 1},
    {"id": "P2", "name": "Sneha",  "rating": 4.5, "distance_km": 3.4, "available": 1},
    {"id": "P3", "name": "Arjun",  "rating": 3.8, "distance_km": 2.1, "available": 0},
    {"id": "P4", "name": "Priya",  "rating": 4.2, "distance_km": 0.9, "available": 1},
    {"id": "P5", "name": "Vikram", "rating": 4.7, "distance_km": 4.5, "available": 1},
]


class DeliveryAgent:
    """
    Agentic AI that:
      1. Observes the incoming order
      2. Chooses the best delivery partner
      3. Predicts delivery time via ML model
      4. Decides the action
      5. Acts / logs reasoning
    """

    def __init__(self, model, encoders):
        self.model    = model
        self.encoders = encoders
        self.logs     = []          # reasoning trace

    # ------------------------------------------------------------------
    # Step 1 – Perceive
    # ------------------------------------------------------------------
    def observe_order(self, order: dict):
        self.order = order
        self._log(f"Observed order → distance={order['distance_km']} km, "
                  f"traffic={order['traffic_level']}, weather={order['weather']}, "
                  f"premium={bool(order['premium_order'])}")

    # ------------------------------------------------------------------
    # Step 2 – Choose partner
    # ------------------------------------------------------------------
    def choose_partner(self) -> dict | None:
        available = [p for p in PARTNER_POOL if p["available"] == 1]

        if not available:
            self._log("No partners available.")
            return None

        if self.order.get("premium_order") == 1:
            partner = max(available, key=lambda p: p["rating"])
            self._log(f"Premium order → selected highest-rated partner: "
                      f"{partner['name']} (rating {partner['rating']})")

        elif self.order.get("traffic_level") == "High":
            partner = min(available, key=lambda p: p["distance_km"])
            self._log(f"High traffic → selected nearest partner: "
                      f"{partner['name']} ({partner['distance_km']} km away)")

        else:
            # Normal: pick best available rating
            partner = max(available, key=lambda p: p["rating"])
            self._log(f"Normal order → selected best-rated available partner: "
                      f"{partner['name']} (rating {partner['rating']})")

        return partner

    # ------------------------------------------------------------------
    # Step 3 – Predict time
    # ------------------------------------------------------------------
    def predict_time(self, partner: dict) -> float:
        # Use partner's distance for ML prediction
        order_for_pred = {**self.order, "distance_km": partner["distance_km"],
                          "partner_rating": partner["rating"],
                          "partner_available": partner["available"]}
        eta = predict_delivery_time(order_for_pred, self.model, self.encoders)
        self._log(f"ML predicted delivery time: {eta} minutes")
        return eta

    # ------------------------------------------------------------------
    # Step 4 – Decide
    # ------------------------------------------------------------------
    def decide(self, partner, eta) -> str:
        if partner is None:
            decision = "WAIT_AND_RETRY"
        elif eta > 60:
            decision = "NOTIFY_CUSTOMER_DELAY"
        elif self.order.get("premium_order") == 1:
            decision = "PRIORITY_DISPATCH"
        else:
            decision = "STANDARD_DISPATCH"

        self._log(f"Decision → {decision}")
        return decision

    # ------------------------------------------------------------------
    # Step 5 – Act
    # ------------------------------------------------------------------
    def act(self) -> dict:
        partner  = self.choose_partner()
        eta      = self.predict_time(partner) if partner else None
        decision = self.decide(partner, eta)

        result = {
            "partner":  partner["name"] if partner else "None",
            "eta_min":  eta,
            "action":   decision,
            "logs":     self.logs,
        }
        return result

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.logs.append(f"  [Agent] {msg}")
        print(f"  [Agent] {msg}")
