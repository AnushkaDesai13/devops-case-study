"""
main.py
Entry point – generates dataset, trains model, runs DeliveryAgent on a sample order.
"""

import os
from dataset_generator import generate_dataset
from model import train_model, load_model
from agent import DeliveryAgent

# ------------------------------------------------------------------
# Sample orders to demonstrate different agent paths
# ------------------------------------------------------------------
SAMPLE_ORDERS = [
    {
        "label":           "Premium Order",
        "distance_km":     5.0,
        "traffic_level":   "Medium",
        "weather":         "Clear",
        "prep_time":       15,
        "partner_rating":  4.5,
        "partner_available": 1,
        "premium_order":   1,
    },
    {
        "label":           "High-Traffic Order",
        "distance_km":     8.0,
        "traffic_level":   "High",
        "weather":         "Rainy",
        "prep_time":       20,
        "partner_rating":  4.0,
        "partner_available": 1,
        "premium_order":   0,
    },
    {
        "label":           "Normal Order",
        "distance_km":     3.5,
        "traffic_level":   "Low",
        "weather":         "Clear",
        "prep_time":       10,
        "partner_rating":  4.2,
        "partner_available": 1,
        "premium_order":   0,
    },
]


def run():
    print("=" * 55)
    print("  AI Food Delivery System – Agentic AI Demo")
    print("=" * 55)

    # 1. Generate dataset
    print("\n[1] Generating Dataset...")
    df = generate_dataset(n=500)

    # 2. Train or load model
    print("\n[2] Training Model...")
    model, encoders, _ = train_model(df)

    # 3. Run agent on each sample order
    print("\n[3] Running DeliveryAgent on sample orders...\n")
    for order in SAMPLE_ORDERS:
        label = order.pop("label")
        print(f"{'─'*50}")
        print(f"  ORDER TYPE : {label}")
        print(f"{'─'*50}")

        agent = DeliveryAgent(model, encoders)
        agent.observe_order(order)
        result = agent.act()

        print(f"\n   Assigned Partner  : {result['partner']}")
        print(f"    Predicted ETA     : {result['eta_min']} minutes")
        print(f"   Action Taken      : {result['action']}")
        print()

        order["label"] = label  # restore for display

    print("=" * 55)
    print("  Demo complete.")
    print("=" * 55)


if __name__ == "__main__":
    run()
