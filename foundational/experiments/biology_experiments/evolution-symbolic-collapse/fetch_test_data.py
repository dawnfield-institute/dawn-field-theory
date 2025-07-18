import requests
import pandas as pd
import random
import os

# --- Config ---
DATA_DIR = "c:\\Users\\peter\\repos\\dawn-field-theory\\biology_tests\\evolution-symbolic-collapse\\data"
TREE_NWK_PATH = os.path.join(DATA_DIR, "tree.nwk")
EXTINCTIONS_CSV_PATH = os.path.join(DATA_DIR, "extinctions.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# --- Fetch a small Newick tree from OpenTree ---
def fetch_newick_tree():
    url = "https://tree.opentreeoflife.org/opentree/phylo_snapshot/ot_ol_2022_04/ot_ol_2022_04.tre"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # Use only a small subtree for testing
        newick_str = response.text.split(";")[0] + ";"
        with open(TREE_NWK_PATH, "w", encoding="utf-8") as f:
            f.write(newick_str)
        print(f"[fetch_test_data] Newick tree saved to {TREE_NWK_PATH}")
    except Exception as e:
        print(f"[fetch_test_data] Failed to fetch Newick tree: {e}")
        # Fallback: generate a tiny synthetic tree
        synthetic = "(A,B,(C,D));"
        with open(TREE_NWK_PATH, "w", encoding="utf-8") as f:
            f.write(synthetic)
        print(f"[fetch_test_data] Synthetic tree saved to {TREE_NWK_PATH}")

# --- Generate synthetic extinction data ---
def generate_extinction_csv():
    # Use leaf names from synthetic tree or fallback
    leaf_names = [f"Species_{i}" for i in range(1, 10)] + ["A", "B", "C", "D"]
    extinction_data = {
        "accepted_name": random.sample(leaf_names, k=min(5, len(leaf_names)))
    }
    df = pd.DataFrame(extinction_data)
    df.to_csv(EXTINCTIONS_CSV_PATH, index=False, encoding="utf-8")
    print(f"[fetch_test_data] Extinction CSV saved to {EXTINCTIONS_CSV_PATH}")

if __name__ == "__main__":
    fetch_newick_tree()
    generate_extinction_csv()
    print("[fetch_test_data] Test data generation complete.")
