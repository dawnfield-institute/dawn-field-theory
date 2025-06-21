import math
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
from datetime import datetime

MIN_DELTA = 0.2
MAX_LABEL_LEN = 30

class CollapseNode:
    def __init__(self, text):
        self.text = text.strip()
        self.entropy = self.compute_entropy()
        self.logic_id = self.generate_logic_id()
        self.children = []
        self.role = None

    def compute_entropy(self):
        char_freq = defaultdict(int)
        for char in self.text.lower():
            if char.isalpha():
                char_freq[char] += 1
        total = sum(char_freq.values())
        if total == 0:
            return 0
        probs = [freq / total for freq in char_freq.values()]
        return round(-sum(p * math.log2(p) for p in probs), 3)

    def generate_logic_id(self):
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()[:10]

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        return {
            "logic_id": self.logic_id,
            "text": self.text,
            "entropy": self.entropy,
            "role": self.role,
            "children": [child.to_dict() for child in self.children]
        }

def entropy_collapse_score(phrase):
    char_freq = defaultdict(int)
    for char in phrase.lower():
        if char.isalpha():
            char_freq[char] += 1
    total = sum(char_freq.values())
    if total == 0:
        return 0
    probs = [freq / total for freq in char_freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    return round(entropy, 3)

def entropy_gradient_split(sentence):
    words = sentence.split()
    if len(words) < 3:
        return [sentence]

    scores = []
    for i in range(1, len(words) - 1):
        left = ' '.join(words[:i])
        right = ' '.join(words[i:])
        delta = abs(entropy_collapse_score(left) - entropy_collapse_score(right))
        scores.append((i, delta))

    if not scores:
        return [sentence]
    best_split_index, best_split_score = max(scores, key=lambda x: x[1])
    if best_split_score < MIN_DELTA:
        return [sentence]
    return [' '.join(words[:best_split_index]), ' '.join(words[best_split_index:])]

def infer_field_roles(node, depth=0):
    if not node.children:
        node.role = "end" if node.entropy < 2.0 else "condition"
    elif len(node.children) == 2:
        left, right = node.children
        if node.entropy > max(left.entropy, right.entropy):
            node.role = "driver"
        elif node.entropy < min(left.entropy, right.entropy):
            node.role = "transform"
        else:
            node.role = "junction"
    else:
        node.role = "undetermined"
    for child in node.children:
        infer_field_roles(child, depth + 1)

def extract_logic_flow(node):
    def walk(node):
        entry = {
            "id": node.logic_id,
            "text": node.text,
            "role": node.role,
            "entropy": node.entropy,
            "children": [walk(child) for child in node.children]
        }
        return entry
    return walk(node)

def collect_instruction_trace(node):
    trace = []
    def walk(node, indent=0):
        prefix = "  " * indent
        line = ""
        if node.role == "condition":
            line = f"{prefix}IF '{node.text}' THEN"
        elif node.role == "driver":
            line = f"{prefix}DO '{node.text}'"
        elif node.role == "transform":
            line = f"{prefix}TRANSFORM '{node.text}'"
        elif node.role == "junction":
            line = f"{prefix}JUNCTION '{node.text}'"
        elif node.role == "end":
            line = f"{prefix}END '{node.text}'"
        else:
            line = f"{prefix}UNKNOWN '{node.text}'"
        trace.append(line)
        for child in node.children:
            walk(child, indent + 1)
    walk(node)
    return trace

def print_instruction_trace(node, indent=0):
    trace = collect_instruction_trace(node)
    for line in trace:
        print(line)

def fractal_entropy_parser(sentence, depth=0, max_depth=4):
    sentence = sentence.strip()
    node = CollapseNode(sentence)

    if depth >= max_depth or len(sentence.split()) <= 2:
        return node

    parts = entropy_gradient_split(sentence)
    if len(parts) == 2:
        node.add_child(fractal_entropy_parser(parts[0], depth + 1))
        node.add_child(fractal_entropy_parser(parts[1], depth + 1))

    return node

def visualize_collapse_tree(node, save_path=None):
    def build_graph(node, G=None, parent=None, level=0):
        if G is None:
            G = nx.DiGraph()
        text_label = node.text[:MAX_LABEL_LEN] + '...' if len(node.text) > MAX_LABEL_LEN else node.text
        label = f"{text_label}\nE:{node.entropy} | {node.role}"
        G.add_node(label, level=level)
        if parent:
            G.add_edge(parent, label)
        for child in node.children:
            build_graph(child, G, label, level + 1)
        return G

    G = build_graph(node)
    pos = nx.multipartite_layout(G, subset_key="level")
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=9, font_weight='bold')
    plt.title("Fractal Entropy Collapse Tree")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def generate_entropy_heatmap(sentence, save_path):
    words = sentence.split()
    if not words:
        return
    entropies = []
    for i in range(len(words)):
        segment = ' '.join(words[:i+1])
        entropies.append(entropy_collapse_score(segment))
    plt.figure(figsize=(10, 2))
    plt.plot(range(len(entropies)), entropies, marker='o')
    plt.xticks(range(len(entropies)), words, rotation=45, ha='right')
    plt.title("Entropy Gradient Heatmap")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_markdown_report(sentence, logic_flow, logic_trace, save_path):
    with open(save_path, 'w') as md:
        md.write(f"# Entropy Logic Report\n\n")
        md.write(f"**Input Sentence:** {sentence}\n\n")
        md.write(f"## Instruction Trace\n")
        for line in logic_trace:
            md.write(f"- {line}\n")
        md.write(f"\n## Top Node Summary\n")
        md.write(f"- Role: {logic_flow['role']}\n")
        md.write(f"- Entropy: {logic_flow['entropy']}\n")
        md.write(f"- Text: `{logic_flow['text']}`\n")

# ----------------------------
# Main execution block
# ----------------------------
if __name__ == "__main__":
    test_sentences = [
        ("testid_1", "If the report isn’t finalized by noon, tell the team to move forward without it and update the timeline accordingly."),
        ("testid_2", "Before you leave for lunch, make sure all systems are shut down and the logs are archived."),
        ("testid_3", "It would be great if you could send over the updated draft once you’re done reviewing it."),
        ("testid_4", "Since the client hasn’t responded, we might need to revise the proposal or consider postponing the launch."),
        ("testid_5", "Although the system is reporting errors, the backup routines appear to be functioning correctly.")
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_dir = os.path.join("reference_material", timestamp)
    os.makedirs(base_dir, exist_ok=True)

    index_summary = {}

    for testid, sentence in test_sentences:
        root = fractal_entropy_parser(sentence)
        infer_field_roles(root)
        logic_flow = extract_logic_flow(root)
        logic_trace = collect_instruction_trace(root)

        output_dir = os.path.join(base_dir, testid)
        os.makedirs(output_dir, exist_ok=True)

        logic_file = os.path.join(output_dir, "logic_flow.json")
        with open(logic_file, "w") as f:
            json.dump({"tree": logic_flow, "logic_trace": logic_trace}, f, indent=2)

        img_file = os.path.join(output_dir, "collapse_tree.png")
        visualize_collapse_tree(root, save_path=img_file)

        heatmap_file = os.path.join(output_dir, "entropy_heatmap.png")
        generate_entropy_heatmap(sentence, save_path=heatmap_file)

        md_file = os.path.join(output_dir, "report.md")
        generate_markdown_report(sentence, logic_flow, logic_trace, md_file)

        index_summary[testid] = {
            "sentence": sentence,
            "role": logic_flow['role'],
            "entropy": logic_flow['entropy'],
            "path": output_dir
        }

        print("\nInstruction Trace:")
        print_instruction_trace(root)
        print(f"\nSaved to: {output_dir}\n")

    index_file = os.path.join(base_dir, "index.json")
    with open(index_file, "w") as idx:
        json.dump(index_summary, idx, indent=2)
