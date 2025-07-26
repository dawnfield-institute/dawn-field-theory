# 🧠 How to Use the DawnField GPT

The **DawnField GPT** is a custom AI assistant built specifically to help users explore and understand the [`dawn-field-theory`](https://github.com/dawnfield-institute/dawn-field-theory) repository. It uses intelligent metadata (from `.cip/meta.yaml`, versioned instructions, and `map.yaml`) to navigate the codebase like a researcher — not a search engine.

---

## 🚀 Try It Now

> [🧠 Launch DawnField GPT v0.2](https://chatgpt.com/g/g-685b2fa488888191bab9a5d7bfc38ca9-dawnfieldframeworkrepogptv0-3)

This is the live GPT agent built to navigate and reason through this repository.

---

## ⚙️ What It Can Do

* ✅ Understand the structure of the repo using a semantic map (`map.yaml`)
* ✅ Loads `.cip/meta.yaml` to determine the current instructions version, then loads the specified instructions file from `.cip` (e.g., `instructions_v2.0.yaml`) for schema and navigation guidance
* ✅ Loads only the relevant files (and their subdirectories) based on your question, using efficient batch/cluster API calls
* ✅ Answers with grounded insights from real theory, code, and experiments
* ✅ Adapts its reasoning using context weight, semantic tags, and your query intent

---

## 👉 How It Works (Under the Hood)

1. **Reads `.cip/meta.yaml`** to determine the current instructions version
2. **Loads the specified instructions file** from `.cip` (e.g., `instructions_v2.0.yaml`) for schema and navigation guidance
3. **Uses `map.yaml`** to understand the overall directory structure and locate files
4. **Identifies relevant files and subdirectories** to your question based on semantic tags and metadata
5. **Loads those files (and their subdirectories)** via efficient batch/cluster API calls
6. **Explains or reasons** based on what it ingested — with no guessing
7. *(Later phases will include validation testing via CIP, but that’s not enabled yet)*

---

## 🗣️ Example Questions to Ask the GPT

* “How does the QBE model regulate entropy in this system?”
* “What does the recursive tree experiment reveal about structural coherence?”
* “List the most important theory documents in this repo and summarize them.”
* “Which models are aligned with the principles in `infodynamics.md`?”
* “Walk me through the neural architecture of `entropy_monitor.py`.”

---

## 🔐 Data Privacy

This GPT does **not** collect or transmit any personal data. All code it reads comes from the public GitHub repository. See [Privacy Policy](./privacy-policy.md) for details.

---

## 📌 License and Usage

All use of this GPT and the content it accesses is bound by the repository’s license. Please refer to the [LICENSE](./../../LICENSE) for terms of use and restrictions.

---

## 🗺️ Where to Start

> 👉 **Just ask the GPT a question about the repo** — it will dynamically read only what it needs to answer intelligently.

---
