# AWS Scrutiny Pipeline Plan (Dockerized)

## 1. Objectives

* **Automate metadata generation** (update meta.yamls with semantic descriptions and tags).
* **Scrutinize theoretical papers** using multiple models (Ollama + API-based LLMs like GPT, Claude, Gemini).
* **Deploy on AWS** for cost-effective GPU and CPU workloads.
* **Ensure Dockerization** of all components for portability (cloud or local VMs).
* **Selective Scrutiny** using `.ignore` rules to exclude non-theoretical folders.

---

## 2. Core Architecture

### Components

1. **AWS EC2 Instances:**

   * **CPU Instance (t3.small or t3.medium):** Handles Git workflows, metadata updates, and light pre-processing.
   * **GPU Instance (g4dn.xlarge or on-demand spot instance with RTX T4/3090):** Runs Ollama for local model inference.

2. **Ollama Environment (Dockerized):**

   * Base image with Ollama runtime and pre-pulled models (e.g., LLaMA3, Mistral).
   * Docker container exposing Ollama API on port `11434`.

3. **Multi-Model Scrutiny Pipeline:**

   * Python script running inside Docker.
   * Calls:

     * **Ollama API** for local inference.
     * **Cloud APIs (GPT-4, Claude, Gemini)** via secure API keys.
   * Generates consolidated scrutiny reports for each paper.

4. **Git Integration:**

   * Detect changes via `git diff` or GitHub webhook.
   * `.ignore` rules to exclude software or documentation folders.
   * Commits updated `meta.yaml` files with semantic scope + tags.

5. **Docker Compose Setup:**

   * `docker-compose.yml` to bring up CPU + GPU containers.
   * One container for metadata pipelines, one for scrutiny pipeline.
   * Shared volumes for model storage and repo data.

---

## 3. Workflow

### Step-by-Step Flow

1. **Detect Changes:**

   * `watchdog` or `git diff` identifies changed theoretical folders.
2. **Trigger Pipeline:**

   * CPU container prepares folder content.
   * GPU container (Ollama) performs LLM-based semantic analysis.
   * Additional scrutiny requests go to GPT/Claude/Gemini APIs.
3. **Coalesce Outputs:**

   * Pipeline merges all model critiques into a single scrutiny report.
4. **Update Metadata:**

   * `meta.yaml` is updated with descriptions/tags based on analysis.
5. **Push to Git:**

   * Changes are committed and pushed automatically.
6. **(Optional) Scheduled Runs:**

   * AWS EventBridge triggers daily/weekly metadata audits.

---

## 4. Dockerization Plan

### Docker Images

* **Base Image:** `ubuntu:22.04` or `nvidia/cuda:12.2.0-runtime` (for GPU).
* **Ollama Container:** Includes Ollama binary + models stored in `/models` volume.
* **Pipeline Container:** Python 3.11 + `watchdog`, `requests`, `pyyaml`, `boto3`.
* **Compose File:**

  ```yaml
  version: '3.9'
  services:
    pipeline:
      build: ./pipeline
      volumes:
        - ./repo:/app/repo
        - ./models:/app/models
      depends_on:
        - ollama
    ollama:
      build: ./ollama
      ports:
        - "11434:11434"
      runtime: nvidia
      environment:
        - NVIDIA_VISIBLE_DEVICES=all
  ```

### Local Development

* Docker setup ensures same pipeline can run on:

  * **AWS EC2**
  * **Local workstation/VMs**
  * **Hybrid workflows** (CPU locally, GPU in cloud)

---

## 5. Cost Optimization

* **GPU:** Use **spot instances** or **on-demand RunPod-like GPU** when running heavy scrutiny.
* **CPU:** Keep small EC2 instance for background tasks (\~\$12/month).
* **Runtime:** Shut down GPU container when not needed (via `docker-compose down`).
* **Multi-model mix:** Use Ollama for cheap local inference + API for high-quality model checks.

Estimated **cost per paper:** \$0.10–\$0.20 (10-min scrutiny across multiple models).

---

## 6. Next Steps

1. **Set up AWS EC2 (CPU + GPU).**
2. **Build Docker images for Ollama and pipeline.**
3. **Implement `.ignore` logic in pipeline scripts.**
4. **Integrate cloud APIs (GPT/Claude/Gemini) with secure key storage.**
5. **Test end-to-end flow with 1–2 sample papers.**
6. **Add GitHub webhook to auto-trigger pipeline on commit.**

---

## 7. Future Enhancements

* Add a **web dashboard** (Flask + Docker) to track scrutiny results.
* Implement **parallel multi-model scrutiny** (Ray/Dask).
* Archive and version all reports in **S3 buckets**.
* Introduce **automated PRs** with updated meta.yamls and scrutiny reports.

---

**This plan is fully Dockerized and AWS-ready, with a focus on cost efficiency, portability, and multi-model integration.**
