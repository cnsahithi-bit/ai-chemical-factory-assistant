# ai-chemical-factory-assistant
AI-powered industrial decision-support system for chemical manufacturing.
# AI-Powered Chemical Factory Assistant

## 1. Project Overview

The **AI-Powered Chemical Factory Assistant** is a software-based industrial decision-support system for a chemical manufacturing plant. It combines **product inspection, process monitoring, worker safety, production analysis, root-cause analysis, and profitability estimation** into a single platform.

The system does not directly control factory equipment. All process interventions and recommendations are **simulated and advisory**.

---

## 2. Factory Process

The factory is divided into multiple production and safety zones:

**Raw Material Storage → Mixing → Reaction → Separation → Packaging → Final Inspection**

Each stage generates process, production, batch and safety-related data.

---

## 3. Complete System Flow

### Step 1 — Collect Factory Data

The system collects available data from:

* Product inspection/images
* Batch and product information
* Process parameters such as temperature, pressure, pH and flow rate
* Machine/station cycle times
* Downtime and utilization
* Worker location and zone exposure
* Scrap, rework and production-cost information

---

### Step 2 — Monitor Worker Safety

Each production area is treated as a safety zone.

The system tracks:

* Worker presence in each zone
* Exposure level
* Time spent in the zone
* Cumulative exposure
* Exposure trends

If exposure becomes elevated, the system generates a **worker-safety alert** and provides actions based on the factory's configured safety procedures.

Example:

```text
Worker: W07
Zone: Reaction
Exposure: Increasing
Risk: High

Advisory:
Leave the affected area according to site procedure
and notify the responsible supervisor/EHS team.
```

---

### Step 3 — Inspect the Product

After production, the product is analyzed using visual inspection AI.

The system determines:

* Acceptable or defective
* Defect category
* Defect location
* Confidence level

If the system is uncertain or encounters a potentially novel defect, it flags the product for further review instead of forcing a prediction.

---

### Step 4 — Identify the Root Cause

For defective products, the system connects inspection results with:

**Batch + Process Parameters + Production History + Station Information**

Example:

```text
Defect detected
      ↓
Packaging Station P-04
      ↓
Abnormal sealing temperature
      ↓
Same condition observed in previous
defective batches
      ↓
Likely process condition identified
```

The system presents the relationship as evidence-based analysis rather than claiming certainty where the data is insufficient.

---

### Step 5 — Detect Production Bottlenecks

The system analyzes production flow using:

* Cycle time
* Station capacity
* Downtime
* Utilization
* Work-in-process
* Changeover time

It identifies stations that may be constraining production.

```text
Mixing       4 min
Reaction     8 min
Separation  18 min
Packaging    6 min

Potential bottleneck:
Separation Station
```

---

### Step 6 — Estimate Production and Economic Impact

The system calculates the potential effect of identified problems on:

* Throughput
* Scrap
* Rework
* Downtime
* Production capacity
* Operating cost
* Expected profitability/margin

This connects a technical problem to its business impact.

---

### Step 7 — Generate Recommendations

The system combines the results from quality, safety and production analysis.

For example:

```text
Problem:
Repeated seal defects

Location:
Packaging Station P-04

Likely associated condition:
Abnormal sealing temperature

Production impact:
Increased rework

Economic impact:
Higher scrap and rework cost

Safety:
No abnormal exposure detected

Recommendation:
Investigate the sealing process and
simulate correction of the operating condition.
```

---

## 4. Unified Decision Dashboard

The final dashboard provides a continuously updated view of:

| Area              | Output                                |
| ----------------- | ------------------------------------- |
| Product Quality   | Defect status, type and location      |
| Worker Safety     | Exposure and safety risk              |
| Root Cause        | Process/batch relationships           |
| Production        | Bottlenecks and throughput            |
| Economics         | Scrap, rework, cost and margin impact |
| AI Recommendation | Evidence-based advisory action        |
| Confidence        | Prediction confidence and uncertainty |

---

## 5. AI Architecture

```text
Factory Data
     ↓
Data Processing & Feature Engineering
     ↓
 ┌──────────────┬──────────────┬──────────────┐
 │              │              │              │
Quality AI   Safety AI     Process AI    Production AI
 │              │              │              │
 └──────────────┴──────────────┴──────────────┘
                    ↓
             Root-Cause Analysis
                    ↓
          Bottleneck & Flow Analysis
                    ↓
          Cost & Profitability Analysis
                    ↓
           Recommendation Engine
                    ↓
               Dashboard
```

---

## 6. Technology Stack

**Backend:** Python, Flask/FastAPI
**Machine Learning:** Scikit-learn
**Computer Vision:** OpenCV, YOLO or suitable detection model
**Data Processing:** Pandas, NumPy
**Frontend:** HTML, CSS, JavaScript
**Model Storage:** Joblib / suitable model format

---

## 7. Key Objective

The system provides one unified view of the factory and answers:

**What went wrong? → Where did it happen? → Why did it happen? → Is production or worker safety affected? → What is the economic impact? → What action should be considered?**

The final goal is to support **better product quality, safer operations, improved production flow and informed economic decisions**.
