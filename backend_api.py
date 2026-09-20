import os
import json
import math
import shutil
import tempfile
import traceback
import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import basic


# ============================================================
# NaN / Infinity CLEANER
# ============================================================
# Converts ANY object into something that is strictly JSON safe.
# - NaN / +Inf / -Inf  -> None (null)
# - DataFrame / Series -> dict / list of dicts (then cleaned)
# - numpy / torch      -> python lists / scalars (then cleaned)
# - anything unknown   -> str(obj)  (so nothing can leak through)

def clean(obj):

    if obj is None:
        return None

    # pandas missing-value singletons
    if obj is pd.NA or obj is pd.NaT:
        return None

    # bool MUST be checked before int (bool is a subclass of int)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None

    if isinstance(obj, Decimal):
        value = float(obj)
        return value if math.isfinite(value) else None

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}

    if isinstance(obj, pd.DataFrame):
        return clean(obj.to_dict(orient="records"))

    if isinstance(obj, pd.Series):
        return clean(obj.to_dict())

    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())

    if isinstance(obj, (list, tuple, set, frozenset)):
        return [clean(v) for v in obj]

    if isinstance(obj, (dt.datetime, dt.date, pd.Timestamp)):
        return obj.isoformat()

    # torch tensors (without importing torch)
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            return clean(obj.detach().cpu().numpy())
        except Exception:
            pass

    # last resort: never let an unknown object reach the encoder
    return str(obj)


# ============================================================
# SAFE JSON RESPONSE
# ============================================================
# Even if something slips past the endpoint code, this cleans the
# payload and serializes with allow_nan=False, so the
# "Out of range float values are not JSON compliant" error
# can no longer happen.

class SafeJSONResponse(JSONResponse):

    def render(self, content) -> bytes:
        return json.dumps(
            clean(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SentinelAI Backend",
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD SENTINELAI
# ============================================================

try:
    CTX = basic.build_context()
    print("SentinelAI context loaded successfully.")

except Exception as e:
    CTX = None
    print("ERROR loading SentinelAI:")
    traceback.print_exc()


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return SafeJSONResponse({
        "status": "ok" if CTX is not None else "error",
        "backend": "SentinelAI"
    })


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard():

    if CTX is None:
        raise HTTPException(
            status_code=500,
            detail="SentinelAI context was not loaded."
        )

    try:

        prod = CTX.get("prod", {})

        response = {
            "primary_bottleneck": prod.get("primary"),
            "secondary_constraint": prod.get("secondary"),
            "average_input": prod.get("avg_in"),
            "average_output": prod.get("avg_out"),
            "average_throughput_gap": prod.get("avg_gap"),
            "average_throughput_loss_pct": prod.get("avg_loss_pct"),
            "average_wip": prod.get("avg_wip"),
            "high_utilization_runs": prod.get("high_util", {}),
            "zone_utilization": {},
            "zone_summary": [],
            "bottleneck_ranking": []
        }

        # Zone summary
        try:
            zone_summary = prod.get("zone_summary")

            if isinstance(zone_summary, pd.DataFrame):

                response["zone_summary"] = zone_summary.to_dict(orient="records")

                for _, row in zone_summary.iterrows():
                    zone = str(row["Zone"])
                    response["zone_utilization"][zone] = row["Average_Utilization_Percent"]

        except Exception as e:
            print("Zone dashboard data skipped:", e)

        # Ranking
        try:
            ranking = prod.get("ranking")

            if isinstance(ranking, pd.DataFrame):
                response["bottleneck_ranking"] = ranking.to_dict(orient="records")

            elif isinstance(ranking, list):
                response["bottleneck_ranking"] = ranking

        except Exception as e:
            print("Ranking skipped:", e)

        # Returned directly -> bypasses FastAPI's own JSON encoder
        return SafeJSONResponse(response)

    except Exception as e:

        print("Dashboard error:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}"
        )


# ============================================================
# INSPECT ONE IMAGE
# ============================================================

@app.post("/api/inspect")
async def inspect(
    file: UploadFile = File(...)
):

    if CTX is None:
        raise HTTPException(
            status_code=500,
            detail="SentinelAI context was not loaded."
        )

    temp_dir = tempfile.mkdtemp()

    try:

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        extension = os.path.splitext(file.filename or "")[1] or ".jpg"

        image_path = os.path.join(temp_dir, "uploaded" + extension)

        data = await file.read()

        with open(image_path, "wb") as f:
            f.write(data)

        print()
        print("========================================")
        print("Running image inspection...")
        print("Image:", file.filename)
        print("========================================")

        # ----------------------------------------------------
        # YOUR EXISTING AI FUNCTION
        # ----------------------------------------------------

        result = basic.inspect_product(
            0,
            CTX,
            image_path=image_path,
            heatmap=False,
            verbose=False
        )

        print("Inspection completed.")

        # ----------------------------------------------------
        # CLEAN ALL NaN / Infinity VALUES
        # ----------------------------------------------------

        result = clean(result)

        if not isinstance(result, dict):
            raise ValueError(
                f"inspect_product returned {type(result).__name__}, expected a dict"
            )

        # ----------------------------------------------------
        # BUILD RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "decision": result.get("Decision"),

            "decision_text": basic.DECISION_TEXT.get(
                result.get("Decision"),
                result.get("Decision")
            ),

            "top_class": result.get("Top_Class"),
            "defect": result.get("Defect"),
            "yolo_confidence": result.get("YOLO_Confidence"),
            "p_defect": result.get("P_Defect"),
            "margin": result.get("Margin_Top1_Top2"),
            "entropy": result.get("Entropy"),
            "type_confidence": result.get("Type_Confidence"),
            "novel_flag": result.get("Novel_Flag"),
            "decision_reason": result.get("decision_reason"),

            "likely_zone": result.get("Likely_Zone"),
            "second_zone": result.get("Second_Zone"),
            "zone_confidence": result.get("Zone_Confidence"),
            "zone_status": result.get("Zone_Status"),
            "zone_reason": result.get("Reason"),
            "evidence": result.get("Evidence"),

            "run_bottleneck_zone": result.get("Run_Bottleneck_Zone"),
            "run_throughput_loss_pct": result.get("Run_Throughput_Loss_Pct"),
            "run_wip_stored": result.get("Run_WIP_Stored"),
            "run_profit": result.get("Run_Profit"),
            "run_margin_pct": result.get("Run_Margin_Pct"),
            "predicted_margin_pct": result.get("Predicted_Margin_Pct"),

            "localization_supported": result.get("Localization_Supported"),
            "localization_box": result.get("Localization_Box"),
            "localization_area": result.get("Localization_Area"),

            "analysis": (
                result.get("Evidence")
                or result.get("Reason")
                or result.get("decision_reason")
            ),

            "recommendations": [
                x.strip()
                for x in str(result.get("Recommendations", "")).split("|")
                if x.strip()
            ],

            "raw": result
        }

        print()
        print("SUCCESS")
        print("Class:", response["top_class"])
        print("Decision:", response["decision"])
        print("Zone:", response["likely_zone"])
        print()

        # Returned directly -> bypasses FastAPI's own JSON encoder
        return SafeJSONResponse(response)

    except Exception as e:

        print()
        print("========================================")
        print("INSPECTION ERROR")
        print("========================================")
        traceback.print_exc()   # full traceback so we can see WHERE it failed
        print("========================================")
        print()

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}"
        )

    finally:

        # Remove temp folder and everything inside it
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print()
    print("========================================")
    print("       SENTINELAI BACKEND")
    print("========================================")
    print()
    print("Backend:")
    print("http://127.0.0.1:8000")
    print()
    print("Health:")
    print("http://127.0.0.1:8000/api/health")
    print()
    print("Docs:")
    print("http://127.0.0.1:8000/docs")
    print()
    print("========================================")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )