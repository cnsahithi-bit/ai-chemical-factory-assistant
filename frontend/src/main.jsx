import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Factory,
  Gauge,
  Image as ImageIcon,
  RefreshCw,
  Search,
  ShieldCheck,
  Upload,
  XCircle,
  Boxes,
  Clock3
} from "lucide-react";

import "./styles.css";

/*
  SENTINELAI FRONTEND
  ---------------------------------------------------------
  Connected to the existing FastAPI + Python backend.

  Backend endpoints:

    GET  http://127.0.0.1:8000/api/dashboard
    GET  http://127.0.0.1:8000/api/health
    POST http://127.0.0.1:8000/api/inspect
    POST http://127.0.0.1:8000/api/batch?n=20
*/

const API_BASE = "http://127.0.0.1:8000";


/* =========================================================
   DEMO DATA
   ========================================================= */

const DEMO_DASHBOARD = {
  primary_bottleneck: "Drilling",
  secondary_constraint: "Milling",

  average_input: 100,
  average_output: 94,
  average_throughput_gap: 6,
  average_throughput_loss_pct: 6.0,
  average_wip: 12,

  high_utilization_runs: {
    Drilling: 18,
    Milling: 11,
    Assembly: 7
  },

  zone_summary: [
    {
      Zone: "Drilling",
      Average_Utilization_Percent: 91.4,
      Average_Waiting_Time: 8.2,
      Runs_Above_90_Percent: 18
    },
    {
      Zone: "Milling",
      Average_Utilization_Percent: 84.7,
      Average_Waiting_Time: 6.1,
      Runs_Above_90_Percent: 11
    },
    {
      Zone: "Assembly",
      Average_Utilization_Percent: 76.8,
      Average_Waiting_Time: 4.2,
      Runs_Above_90_Percent: 7
    }
  ],

  bottleneck_ranking: [
    {
      Zone: "Drilling",
      Bottleneck_Percentage: 52.0
    },
    {
      Zone: "Milling",
      Bottleneck_Percentage: 31.0
    },
    {
      Zone: "Assembly",
      Bottleneck_Percentage: 17.0
    }
  ]
};


/* =========================================================
   API HELPER
   ========================================================= */

function api(path, options = {}) {
  return fetch(`${API_BASE}${path}`, options);
}


/* =========================================================
   SMALL UI COMPONENTS
   ========================================================= */

function Metric({ label, value, icon: Icon, tone = "" }) {
  return (
    <div className={`metric ${tone}`}>
      <div className="metricIcon">
        <Icon size={18} />
      </div>

      <div>
        <div className="metricLabel">
          {label}
        </div>

        <div className="metricValue">
          {value}
        </div>
      </div>
    </div>
  );
}


function Decision({ value }) {
  const v = value || "REVIEW";

  const cls =
    v === "PASS"
      ? "pass"
      : v === "REJECT" || v === "REJECT_NOVEL"
        ? "reject"
        : "review";

  const Icon =
    cls === "pass"
      ? CheckCircle2
      : cls === "reject"
        ? XCircle
        : AlertTriangle;

  const label =
    v === "REJECT_NOVEL"
      ? "REJECT · NOVEL"
      : v;

  return (
    <span className={`decision ${cls}`}>
      <Icon size={15} />
      {label}
    </span>
  );
}


function Panel({ title, icon: Icon, children, action }) {
  return (
    <div className="panel">

      <div className="panelHeader">

        <h3>
          {Icon && <Icon size={16} />}
          {title}
        </h3>

        {action}

      </div>

      {children}

    </div>
  );
}


function ProgressBar({ label, value }) {

  const n = Number(value) || 0;

  return (
    <div className="barWrap">

      <div className="barLabel">
        <span>{label}</span>

        <b>
          {n.toFixed(1)}%
        </b>
      </div>

      <div className="bar">
        <i
          style={{
            width: `${Math.min(
              Math.max(n, 0),
              100
            )}%`
          }}
        />
      </div>

    </div>
  );
}


function DataRow({ label, value }) {
  return (
    <div className="dataRow">

      <span>
        {label}
      </span>

      <strong>
        {value ?? "—"}
      </strong>

    </div>
  );
}


/* =========================================================
   MAIN APP
   ========================================================= */

function App() {

  const [tab, setTab] =
    useState("overview");

  const [dashboard, setDashboard] =
    useState(null);

  const [inspection, setInspection] =
    useState(null);

  const [preview, setPreview] =
    useState(null);

  const [batch, setBatch] =
    useState([]);

  const [loading, setLoading] =
    useState(false);

  const [backendOnline, setBackendOnline] =
    useState(false);

  const [error, setError] =
    useState("");

  const [usingDemo, setUsingDemo] =
    useState(false);


  const data =
    dashboard || DEMO_DASHBOARD;

  const zones =
    data.zone_summary || [];


  /* =======================================================
     LOAD DASHBOARD
     ======================================================= */

  async function loadDashboard() {

    try {

      setError("");

      const response =
        await api("/api/dashboard");

      if (!response.ok) {
        throw new Error(
          "Dashboard API unavailable"
        );
      }

      const json =
        await response.json();

      setDashboard(json);

      setBackendOnline(true);

      setUsingDemo(false);

    } catch (err) {

      console.error(
        "Dashboard error:",
        err
      );

      setBackendOnline(false);

      setUsingDemo(true);

      setDashboard(
        DEMO_DASHBOARD
      );
    }
  }


  useEffect(() => {

    loadDashboard();

  }, []);


  /* =======================================================
     SINGLE IMAGE INSPECTION
     ======================================================= */

  async function inspectProduct(file) {

    if (!file) {
      return;
    }


    /* Show image immediately */
    setPreview(
      URL.createObjectURL(file)
    );

    setInspection(null);

    setLoading(true);

    setError("");


    /* -------------------------------------------------------
       IMPORTANT FIX

       FastAPI backend expects:

           file: UploadFile

       NOT:

           image: UploadFile
       ------------------------------------------------------- */

    const form =
      new FormData();

    form.append(
      "file",
      file
    );


    try {

      const response =
        await api(
          "/api/inspect",
          {
            method: "POST",
            body: form
          }
        );


      if (!response.ok) {

        const errorText =
          await response.text();

        throw new Error(
          errorText
        );
      }


      const json =
        await response.json();


      /* -----------------------------------------------------
         BACKEND RETURNS LOWERCASE FIELD NAMES.

         FRONTEND UI USES UPPERCASE FIELD NAMES.

         Convert them here.
         ----------------------------------------------------- */

      const normalized = {

        Decision:
          json.decision,

        Decision_Text:
          json.decision_text,


        Top_Class:
          json.top_class,

        Defect:
          json.defect,


        YOLO_Confidence:
          json.yolo_confidence,

        P_Defect:
          json.p_defect,


        Margin_Top1_Top2:
          json.margin,

        Entropy:
          json.entropy,

        Type_Confidence:
          json.type_confidence,

        Novel_Flag:
          json.novel_flag,


        decision_reason:
          json.decision_reason,


        Likely_Zone:
          json.likely_zone,

        Second_Zone:
          json.second_zone,

        Zone_Confidence:
          json.zone_confidence,

        Zone_Status:
          json.zone_status,


        Reason:
          json.zone_reason,

        Evidence:
          json.evidence,


        Run_Bottleneck_Zone:
          json.run_bottleneck_zone,

        Run_Throughput_Loss_Pct:
          json.run_throughput_loss_pct,

        Run_WIP_Stored:
          json.run_wip_stored,

        Run_Profit:
          json.run_profit,

        Run_Margin_Pct:
          json.run_margin_pct,

        Predicted_Margin_Pct:
          json.predicted_margin_pct,


        Localization_Supported:
          json.localization_supported,

        Localization_Box:
          json.localization_box,

        Localization_Area:
          json.localization_area,


        Recommendations:
          json.recommendations || [],


        raw:
          json.raw
      };


      console.log(
        "Inspection result:",
        normalized
      );


      setInspection(
        normalized
      );

      setBackendOnline(
        true
      );

      setUsingDemo(
        false
      );

      setTab(
        "inspection"
      );


    }  catch (e) {
  console.error("Inspection API error:", e);

  setError(
    `Inspection failed: ${e.message || "Unknown error"}`
  );

  setBackendOnline(true);
}


     finally {

      setLoading(
        false
      );

    }
  }


  /* =======================================================
     BATCH INSPECTION
     ======================================================= */

  async function runBatch() {

    setLoading(true);

    setError("");


    try {

      const response =
        await api(
          "/api/batch?n=20",
          {
            method: "POST"
          }
        );


      if (!response.ok) {

        const errorText =
          await response.text();

        throw new Error(
          errorText
        );
      }


      const json =
        await response.json();


      /*
        Backend returns:

            results: [...]

        Older frontend expected:

            rows: [...]

        Support both.
      */

      const rows =
        json.results ||
        json.rows ||
        [];


      setBatch(
        rows
      );


      setBackendOnline(
        true
      );

      setUsingDemo(
        false
      );


    }  catch (e) {

  console.error("Inspection API error:", e);

  setError(
    `Inspection failed: ${e.message || "Unknown error"}`
  );

  // Keep the backend marked as online if the server responded.
  setBackendOnline(true);

    } finally {

      setLoading(
        false
      );

    }
  }


  /* =======================================================
     BATCH STATISTICS
     ======================================================= */

  const batchStats =
    useMemo(() => {

      const total =
        batch.length;


      const rejected =
        batch.filter(
          x =>
            [
              "REJECT",
              "REJECT_NOVEL"
            ].includes(
              x.Decision
            )
        ).length;


      const review =
        batch.filter(
          x =>
            x.Decision ===
            "REVIEW"
        ).length;


      const passed =
        batch.filter(
          x =>
            x.Decision ===
            "PASS"
        ).length;


      return {
        total,
        rejected,
        review,
        passed
      };

    }, [batch]);


  /* =======================================================
     RENDER
     ======================================================= */

  return (

    <div className="app">


      {/* ===================================================
          SIDEBAR
          =================================================== */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brandMark">
            S
          </div>

          <div>

            <b>
              SentinelAI
            </b>

            <small>
              INDUSTRIAL INTELLIGENCE
            </small>

          </div>

        </div>


        <div className="nav">

          <NavButton
            icon={Activity}
            text="Overview"
            active={
              tab === "overview"
            }
            onClick={() =>
              setTab("overview")
            }
          />


          <NavButton
            icon={Search}
            text="Visual Inspection"
            active={
              tab === "inspection"
            }
            onClick={() =>
              setTab("inspection")
            }
          />


          <NavButton
            icon={ShieldCheck}
            text="Root Cause"
            active={
              tab === "root"
            }
            onClick={() =>
              setTab("root")
            }
          />


          <NavButton
            icon={Factory}
            text="Production"
            active={
              tab === "production"
            }
            onClick={() =>
              setTab("production")
            }
          />


          <NavButton
            icon={BarChart3}
            text="Batch Results"
            active={
              tab === "batch"
            }
            onClick={() =>
              setTab("batch")
            }
          />

        </div>


        <div
          className={`connection ${
            backendOnline
              ? "online"
              : "offline"
          }`}
        >

          <span />

          {backendOnline
            ? "AI ENGINE ONLINE"
            : "DEMO MODE"}

        </div>

      </aside>


      {/* ===================================================
          MAIN
          =================================================== */}

      <main>


        {/* HEADER */}

        <header>

          <div>

            <span className="eyebrow">
              AI QUALITY CONTROL
            </span>

            <h1>

              {tab === "overview" &&
                "Operations Overview"}

              {tab === "inspection" &&
                "Visual Inspection"}

              {tab === "root" &&
                "Root-Cause Analysis"}

              {tab === "production" &&
                "Production Analytics"}

              {tab === "batch" &&
                "Batch Inspection"}

            </h1>

          </div>


          <div className="headerActions">

            {usingDemo && (
              <span className="demoBadge">
                DEMO DATA
              </span>
            )}


            <button
              className="refresh"
              onClick={
                loadDashboard
              }
            >

              <RefreshCw
                size={15}
              />

              Refresh

            </button>

          </div>

        </header>


        {/* ERROR */}

        {error && (

          <div className="error">

            <AlertTriangle
              size={17}
            />

            {error}

          </div>

        )}


        {/* =================================================
            OVERVIEW
            ================================================= */}

        {tab === "overview" && (

          <section>

            <div className="hero">

              <div>

                <span className="eyebrow light">
                  SENTINELAI MONITORING
                </span>

                <h2>
                  See defects. Find causes. Improve flow.
                </h2>

                <p>
                  YOLO visual inspection combined
                  with production-zone stress and
                  bottleneck analysis.
                </p>

              </div>


              <label className="uploadBtn">

                <Upload
                  size={17}
                />

                Inspect Product

                <input
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={e =>
                    inspectProduct(
                      e.target.files?.[0]
                    )
                  }
                />

              </label>

            </div>


            <div className="metrics">

              <Metric
                label="Primary bottleneck"
                value={
                  data.primary_bottleneck ||
                  "—"
                }
                icon={Factory}
              />


              <Metric
                label="Avg throughput loss"
                value={`${
                  Number(
                    data.average_throughput_loss_pct ??
                    data.average_loss_pct ??
                    0
                  ).toFixed(2)
                }%`}
                icon={Gauge}
              />


              <Metric
                label="Average WIP"
                value={
                  Number(
                    data.average_wip ||
                    0
                  ).toFixed(0)
                }
                icon={Boxes}
              />


              <Metric
                label="Average output"
                value={
                  Number(
                    data.average_output ||
                    0
                  ).toFixed(0)
                }
                icon={CheckCircle2}
              />

            </div>


            <div className="grid2">


              {/* ZONE UTILIZATION */}

              <Panel
                title="Zone utilization"
                icon={Gauge}
              >

                {zones.map(
                  z => (

                    <ProgressBar
                      key={z.Zone}
                      label={z.Zone}
                      value={
                        z.Average_Utilization_Percent
                      }
                    />

                  )
                )}

              </Panel>


              {/* BOTTLENECK */}

              <Panel
                title="Bottleneck ranking"
                icon={Factory}
              >

                {(
                  data.bottleneck_ranking ||
                  []
                ).map(
                  (z, i) => (

                    <div
                      className="rank"
                      key={z.Zone}
                    >

                      <b>
                        {String(
                          i + 1
                        ).padStart(
                          2,
                          "0"
                        )}
                      </b>

                      <span>
                        {z.Zone}
                      </span>

                      <strong>
                        {Number(
                          z.Bottleneck_Percentage ||
                          0
                        ).toFixed(1)}
                        %
                      </strong>

                    </div>

                  )
                )}

              </Panel>

            </div>

          </section>

        )}


        {/* =================================================
            VISUAL INSPECTION
            ================================================= */}

        {tab === "inspection" && (

          <section>

            <div className="sectionHead">

              <div>

                <h2>
                  Product inspection
                </h2>

                <p>
                  Upload an image and send it
                  through the existing YOLO pipeline.
                </p>

              </div>


              <label className="uploadBtn">

                <Upload
                  size={16}
                />

                Choose image

                <input
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={e =>
                    inspectProduct(
                      e.target.files?.[0]
                    )
                  }
                />

              </label>

            </div>


            <div className="inspectionGrid">


              {/* IMAGE */}

              <div className="imageBox">

                {preview ? (

                  <img
                    src={preview}
                    alt="Selected product"
                  />

                ) : (

                  <div className="emptyImage">

                    <ImageIcon
                      size={44}
                    />

                    <b>
                      Drop or select a product image
                    </b>

                    <span>
                      JPG, PNG, JPEG, WEBP
                    </span>

                  </div>

                )}

              </div>


              {/* RESULTS */}

              <div className="resultCard">

                {loading ? (

                  <div className="loading">

                    <RefreshCw
                      className="spin"
                    />

                    Running inspection…

                  </div>

                ) : inspection ? (

                  <>

                    <div className="resultTop">

                      <span>
                        INSPECTION DECISION
                      </span>

                      <Decision
                        value={
                          inspection.Decision
                        }
                      />

                    </div>


                    <div className="bigClass">

                      {(
                        inspection.Top_Class ||
                        "unknown"
                      ).toUpperCase()}

                    </div>


                    {/* CONFIDENCE */}

                    <div className="confidenceBar">

                      <div>

                        <span>
                          YOLO confidence
                        </span>

                        <b>
                          {(
                            Number(
                              inspection.YOLO_Confidence ||
                              0
                            ) * 100
                          ).toFixed(1)}
                          %
                        </b>

                      </div>


                      <div className="bar">

                        <i
                          style={{
                            width: `${
                              Number(
                                inspection.YOLO_Confidence ||
                                0
                              ) * 100
                            }%`
                          }}
                        />

                      </div>

                    </div>


                    {/* DATA */}

                    <DataRow
                      label="Defect probability"
                      value={`${
                        (
                          Number(
                            inspection.P_Defect ||
                            0
                          ) * 100
                        ).toFixed(1)
                      }%`}
                    />


                    <DataRow
                      label="Defect type"
                      value={
                        inspection.Defect ||
                        "normal"
                      }
                    />


                    <DataRow
                      label="Likely zone"
                      value={
                        inspection.Likely_Zone ||
                        "N/A"
                      }
                    />


                    <DataRow
                      label="Zone confidence"
                      value={
                        inspection.Zone_Confidence ||
                        "N/A"
                      }
                    />


                    <DataRow
                      label="Zone status"
                      value={
                        inspection.Zone_Status ||
                        "N/A"
                      }
                    />


                    <DataRow
                      label="Bottleneck"
                      value={
                        inspection.Run_Bottleneck_Zone ||
                        "N/A"
                      }
                    />


                    <DataRow
                      label="Throughput loss"
                      value={
                        inspection.Run_Throughput_Loss_Pct != null
                          ? `${Number(
                              inspection.Run_Throughput_Loss_Pct
                            ).toFixed(2)}%`
                          : "N/A"
                      }
                    />


                    {/* ANALYSIS */}

                    <div className="reason">

                      <b>
                        Analysis
                      </b>

                      <p>
                        {
                          inspection.Reason ||
                          inspection.Evidence ||
                          inspection.decision_reason ||
                          "No explanation returned."
                        }
                      </p>

                    </div>


                    {/* EVIDENCE */}

                    {inspection.Evidence &&
                      inspection.Evidence !==
                        inspection.Reason && (

                        <div className="reason">

                          <b>
                            Evidence
                          </b>

                          <p>
                            {
                              inspection.Evidence
                            }
                          </p>

                        </div>

                      )}


                    {/* RECOMMENDATIONS */}

                    {inspection.Recommendations &&
                      inspection.Recommendations.length > 0 && (

                        <div className="recommendations">

                          <b>
                            Recommendations
                          </b>

                          <ul>

                            {inspection.Recommendations.map(
                              (rec, index) => (

                                <li
                                  key={index}
                                >
                                  {rec}
                                </li>

                              )
                            )}

                          </ul>

                        </div>

                      )}

                  </>

                ) : (

                  <div className="emptyResult">

                    <ShieldCheck
                      size={35}
                    />

                    <h3>
                      Ready for inspection
                    </h3>

                    <p>
                      Upload a product image
                      to see the AI decision.
                    </p>

                  </div>

                )}

              </div>

            </div>

          </section>

        )}


        {/* =================================================
            ROOT CAUSE
            ================================================= */}

        {tab === "root" && (

          <section>

            <div className="sectionHead">

              <div>

                <h2>
                  Root-cause zones
                </h2>

                <p>
                  Historical utilization and
                  waiting-time stress used by
                  the backend.
                </p>

              </div>

            </div>


            <div className="zoneGrid">


              {[
                "Drilling",
                "Milling",
                "Assembly"
              ].map(
                zone => {

                  const item =
                    zones.find(
                      x =>
                        x.Zone === zone
                    );


                  return (

                    <div
                      className="zoneCard"
                      key={zone}
                    >

                      <span className="zoneTag">
                        {zone}
                      </span>


                      <h3>

                        {item
                          ? Number(
                              item.Average_Utilization_Percent
                            ).toFixed(1)
                          : "—"}
                        %

                      </h3>


                      <p>
                        average utilization
                      </p>


                      <ProgressBar
                        label="Utilization"
                        value={
                          item?.Average_Utilization_Percent ||
                          0
                        }
                      />


                      <div className="miniStats">

                        <span>
                          Waiting
                        </span>

                        <b>
                          {item
                            ? Number(
                                item.Average_Waiting_Time
                              ).toFixed(2)
                            : "—"}
                        </b>

                      </div>


                      <div className="miniStats">

                        <span>
                          Runs ≥90%
                        </span>

                        <b>
                          {
                            item?.Runs_Above_90_Percent ??
                            "—"
                          }
                        </b>

                      </div>

                    </div>

                  );

                }
              )}


              {/* STORAGE */}

              <div className="zoneCard storageCard">

                <span className="zoneTag">
                  STORAGE
                </span>

                <h3>
                  WIP / dwell
                </h3>

                <p>
                  Included in defect attribution
                </p>

                <div className="storageVisual">

                  <Clock3
                    size={25}
                  />

                  <span>
                    Storage stress
                  </span>

                </div>

                <small>
                  Storage time and stored-part
                  deviations are used by the
                  existing root-cause logic.
                </small>

              </div>

            </div>


            {/* ROOT CAUSE FLOW */}

            <div className="rootDiagram">

              <div className="diagramTitle">
                DEFECT → PRODUCTION STRESS
              </div>


              <div className="diagram">

                <div className="diagramNode defect">
                  DEFECT
                </div>

                <div className="arrow">
                  →
                </div>

                <div className="diagramNode">
                  DEFECT PRIOR
                </div>

                <div className="arrow">
                  ×
                </div>

                <div className="diagramNode">
                  ZONE STRESS
                </div>

                <div className="arrow">
                  →
                </div>

                <div className="diagramNode result">
                  LIKELY ZONE
                </div>

              </div>

            </div>

          </section>

        )}


        {/* =================================================
            PRODUCTION
            ================================================= */}

        {tab === "production" && (

          <section>

            <div className="metrics">

              <Metric
                label="Input parts"
                value={
                  Number(
                    data.average_input ||
                    0
                  ).toFixed(0)
                }
                icon={Factory}
              />


              <Metric
                label="Output parts"
                value={
                  Number(
                    data.average_output ||
                    0
                  ).toFixed(0)
                }
                icon={CheckCircle2}
              />


              <Metric
                label="Throughput gap"
                value={
                  Number(
                    data.average_throughput_gap ||
                    0
                  ).toFixed(1)
                }
                icon={AlertTriangle}
              />


              <Metric
                label="WIP stored"
                value={
                  Number(
                    data.average_wip ||
                    0
                  ).toFixed(0)
                }
                icon={Boxes}
              />

            </div>


            <Panel
              title="Production zones"
              icon={Factory}
            >

              {zones.map(
                z => (

                  <div
                    className="prodRow"
                    key={z.Zone}
                  >

                    <b>
                      {z.Zone}
                    </b>

                    <span>
                      {Number(
                        z.Average_Utilization_Percent
                      ).toFixed(1)}
                      % utilization
                    </span>

                    <span>
                      {Number(
                        z.Average_Waiting_Time
                      ).toFixed(2)}
                      waiting
                    </span>

                    <span>
                      {
                        z.Runs_Above_90_Percent
                      } high-util runs
                    </span>

                  </div>

                )
              )}

            </Panel>


            <div className="productionCards">

              <div className="smallCard">

                <span>
                  Primary bottleneck
                </span>

                <b>
                  {
                    data.primary_bottleneck ||
                    "—"
                  }
                </b>

              </div>


              <div className="smallCard">

                <span>
                  Secondary constraint
                </span>

                <b>
                  {
                    data.secondary_constraint ||
                    "—"
                  }
                </b>

              </div>

            </div>

          </section>

        )}


        {/* =================================================
            BATCH RESULTS
            ================================================= */}

        {tab === "batch" && (

          <section>

            <div className="sectionHead">

              <div>

                <h2>
                  Batch inspection
                </h2>

                <p>
                  Run the backend batch pipeline
                  and review the returned records.
                </p>

              </div>


              <button
                className="uploadBtn"
                onClick={runBatch}
                disabled={loading}
              >

                <RefreshCw
                  size={16}
                  className={
                    loading
                      ? "spin"
                      : ""
                  }
                />

                Run batch

              </button>

            </div>


            {batch.length > 0 && (

              <div className="metrics">

                <Metric
                  label="Inspected"
                  value={
                    batchStats.total
                  }
                  icon={Activity}
                />


                <Metric
                  label="Passed"
                  value={
                    batchStats.passed
                  }
                  icon={CheckCircle2}
                />


                <Metric
                  label="Review"
                  value={
                    batchStats.review
                  }
                  icon={AlertTriangle}
                />


                <Metric
                  label="Rejected"
                  value={
                    batchStats.rejected
                  }
                  icon={XCircle}
                />

              </div>

            )}


            <Panel
              title="Inspection results"
              icon={BarChart3}
            >

              {batch.length === 0 ? (

                <div className="emptyTable">
                  Run a batch to populate this table.
                </div>

              ) : (

                <div className="tableWrap">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          Product
                        </th>

                        <th>
                          Class
                        </th>

                        <th>
                          Confidence
                        </th>

                        <th>
                          Decision
                        </th>

                        <th>
                          Likely zone
                        </th>

                        <th>
                          Zone confidence
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {batch.map(
                        (r, i) => (

                          <tr key={i}>

                            <td>
                              {
                                r.Product_Index ??
                                i
                              }
                            </td>


                            <td>
                              {
                                r.Top_Class ||
                                "—"
                              }
                            </td>


                            <td>

                              {
                                r.YOLO_Confidence != null

                                  ? `${(
                                      Number(
                                        r.YOLO_Confidence
                                      ) * 100
                                    ).toFixed(1)}%`

                                  : "—"
                              }

                            </td>


                            <td>

                              <Decision
                                value={
                                  r.Decision
                                }
                              />

                            </td>


                            <td>

                              {
                                r.Likely_Zone ||
                                "N/A"
                              }

                            </td>


                            <td>

                              {
                                r.Zone_Confidence ||
                                "N/A"
                              }

                            </td>

                          </tr>

                        )
                      )}

                    </tbody>

                  </table>

                </div>

              )}

            </Panel>

          </section>

        )}

      </main>

    </div>
  );
}


/* =========================================================
   NAVIGATION BUTTON
   ========================================================= */

function NavButton({
  icon: Icon,
  text,
  active,
  onClick
}) {

  return (

    <button
      className={`navButton ${
        active
          ? "active"
          : ""
      }`}
      onClick={onClick}
    >

      <Icon
        size={17}
      />

      {text}

    </button>

  );
}


/* =========================================================
   START REACT
   ========================================================= */

createRoot(
  document.getElementById(
    "root"
  )
).render(

  <React.StrictMode>

    <App />

  </React.StrictMode>

);