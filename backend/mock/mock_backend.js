

// server.js
const express = require("express");
const cors = require("cors");
const app = express();
app.use(express.json());
app.use(cors());

const jobs = new Map();

app.post("/jobs", (req, res) => {
  const id = Math.random().toString(36).slice(2);
  // immediately mark as processing
  jobs.set(id, { status: "processing", started: Date.now() });
  console.log(`Job ${id} started`);
  res.json({ jobId: id });
});

app.get("/jobs/:id", (req, res) => {
  const { id } = req.params;              // <-- define id
  const job = jobs.get(id);
  if (!job) return res.status(404).send("No such job");

  // after 4 seconds, flip to done
  if (Date.now() - job.started > 4000 && job.status !== "done") {
    job.status = "done";
    job.result = [
      { t0: 0.0, t1: 2.5, box: { x: 12, y: 34, w: 80, h: 60 } },
      { t0: 3.1, t1: 7.0, box: { x: 50, y: 20, w: 90, h: 50 } },
    ];
    console.log(`Job ${id} done`);        // <-- now defined
  }

  res.json({ status: job.status, ...(job.status === "done" ? { result: job.result } : {}) });
});


app.listen(4000, () => console.log("API running on :4000"));
