# metrics.py
import csv
import os
from config import METRICS_CSV

class MetricsLogger:
    def __init__(self, filename=METRICS_CSV):
        self.filename = filename
        # create header if not exists
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "hazard_id",
                    "building_name",
                    "t_start",
                    "t_detect",
                    "t_consensus",
                    "t_aggregated",
                    "detect_delay",
                    "consensus_delay",
                    "aggregate_delay"
                ])

    def log_hazard(self, hazard_mgr, buildings):
        if hazard_mgr.start_time is None:
            return
        hid = hazard_mgr.id_counter
        bname = buildings[hazard_mgr.building_index]["name"] if hazard_mgr.building_index is not None else "UNKNOWN"

        t0 = hazard_mgr.start_time
        td = hazard_mgr.detect_time
        tc = hazard_mgr.consensus_time
        ta = hazard_mgr.aggregated_time

        def diff(t):
            return None if t is None else (t - t0)

        row = [
            hid,
            bname,
            t0,
            td,
            tc,
            ta,
            diff(td),
            diff(tc),
            diff(ta),
        ]
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[METRICS] Logged hazard {hid} to {self.filename}")
