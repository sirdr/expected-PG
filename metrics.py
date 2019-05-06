import time
import os
from tensorboardX import SummaryWriter

class MetricsWriter:
    def __init__(self, run_name):
        # All metrics are written to both TensorBoard and a CSV file
        self.tb_writer = SummaryWriter(f"score/{run_name}")
        # add directory creation
        outdir = f"runs/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.file_writer = open(os.path.join(outdir, f"{run_name}"), "w+")
        self.file_writer.write("episode,timestamp,metric_name,value\n")

    def write_metric(self, episode, metric_name, value):
        timestamp = time.time()
        self.file_writer.write("{},{},{},{}\n".format(episode, timestamp, metric_name, value))
