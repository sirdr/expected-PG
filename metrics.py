import time
from tensorboardX import SummaryWriter

class MetricsWriter:
    def __init__(self, run_name):
        # All metrics are written to both TensorBoard and a CSV file
        self.tb_writer = SummaryWriter(f"score/{run_name}")
        # add directory creation
        self.file_writer = open(f"runs/{run_name}", "w+")
        self.file_writer.write("episode,timestamp,metric_name,value\n")

    def write_metric(self, episode, metric_name, value):
        timestamp = time.time()
        self.file_writer.write("{},{},{},{}\n".format(episode, timestamp, metric_name, value))
