import time
from tensorboardX import SummaryWriter

class MetricsWriter:
    def __init__(self, run_name, runs_dir='runs/', score_dir='score/'):
        score_output_file = os.path.join(score_dir, run_name)
        # All metrics are written to both TensorBoard and a CSV file
        self.tb_writer = SummaryWriter(score_output_file)
        # add directory creation
        runs_output_file = os.path.join(runs_dir, run_name)
        self.file_writer = open(output_file, "w+")
        self.file_writer.write("episode,timestamp,metric_name,value\n")

    def write_metric(self, episode, metric_name, value):
        timestamp = time.time()
        self.file_writer.write("{},{},{},{}\n".format(episode, timestamp, metric_name, value))
        self.tb_writer.add_scalar(metric_name, value, episode)
