import os, sys
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector
# from paper_runner import cfg_list_collector

# selected_col = ['system', 'dataset_short', 'global_batch_size',]
# selected_col += ['cache_policy_short', 'cache_percentage']
# selected_col += ['mock_embedding', 'max_vocabulary_size']

selected_col = ['short_app']
selected_col += ['policy_impl', 'cache_percentage', 'global_batch_size']
# selected_col += ['unsupervised']
selected_col += ['dataset_short', 'coll_cache:solve_time']

selected_col += ['Step(average) L1 sample']
selected_col += ['Step(average) L1 recv']
selected_col += ['Step(average) L2 feat copy']
selected_col += ['Step(average) L1 train total']
selected_col += ['Time.L','Time.R','Time.C']
selected_col += ['Wght.L','Wght.R','Wght.C']
selected_col += ['optimal_local_rate','optimal_remote_rate','optimal_cpu_rate']
selected_col += ['Thpt.L','Thpt.R','Thpt.C']
selected_col += ['SizeGB.L','SizeGB.R','SizeGB.C']
selected_col += ['coll_cache:local_cache_rate']
selected_col += ['coll_cache:remote_cache_rate']
selected_col += ['coll_cache:global_cache_rate']
selected_col += ['train_process_time', 'epoch_time:train_total', 'epoch_time:copy_time']
selected_col += ['coll_cache:z']
selected_col += ['coll_cache_scale']


cfg_list_collector = (cfg_list_collector.copy()
  # .select('dataset', [Dataset.twitter, Dataset.uk_2006_05])
  # .select('cache_policy', [CachePolicy.coll_cache_10])
#   .select('pipeline', [False])
  # .override_T('logdir', [
  #   # 'run-logs-backup-pcvyatta',
  #   # 'run-logs-backup',
  #   'run-logs',
  # ])
)


if __name__ == '__main__':
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(bench_list, f, selected_col)