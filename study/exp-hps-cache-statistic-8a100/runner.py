import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, RunConfig, ConfigList, RandomDataset, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  .override('empty_feat', [25])
  .override('scalability_test', [True])
  .override('hps_cache_statistic', [True])
  )

cfg_list_collector = ConfigList.Empty()

'''
Coll Cache
'''
cur_common_base = (cur_common_base.copy().override('system', [System.hps]))
cur_common_base = (cur_common_base.copy().override('plain_dense_model', [True]))
cur_common_base = (cur_common_base.copy().override('mock_embedding', [True]))

# Criteo Kaggle
cfg_list_collector.concat(cur_common_base.copy().override("dataset", [Dataset.criteo_kaggle])
                                                .override("cache_percent", [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.32, 0.64, 0.8, 0.99])
                                                .override('iteration_per_epoch', [320])
                                                .hyper_override(
                                                  ['gpu_num', 'global_batch_size', 'epoch'],
                                                  # [[8, 65536, 2]])
                                                  [[8, 65536, 2], [2, 16384, 8]])
                                                )

# Criteo TB
cfg_list_collector.concat(cur_common_base.copy().override("dataset", [Dataset.criteo_tb])
                                                .override("cache_percent", [0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.12])
                                                .override('iteration_per_epoch', [1000])
                                                .hyper_override(
                                                  ['gpu_num', 'global_batch_size', 'epoch'],
                                                  [[8, 65536, 3], [2, 16384, 11]])
                                                )

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
  cfg_list_collector.run(do_mock, durable_log)