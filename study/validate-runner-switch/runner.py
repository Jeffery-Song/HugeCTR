import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, RunConfig, ConfigList, RandomDataset, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  .override('epoch', [3])
  .override('num_gpu', [8])
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  )

cfg_list_collector = ConfigList.Empty()

'''
Coll Cache
'''
cur_common_base = (cur_common_base.copy().override('system', [System.collcache]))
cur_common_base = (cur_common_base.copy()
  .override('global_batch_size', [65536])
  .override('cache_percent', percent_gen(10, 10, 2))
)

'''
On disk sparse/dense model, on disk dataset
'''
cfg_list_collector.concat(
cur_common_base.copy().hyper_override(
    ["model_root_path",                                  "dataset"],
  [
    ["/nvme/songxiaoniu/hps-model/dlrm_criteo/",         Dataset.criteo_like_uniform],
    ["/nvme/songxiaoniu/hps-model/dlrm_simple/",         Dataset.simple_power1],
    ["/nvme/songxiaoniu/hps-model/dlrm_simple_slot100/", Dataset.simple_power1_slot100],
  ]
)
)

'''
Plain sparse/dense model, on disk or random dataset
max vocabulary size, slot num are inferred from dataset
'''
cfg_list_collector.concat(cur_common_base.copy().hyper_override(
    ["random_request", "alpha", "dataset"],
  [
    [True,              0.1,    RandomDataset("simple_power0.1_slot100", "SP_01_S100", 100000000, 100)],
    [True,              0.1,    RandomDataset("simple_power0.1_slot50", "SP_01_S50", 100000000, 50)],
    [True,              0.1,    RandomDataset("simple_power0.1_slot25", "SP_01_S25", 100000000, 25)],
    [False,             None,   Dataset.simple_power1_slot100],
    [False,             None,   Dataset.criteo_like_uniform],
  ]
).override('plain_dense_model', [True]).override('mock_embedding', [True]))

cfg_list_collector.override_T('coll_cache_policy', [
  CachePolicy.clique_part,
  # CachePolicy.rep_cache,
  # CachePolicy.coll_cache_asymm_link,
  ])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
  cfg_list_collector.run(do_mock, durable_log)