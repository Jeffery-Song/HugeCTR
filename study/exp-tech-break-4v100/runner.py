import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, RunConfig, ConfigList, RandomDataset, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  # .override('root_path', ['/disk1/graph-learning-copy/samgraph/'])
  .override('epoch', [2])
  .override('gpu_num', [4])
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  .override('model', [
    Model.dlrm, 
    Model.dcn,
  ])
  )

cfg_list_collector = ConfigList.Empty()

'''
Coll Cache
'''
(cur_common_base
  .override('system', [System.collcache])
  .override('global_batch_size', [32768])
  .override('plain_dense_model', [True])
  .override('mock_embedding', [True])
)

# cfg_list_collector.concat(cur_common_base.copy().hyper_override(
#   ["random_request", "alpha", "dataset", "custom_env"],
#   [
#     [False,  None,  Dataset.criteo_tb, "SAMGRAPH_EMPTY_FEAT=27"],
#   ]
#   ).override('cache_percent', [0.01]))

cfg_list_collector.concat(cur_common_base.copy().hyper_override(
  ["random_request", "alpha", "dataset", "custom_env"],
  [
    [True,  0.1,  RandomDataset("simple_power0.1_slot100", "SP_01_S100", 100000000, 100), ""],
    # [True,  0.1,  RandomDataset("simple_power0.1_slot50", "SP_01_S50", 100000000, 50), ""],
  ]
  ).override('cache_percent', 
    [0.01] + percent_gen(2, 8, 2) + 
    [0.10, 0.12, 0.14, 0.16, 0.18] +
    []
  ))

cfg_list_collector.hyper_override(
  ['coll_cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [CachePolicy.clique_part, "DIRECT", ""],
    [CachePolicy.clique_part, "", "MPS"],
    [CachePolicy.rep_cache, "DIRECT", ""],
    [CachePolicy.rep_cache, "", "MPS"],
    [CachePolicy.coll_cache_asymm_link, "", "MPS"],
    [CachePolicy.coll_cache_asymm_link, "", "MPSPhase"],
  ])

# selector for fast validation
(cfg_list_collector
  # .select('cache_percent', [
  #   # 0.01,
  # ])
  # .select('coll_cache_policy', [
  #   # CachePolicy.coll_cache_asymm_link,
  #   # CachePolicy.clique_part,
  #   # CachePolicy.rep_cache,
  # ])
  # .select('coll_cache_no_group', [
  #   # 'DIRECT',
  #   # '',
  # ])
  # .select('model', [Model.dlrm
  # ])
  # .select('dataset', [Dataset.criteo_tb
  # ])
  # .override('custom_env', ["SAMGRAPH_EMPTY_FEAT=10"])
  )

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
  cfg_list_collector.run(do_mock, durable_log)