import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, RunConfig, ConfigList, RandomDataset, percent_gen

do_mock = False
durable_log = True
fail_only = False

cur_common_base = (ConfigList()
  # .override('root_path', ['/disk1/graph-learning-copy/samgraph/'])
  .override('epoch', [2])
  .override('gpu_num', [4])
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  .override('empty_feat', [23])
  .override('coll_cache_scale', [16])
  .override('model', [
    Model.dlrm, 
    # Model.dcn,
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

cfg_list_collector.concat(cur_common_base.copy().hyper_override(
  ["random_request", "alpha", "dataset", "custom_env"],
  [
    # [True,  0.2,  RandomDataset("simple_power0.2_slot100_C20m", "SP_02_S100_C20m", 20000000, 100), ""],
    [True,  0.2,  RandomDataset("simple_power0.2_slot100_C800m", "SP_02_S100_C800m", 800000000, 100), ""],
    [False,  None,  Dataset.criteo_tb, ""],
  ]
  ).override('cache_percent', 
    # percent_gen(1,4,1) + 
    # percent_gen(6,8,2) + 
    # percent_gen(10, 30, 5) + 
    [0.01]
  ))
  # .override('cache_percent', [0.01] + percent_gen(10, 90, 5) + percent_gen(95, 100, 1)))

cfg_list_collector.hyper_override(
  ['coll_cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link", "sok_use_hashtable"], 
  [
    [CachePolicy.clique_part, "DIRECT", "", None],
    # [CachePolicy.rep_cache, "DIRECT", "", None],
    # [CachePolicy.clique_part, "", "MPS", None],
    # [CachePolicy.rep_cache, "", "MPS", None],
    # [CachePolicy.coll_cache_asymm_link, "", "MPS", None],
    [CachePolicy.coll_cache_asymm_link, "", "MPSPhase", None],
    # [CachePolicy.sok, "", "", False],
    [CachePolicy.sok, "", "", True],
    # [CachePolicy.hps, "", "", None],
  ])

# selector for fast validation
(cfg_list_collector
  # .select('cache_percent', [
  #   # 0.01,
  # ])
  # .select('coll_cache_policy', [
  #   CachePolicy.coll_cache_asymm_link,
  #   # CachePolicy.clique_part,
  #   # CachePolicy.rep_cache,
  # ])
  # .select('coll_cache_no_group', [
  #   # 'DIRECT',
  #   '',
  # ])
  # .select('model', [
  #   Model.dlrm,
  #   # Model.dcn,
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
    elif arg == '-f' or arg == '--fail':
      fail_only = True
  cfg_list_collector.run(do_mock, durable_log)