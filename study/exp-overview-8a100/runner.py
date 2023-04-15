import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, RunConfig, ConfigList, RandomDataset, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  # .override('root_path', ['/disk1/graph-learning-copy/samgraph/'])
  .override('epoch', [3])
  .override('gpu_num', [8])
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  .override('coll_cache_scale', [16])
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
  .override('global_batch_size', [65536])
  .override('plain_dense_model', [True])
  .override('mock_embedding', [True])
)

cfg_list_collector.concat(cur_common_base.copy()
  .hyper_override(
    ["random_request", "alpha", "dataset", "custom_env"],
    [
      [False,  None,  Dataset.criteo_tb, "SAMGRAPH_EMPTY_FEAT=24"],
    ])
  .override('cache_percent', 
    # [0.01] + 
    # [0.02, 0.04] + 
    # [0.08, 0.12] + 
    # [0.10, 0.15] + 
    [0.12] +
    []
  ).hyper_override(
  ['coll_cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [CachePolicy.clique_part, "DIRECT", ""],
    [CachePolicy.clique_part, "", "MPSPhase"],
    [CachePolicy.rep_cache, "DIRECT", ""],
    [CachePolicy.rep_cache, "", "MPSPhase"],
    [CachePolicy.coll_cache_asymm_link, "DIRECT", ""],
    [CachePolicy.coll_cache_asymm_link, "", "MPSPhase"],
    [CachePolicy.sok, "", ""],
    [CachePolicy.hps, "", ""],
  ]))


cfg_list_collector.concat(cur_common_base.copy()
  .hyper_override(
    ["random_request", "alpha", "dataset", "custom_env"],
    [
      [True,  0.2,  RandomDataset("simple_power0.2_slot100_C800m", "SP_02_S100_C800m", 800000000, 100), "SAMGRAPH_EMPTY_FEAT=24"],
    ])
  .override('cache_percent', 
    # [0.01] + 
    # [0.02, 0.04] + 
    [0.08, 0.12] + 
    # [0.10, 0.15] + 
    []
  ).hyper_override(
  ['coll_cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    [CachePolicy.clique_part, "DIRECT", ""],
    [CachePolicy.clique_part, "", "MPSPhase"],
    [CachePolicy.rep_cache, "DIRECT", ""],
    [CachePolicy.rep_cache, "", "MPSPhase"],
    [CachePolicy.coll_cache_asymm_link, "DIRECT", ""],
    [CachePolicy.coll_cache_asymm_link, "", "MPSPhase"],
    [CachePolicy.sok, "", ""],
    [CachePolicy.hps, "", ""],
  ]))

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
  retry = False
  fail_only = False
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
    elif arg == '-r' or arg == '--retry':
      retry = True
    elif arg == '-f' or arg == '--fail':
      fail_only = True
  cfg_list_collector.run(do_mock, durable_log, retry=retry, fail_only=fail_only)