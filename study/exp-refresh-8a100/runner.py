import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, RunConfig, ConfigList, RandomDataset, percent_gen

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  # .override('root_path', ['/disk1/graph-learning-copy/samgraph/'])
  .override('epoch', [30])
  .override('gpu_num', [8])
  .override('logdir', ['run-logs'])
  .override('confdir', ['run-configs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
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
  .override('global_batch_size', [65536])
  .override('plain_dense_model', [True])
  .override('mock_embedding', [True])
  .override('coll_cache_enable_refresh', [True])
  .override('coll_cache_refresh_iter', [10000])
  .override('coll_cache_refresh_seq_bucket_sz', [1000, 2000, 4000, 8000])
  .override('log_level', ["info"])
)

cfg_list_collector.concat(cur_common_base.copy()
  .hyper_override(
    ["random_request", "alpha", "dataset", "custom_env"],
    [
      [False,  None,  Dataset.criteo_tb, "SAMGRAPH_EMPTY_FEAT=24"],
    ])
  .override('cache_percent', 
    [0.01] + 
    # [0.02, 0.04] + 
    # [0.08, 0.12] + 
    # [0.10, 0.15] + 
    []
  ).hyper_override(
  ['coll_cache_policy', "coll_cache_no_group", "coll_cache_concurrent_link"], 
  [
    # [CachePolicy.clique_part, "DIRECT", ""],
    # [CachePolicy.clique_part, "", "MPSPhase"],
    # [CachePolicy.rep_cache, "DIRECT", ""],
    # [CachePolicy.rep_cache, "", "MPSPhase"],
    [CachePolicy.coll_cache_asymm_link, "", "MPSPhase"],
  ]))


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