import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from runner_helper import System, Model, Dataset, CachePolicy, RunConfig, ConfigList

do_mock = False
durable_log = True

cur_common_base = (ConfigList()
  # .override('root_path', ['/disk1/graph-learning-copy/samgraph/'])
  .override('epoch', [5])
  .override('num_gpu', [8])
  .override('logdir', ['run-logs'])
  .override('profile_level', [3])
  .override('multi_gpu', [True])
  )

cfg_list_collector = ConfigList.Empty()

'''
HPS
'''
cur_common_base = (cur_common_base.copy().override('system', [System.hps]))
cur_common_base = (cur_common_base.copy().override('global_batch_size', [65536]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.criteo_like_uniform,        ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.criteo_like_uniform_small,  ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.simple_power02,             ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.simple_power02_slot100,     ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.simple_power1,              ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.simple_uniform,             ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.criteo_tb,                  ]))

'''
Coll Cache
'''
cur_common_base = (cur_common_base.copy().override('system', [System.collcache]))
cur_common_base = (cur_common_base.copy().override('global_batch_size', [65536]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.criteo_like_uniform,        ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.criteo_like_uniform_small,  ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.simple_power02,             ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.simple_power02_slot100,     ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.simple_power1,              ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.simple_uniform,             ]))
cfg_list_collector.concat(cur_common_base.copy().override('dataset', [Dataset.criteo_tb,                  ]))

cfg_list_collector.override_T('cache_policy', [
  CachePolicy.clique_part,
  CachePolicy.rep_cache,
  CachePolicy.coll_cache_asymm_link,
  # CachePolicy.clique_part_by_degree_2,
  ])

if __name__ == '__main__':
  from sys import argv
  for arg in argv[1:]:
    if arg == '-m' or arg == '--mock':
      do_mock = True
    elif arg == '-i' or arg == '--interactive':
      durable_log = False
  cfg_list_collector.run(do_mock, durable_log)