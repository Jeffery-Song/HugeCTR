import os, sys, copy
sys.path.append(os.getcwd()+'/../common')
from common_parser import *
from runner import cfg_list_collector

# selected_col = ['system', 'dataset_short', 'global_batch_size',]
# selected_col += ['cache_policy_short', 'cache_percentage']
# selected_col += ['mock_embedding', 'max_vocabulary_size']

selected_col = ['short_app']
selected_col += ['policy_impl', 'cache_percentage', 'global_batch_size']
# selected_col += ['unsupervised']
selected_col += ['dataset_short']
selected_col += ['Sequence']
selected_col += ['Sequence(Average) extract time']
selected_col += ['Sequence(Average) e2e time']
selected_col += ['Sequence(Average) seq duration']

# selected_col += ['Step(average) L1 sample']
# selected_col += ['Step(average) L1 recv']
# selected_col += ['Step(average) L2 feat copy']
# selected_col += ['Step(average) L1 train total']

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

def div_nan(a,b):
  if b == 0:
    return math.nan
  return a/b

def max_nan(a,b):
  if math.isnan(a):
    return b
  elif math.isnan(b):
    return a
  else:
    return max(a,b)

def handle_nan(a, default=0):
  if math.isnan(a):
    return default
  return a
def zero_nan(a):
  return handle_nan(a, 0)

def short_app_name(inst: BenchInstance):
  # suffix = "_unsup" if inst.get_val('unsupervised') else "_sup"
  inst.vals['short_app'] = inst.cfg.model.name

def full_policy_name(inst: BenchInstance):
  inst.vals['policy_impl'] = inst.get_val('coll_cache_concurrent_link') + inst.get_val('cache_policy_short')

if __name__ == '__main__':
  bench_list = [BenchInstance().init_from_cfg(cfg) for cfg in cfg_list_collector.conf_list]
  seq_bench_list = []
  for inst in bench_list:
    inst : BenchInstance
    short_app_name(inst)
    full_policy_name(inst)
    try:
      # get bucket num
      profile_steps = inst.get_val('epoch') * inst.get_val('iteration_per_epoch') * inst.get_val('gpu_num')
      bucket_num = profile_steps / inst.get_val('coll_cache_refresh_seq_bucket_sz')

      # example: [Step(Seq_23) Profiler Level 3 E2 S7999]
      for i in range(int(bucket_num)):
        inst.vals['Sequence'] = i
        inst.vals['Sequence(Average) convert time'] = inst.vals[f'Step(Seq_{i}) L1 convert time']
        inst.vals['Sequence(Average) e2e time'] = inst.vals[f'Step(Seq_{i}) L1 train'] + inst.vals['Sequence(Average) convert time'] 
        inst.vals['Sequence(Average) extract time'] = inst.vals[f'Step(Seq_{i}) L2 cache feat copy']
        inst.vals['Sequence(Average) seq duration'] = inst.vals[f'Step(Seq_{i}) L1 seq duration']
        # when cache rate = 0, extract time has different log name...
        # inst.vals['Step(average) L2 feat copy'] = max_nan(inst.get_val('Step(average) L2 cache feat copy'), inst.get_val('Step(average) L2 extract'))
        seq_bench_list.append(copy.deepcopy(inst))
    except Exception as e:
      print(e)
      print("Error when " + inst.cfg.get_log_fname() + '.log')
  with open(f'data.dat', 'w') as f:
    BenchInstance.print_dat(seq_bench_list, f, selected_col)