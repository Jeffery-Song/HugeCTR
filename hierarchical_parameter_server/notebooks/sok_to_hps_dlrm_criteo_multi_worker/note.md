# performance notes on dataset loading
(step time is measure by 6000 iter, 8192 per gpu batch size, 10% cache, coll cache policy)

## multi processing
- use `distibute_dataset_from_function` directly
  - 5ms data iter time, 3.5ms step time
- use local shard dataset, without building, no distributed dataset
  - 0.3ms data iter time, 3ms step time
- use `distibute_dataset_from_function`, then call a dump tf.function and store returned per_replica batch to python list
  - the fastest: 0 data cost, 2ms step time
  - but entire dataset is loaded to dataset, which leads to 4GB cost 
- use local shard dataset, then call a dump tf.function and store returned per_replica batch to python list
  - the fastest: 0 data cost, 2.2ms step time
  - dataset stays at cpu. this is the most ideal case!