outputfname = "intersect.svg"
dat_file='data.dat'
machine='A100'

# col numbers
col_system = 1
col_dataset = 2
col_batch_size = 3
col_cache_policy = 4
col_cache_percent = 5
col_mock = 6
col_vocab_size = 7
col_sample_time = 8
col_recv_time = 9
col_copy_time = 10
col_train_time = 11
col_local_time = 12
col_remote_time = 13
col_cpu_time = 14
col_local_weight = 15
col_remote_weight = 16
col_cpu_weight = 17
col_train_process_time=30
col_gpu_num=33
col_intersect_ratio=34
col_hit_ratio=35
col_hit_overlap_ratio=36
col_miss_overlap_ratio=37

# batch_size = "65536"
fig_col_num = 2
fig_row_num = 2

set datafile sep '\t'
set output outputfname

# set terminal svg "Helvetica,16" enhance color dl 2 background rgb "white"
set terminal svg size 500,400 font "Helvetica,16" enhanced background rgb "white" dl 2
set multiplot layout 2,2

set style fill solid border -2
set pointsize 1
set boxwidth 0.5 relative

set tics font ",12" scale 0.5

set rmargin 1.6
set lmargin 5.5
set tmargin 1.5
set bmargin 2

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if(". \
                              "$".col_dataset."      ~ \"%s$\"     && ". \
                              "$".col_batch_size."   ~ \"%s$\"     && ". \
                              "$".col_cache_policy." ~ \"%s$\"     ". \
                              ") { print }}' ".dat_file." "
cmd_filter_dat_by_policy(dataset, batch_size, policy)=sprintf(format_str, dataset, batch_size, policy)
##########################################################################################
step_plot_func(dataset)=sprintf( \
"dataset=\"%s\"; \
set title dataset.\" Access Hit Overlap Ratio\" offset 0,-1 ; \
plot cmd_filter_dat_by_policy(dataset, \"65536\", \"CollAsymm\") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_hit_overlap_ratio)*100) w lp ps 0.5 pt 4 lw 1 lc 1 title \"8 GPU\" \
    ,cmd_filter_dat_by_policy(dataset, \"16384\", \"CollAsymm\") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_hit_overlap_ratio)*100) w lp ps 0.5 pt 6 lw 1 lc 2 title \"2 GPU\"; \
set logscale x;\
set title dataset.\" Access Hit Overlap Ratio\" offset 0,-1 ; \
plot cmd_filter_dat_by_policy(dataset, \"65536\", \"CollAsymm\") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_hit_overlap_ratio)*100) w lp ps 0.5 pt 4 lw 1 lc 1 title \"8 GPU\" \
    ,cmd_filter_dat_by_policy(dataset, \"16384\", \"CollAsymm\") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_hit_overlap_ratio)*100) w lp ps 0.5 pt 6 lw 1 lc 2 title \"2 GPU\"; \
unset logscale x;\
    ", dataset)


### Key
set key inside left Left reverse top enhanced box 
set key samplen 1 spacing 1 height 0.2 width -2 font ',13' maxrows 4 center at graph 0.65, graph 0.6 noopaque

set xlabel "Cache Rate" offset 0,1.7 font ",13"
# set xrange [0:]
# set xtics rotate by -45
set xtics nomirror offset 0,0.7

## Y-axis
set ylabel "Overlap Ratio(%)" offset 3.5,0 font ",13"
set yrange [90:100]
set ytics 90,2,100 
set ytics offset 0.5,0 #format "%.1f" #nomirror
set grid ytics

set xrange [0.01:99];\
eval(step_plot_func("CK"))
set xrange [0.01:15];\
eval(step_plot_func("CR"))

