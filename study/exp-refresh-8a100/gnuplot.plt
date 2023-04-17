outputfname = "figure.svg"
dat_file='data.dat'

col_app                          = 1
col_cache_policy                 = 2
col_cache_percent                = 3
col_batch_size                   = 4
col_dataset                      = 5

col_seq                          = 6
col_seq_feat_copy                = 7
col_e2e_time                     = 8
col_seq_duration                 = 9
col_bucket_size                  = 10
col_enable_refresh               = 11
col_python_e2e_time              = 12

batch_size = "65536"
fig_col_num = 2
fig_row_num = 2

set datafile sep '\t'
set output outputfname

# set terminal svg "Helvetica,16" enhance color dl 2 background rgb "white"
set terminal svg size 700,1800 font "Helvetica,16" enhanced background rgb "white" dl 2
set multiplot layout 6,2

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
                              "$".col_app."             ~ \"%s\"     && ". \
                              "$".col_dataset."         ~ \"%s\"     && ". \
                              "$".col_batch_size."      ~ \"%s\"     && ". \
                              "$".col_cache_percent."   ~ \"%s\"     && ". \
                              "$".col_cache_policy."    ~ \"%s\"     && ". \
                              "$".col_enable_refresh."  ~ \"%s\"     ". \
                              ") { print }}' ".dat_file." "
cmd_filter_dat_by_policy(app, dataset, batch_size, cache_percent, policy, refresh)=sprintf(format_str, app, dataset, batch_size, cache_percent, policy, refresh)
##########################################################################################

### Key
set key inside left Left reverse top enhanced box 
set key samplen 1 spacing 1 height 0.2 width -2 font ',13' center at graph 0.65, graph 0.6 noopaque

set xlabel "Cache Rate" offset 0,1.7 font ",13"
# set xtics 0,4,24 
set xrange [0:]
# set xtics rotate by -45
set xtics nomirror offset 0,0.7
set grid xtics

## Y-axis
set ylabel "Copy Time(ms)" offset 3.5,0 font ",13"
set yrange [0:]
# set ytics 0,20,100 
set ytics offset 0.5,0 #format "%.1f" #nomirror

set grid ytics

# plot cmd_filter_dat_by_policy(".*", "CRU", ".*", "Rep") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_step_time_feat_copy)*1000) w lp ps 0.5 lw 1 lc 3 title "Rep" \
#     ,cmd_filter_dat_by_policy(".*", "CRU", ".*", "CliqPart") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_step_time_feat_copy)*1000) w lp ps 0.5 lw 1 lc 2 title "CliqPart" \
#     ,cmd_filter_dat_by_policy(".*", "CRU", ".*", "CollAsymm") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_step_time_feat_copy)*1000) w lp ps 0.5 lw 1 lc 1 title "CollAsymm" \
#     ,cmd_filter_dat_by_policy(".*", "CRU", ".*", "CollAsymm") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):((column(col_step_time_train_total) - column(col_step_time_feat_copy))*1000) w lp ps 0.5 lw 1 lc "black" title "APP" \

# plot cmd_filter_dat_by_policy(".*", "SP_01_S100", ".*", "Rep") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_step_time_feat_copy)*1000) w lp ps 0.5 lw 1 lc 3 title "Rep" \
#     ,cmd_filter_dat_by_policy(".*", "SP_01_S100", ".*", "CliqPart") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_step_time_feat_copy)*1000) w lp ps 0.5 lw 1 lc 2 title "CliqPart" \
#     ,cmd_filter_dat_by_policy(".*", "SP_01_S100", ".*", "CollAsymm") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):(column(col_step_time_feat_copy)*1000) w lp ps 0.5 lw 1 lc 1 title "CollAsymm" \
#     ,cmd_filter_dat_by_policy(".*", "SP_01_S100", ".*", "CollAsymm") using (column(col_cache_percent) > 0 ? column(col_cache_percent) : 1/0):((column(col_step_time_train_total) - column(col_step_time_feat_copy))*1000) w lp ps 0.5 lw 1 lc "black" title "APP" \


cache_percent_lb=0

app_str = "dlrm dlrm dcn dcn"
ds_str = "SP_02_S100_C800m CR SP_02_S100_C800m CR"
bs_str = ".* .*"
policy="MPSPhaseCollAsymm"


app="dlrm"
ds="CR"
bs=".*"
refresh_on="True"
refresh_off="False"
do for [cache_percent in "1.000000"] {
set xlabel "Seq bucket" offset 0,1.7 font ",13"
set ylabel "Copy Time(ms)" offset 3.5,0 font ",13"
set title app." ".ds." ".bs." ".cache_percent offset 0,-1
plot cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_on) using (column(col_seq)):(column(col_seq_feat_copy)*1000) w lp ps 0.5 lw 1 lc 3 title "refresh" \
    ,cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_off) using (column(col_seq)):(column(col_seq_feat_copy)*1000) w lp ps 0.5 lw 1 lc 1 title "normal"
set ylabel "E2E Time(ms)" offset 3.5,0 font ",13"
set title app." ".ds." ".bs." ".cache_percent offset 0,-1
plot cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_on) using (column(col_seq)):(column(col_e2e_time)*1000) w lp ps 0.5 lw 1 lc 3 title "refresh" \
    ,cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_off) using (column(col_seq)):(column(col_e2e_time)*1000) w lp ps 0.5 lw 1 lc 1 title "normal" \
    # ,cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_on) using (column(col_seq)):(column(col_python_e2e_time)*1000) w lp ps 0.5 lw 1 lc 1 title "refresh-python" \
    # ,cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_off) using (column(col_seq)):(column(col_python_e2e_time)*1000) w lp ps 0.5 lw 1 lc 1 title "normal-python"

set xlabel "time(s)" offset 0,1.7 font ",13"
set ylabel "Copy Time(ms)" offset 3.5,0 font ",13"
set title app." ".ds." ".bs." ".cache_percent offset 0,-1
plot cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_on) using (column(col_seq_duration)):(column(col_seq_feat_copy)*1000) w lp ps 0.5 lw 1 lc 3 title "refresh" \
    ,cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_off) using (column(col_seq_duration)):(column(col_seq_feat_copy)*1000) w lp ps 0.5 lw 1 lc 1 title "normal"
set ylabel "E2E Time(ms)" offset 3.5,0 font ",13"
set grid x2tics
set x2tics 10 format "" scale 0
set title app." ".ds." ".bs." ".cache_percent offset 0,-1
plot cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_on) using (column(col_seq_duration)):(column(col_e2e_time)*1000) w lp ps 0.5 lw 1 lc 3 title "refresh" \
    ,cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_off) using (column(col_seq_duration)):(column(col_e2e_time)*1000) w lp ps 0.5 lw 1 lc 1 title "normal" \
    # ,cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_on) using (column(col_seq_duration)):(column(col_python_e2e_time)*1000) w lp ps 0.5 lw 1 lc 1 title "refresh-python" \
    # ,cmd_filter_dat_by_policy(app, ds, bs, cache_percent, policy, refresh_off) using (column(col_seq_duration)):(column(col_python_e2e_time)*1000) w lp ps 0.5 lw 1 lc 1 title "normal-python"
}


# do for [i=1:words(app_str)]  {
# app = word(app_str, i)
# ds = word(ds_str, i)
# bs = word(bs_str, i)
# set title app." ".ds." ".bs offset 0,-1
# # print(cmd_filter_dat_by_policy(app, ds, bs, "^Cliq"))
# plot cmd_filter_dat_by_policy(app, ds, bs, "1.000000", policy) using (column(col_seq)):(column(col_seq_feat_copy)*1000) w lp ps 0.5 lw 1 lc 3 title "0.08" \
# }

# set ylabel "E2E Time(ms)" offset 3.5,0 font ",13"
# do for [i=1:words(app_str)]  {
# app = word(app_str, i)
# ds = word(ds_str, i)
# bs = word(bs_str, i)
# set title app." ".ds." ".bs offset 0,-1
# # print(cmd_filter_dat_by_policy(app, ds, bs, "^Cliq"))
# plot cmd_filter_dat_by_policy(app, ds, bs, "1.000000", policy) using (column(col_seq)):(column(col_e2e_time)*1000) w lp ps 0.5 lw 1 lc 3 title "0.08" \
# }


# do for [i=1:words(app_str)]  {
# app = word(app_str, i)
# ds = word(ds_str, i)
# bs = word(bs_str, i)
# set title app." ".ds." ".bs
# print(cmd_filter_dat_by_policy(app, ds, bs, "^Cliq"))
# plot cmd_filter_dat_by_policy(app, ds, bs, "^Rep") using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_z)) w lp ps 0.5 lw 1 lc 3 title "Rep" \
#     ,cmd_filter_dat_by_policy(app, ds, bs, "MPSRep") using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_z)) w l lw 1 lc 3 title "MPSRep" \
#     ,cmd_filter_dat_by_policy(app, ds, bs, "^Cliq") using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_z)) w lp ps 0.5 lw 1 lc 2 title "CliqPart" \
#     ,cmd_filter_dat_by_policy(app, ds, bs, "MPSCliq") using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_z)) w l lw 1 lc 2 title "MPSCliqPart" \
#     ,cmd_filter_dat_by_policy(app, ds, bs, "MPSColl") using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_z)) w lp ps 0.5 lw 1 lc 1 title "CollAsymm" \
# }
