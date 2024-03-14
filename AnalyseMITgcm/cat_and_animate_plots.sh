# Bash script to cat together plots of mean, std into a set of four (subplot for each variable)
# and animating these to scroll through vertical levels. Also to cat together the 4 subplots
# of fields themselves for a specific time point

dir=/data/hpcdata/users/racfur/DynamicPrediction/MITGCM_Analysis_Channel/12hrs/PLOTS
timeslice=32405

for level in {0..37}
do
   convert ${dir}/MeanTemp_z${level}.png ${dir}/MeanUVel_z${level}.png -append ${dir}/tmp1.png
   convert ${dir}/MeanEta.png ${dir}/MeanVVel_z${level}.png -append ${dir}/tmp2.png;
   convert ${dir}/tmp1.png ${dir}/tmp2.png +append ${dir}/Mean_z${level}.png

   convert ${dir}/StdTemp_z${level}.png ${dir}/StdUVel_z${level}.png -append ${dir}/tmp1.png
   convert ${dir}/StdEta.png ${dir}/StdVVel_z${level}.png -append ${dir}/tmp2.png;
   convert ${dir}/tmp1.png ${dir}/tmp2.png +append ${dir}/Std_z${level}.png
done

convert -delay 100 ${dir}/Mean_z?.png ${dir}/Mean_z??.png ${dir}/../Mean.gif
convert -delay 100 ${dir}/Std_z?.png ${dir}/Std_z??.png ${dir}/../Std.gif

convert ${dir}/Temp_z2_time${timeslice}.png ${dir}/UVel_z2_time${timeslice}.png -append ${dir}/tmp1.png
convert ${dir}/Eta_time${timeslice}.png ${dir}/VVel_z2_time${timeslice}.png -append ${dir}/tmp2.png;
convert ${dir}/tmp1.png ${dir}/tmp2.png +append ${dir}/fields_z2_time${timeslice}.png

convert ${dir}/TempTend_z2_time${timeslice}.png ${dir}/UVelTend_z2_time${timeslice}.png -append ${dir}/tmp1.png
convert ${dir}/EtaTend_time${timeslice}.png ${dir}/VVelTend_z2_time${timeslice}.png -append ${dir}/tmp2.png;
convert ${dir}/tmp1.png ${dir}/tmp2.png +append ${dir}/fieldsTend_z2_time${timeslice}.png

rm -f ${dir}/tmp1.png ${dir}/tmp2.png

