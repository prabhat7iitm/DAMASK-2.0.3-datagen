module load anaconda3
DAMASK_NUM_THREADS=24
echo 'your number of thread' $DAMASK_NUM_THREADS
seeds_fromRandom -N 64 --grid 100 100 1 > d_1.seeds
geom_fromVoronoiTessellation -g 100 100 1 < d_1.seeds > d_1.geom
#geom_check d_1.geom
