module load anaconda3
postResults *.spectralOut --cr f,p,texture,orientation,grainrotation --separation x,y,z --increments --range 10 10 1 --split
postResults *.spectralOut --cr f,p
