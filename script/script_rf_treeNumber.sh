python handsomeman.py 1
python handsomeman.py 5
python handsomeman.py 10
python handsomeman.py 25
for ((i=50;i<500;i=i+50));do
echo $i
python handsomeman.py $i
done
