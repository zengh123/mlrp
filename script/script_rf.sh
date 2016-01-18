python handsomeman.py 1
python handsomeman.py 3
python handsomeman.py 5
python handsomeman.py 7
for ((i=10;i<120;i=i+10));do
echo $i
python handsomeman.py $i
done
