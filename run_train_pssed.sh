# run python train_hybnet.py 10 times
#!/bin/bash
for i in {1..10}
do
    echo "Run $i"
    python train_PSSED.py
done
