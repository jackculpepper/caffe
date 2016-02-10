
output=train_test.prototxt
lambda=0.5

echo >$output
cat data.prototxt >>$output
cat top.prototxt | sed "s/WWW/w0/g" \
                 | sed "s/ZZZ/b0/g" >>$output

max=10
x=0
y=$((x+1))
while [ $x -lt $max ]
do
    xxx=`printf "%03d" $x`
    yyy=`printf "%03d" $y`

    cat module.prototxt | sed "s/XXX/$xxx/g" \
                        | sed "s/WWW/w1/g" \
                        | sed "s/ZZZ/b1/g" \
                        | sed "s/YYY/$yyy/g" >>$output

    x=$((x+1))
    y=$((y+1))
done

x=2
while [ $x -lt $max ]
do
    xxx=`printf "%03d" $x`

    cat watch.prototxt | sed "s/XXX/$xxx/g" \
                       | sed "s/WWW/w2/g" \
                       | sed "s/UUU/ip0/g" \
                       | sed "s/ZZZ/b2/g" >>$output

    x=$((x+2))
done

xxx=`printf "%03d" $max`
cat bottom.prototxt | sed "s/XXX/$xxx/g" \
                    | sed "s/LAMBDA/$lambda/g" \
                    | sed "s/WWW/w2/g" \
                    | sed "s/ZZZ/b2/g" \
                    | sed "s/YYY/$yyy/g" >>$output


#cat log.prototxt | sed "s/XXX/$xxx/g" >>$output

