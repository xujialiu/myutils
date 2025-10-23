for pid in $(pstree -p 888888 | grep -o '([0-9]\+)' | tr -d '()'); do 
    kill -STOP $pid
done

for pid in $(pstree -p 888888 | grep -o '([0-9]\+)' | tr -d '()'); do 
    kill -CONT $pid
done