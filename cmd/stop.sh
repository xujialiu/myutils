for pid in $(pstree -p 888888 | grep -o '([0-9]\+)' | tr -d '()'); do 
    kill -STOP $pid # -CONT to resume
done


for pid in $(ps -u user -o pid --no-headers); do 
    kill -STOP $pid  # -CONT to resume
done