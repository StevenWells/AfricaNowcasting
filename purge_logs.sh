# Pass in the directory you want to purge as the first argument
DIR=$1
echo "Checking and removing logs in $DIR"
find $DIR -type f -mtime +31 -print -delete
echo "Done"