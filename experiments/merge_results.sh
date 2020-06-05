OUTPUT=../results
CSV_FILENAME=results.csv

FILES=($(ls -d ${OUTPUT}/*/${CSV_FILENAME}))

header=$(head -1 "${FILES[0]}")

echo $header > ${OUTPUT}/${CSV_FILENAME}_tmp
for file in ${FILES[@]}
do
    tail +2 ${file} >> ${OUTPUT}/${CSV_FILENAME}_tmp
done

cut -d";" -f2- ${OUTPUT}/${CSV_FILENAME}_tmp > ${OUTPUT}/${CSV_FILENAME}
rm -f ${OUTPUT}/${CSV_FILENAME}_tmp

echo $'\u221a' Successfully merged ${#FILES[@]} files into ${OUTPUT}/${CSV_FILENAME} 
