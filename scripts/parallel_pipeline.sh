#!/bin/bash
start_time=$(date +%s)
echo "Starting full mapping run at $(date)" | tee parallel_timings.log
echo "----------------------------------------" | tee -a parallel_timings.log
cat debug_acc.txt | xargs -n 1 -P 4 -I {} bash -c '
    accession={}
    echo "Processing $accession..."
    ../minimap2/minimap2 -x asm20 -t 3 -a \
        <(/fs/nexus-scratch/ryansynk/aws-bin/aws s3 cp s3://logan-pub/c/$accession/$accession.contigs.fa.zst - --no-sign-request | zstdcat) \
        ./data/dataset/gencode.v49.transcripts.fa \
        | awk '$10 / $11 > 0.9 && $12 >= 30 {print $1"\t"$6}' \
        > mapping/${acc}_pairs.txt
'
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "----------------------------------------" | tee -a parallel_timings.log
echo "Total runtime: ${elapsed}s" | tee -a parallel_timings.log
echo "Completed at $(date)" | tee -a parallel_timings.log

