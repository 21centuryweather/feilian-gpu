#!/bin/bash

LOGFILE="$1"

if [ -z "$LOGFILE" ]; then
	echo "Usage: $0 <logfile>"
	exit 1
fi

trap  'echo -e "\nMonitoring stopped at $(date)" >> "$LOGFILE"; echo "Exiting..."; exit 0' SIGINT SIGTERM

SAMPLE_RATE_S=1
ROCM_SMI_CMD='rocm-smi --showuse --showtemp --showpower --showenergy --showmemuse --csv'

echo "Monitoring rocm-smi to log file $LOGFILE"
echo "Monitoring starting at $(date)" >> "$LOGFILE"
echo "Monitoring available devices: $ROCR_VISIBLE_DEVICES" >> "$LOGFILE"
echo "Sample rate (s): $SAMPLE_RATE_S" >> "$LOGFILE"  
rocm-smi --showhw >> "$LOGFILE"

echo "Time,$(eval $ROCM_SMI_CMD | head -n 1)" >> "$LOGFILE"
while true; do
  eval $ROCM_SMI_CMD | tail -n +2 | while read -r line; do
    [ -z "$line" ] && continue
    echo "$(date '+%H:%M:%S'),$line"
  done >> "$LOGFILE"
  sleep $SAMPLE_RATE_S
done

