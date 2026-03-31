#!/bin/bash

temperature=1.0
sampling="Default"
topK=3000
np=0.8
config_file="config/GlueX_config.json"
echo "Running generations for Kaons."

python generate_GlueX.py --config "$config_file" --sampling "$sampling" --temperature $temperature --topK $topK --nucleus_p $np --method "Kaon" 
echo "------------------------------------------------- "
echo " " 

echo "Running generations for Pions."
python generate_GlueX.py --config "$config_file" --sampling "$sampling" --temperature $temperature --topK $topK --nucleus_p $np --method "Pion" 
echo "------------------------------------------------- "
echo " " 

