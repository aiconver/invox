// To Generate
python3 single_shot.py   --dataset pretty_test.json   --examples-file examples.json   --out out_chatgpt.json   --model-id gpt-4o-mini   --examples-per-type 3   --example-max-chars 1400   --pretty --progress --debug


// To evaluate
python3 eval_bleurt_spans.py --gold pretty_test.json --pred out_chatgpt.json \
  --model Elron/bleurt-base-128 --threshold 0.3 --pretty